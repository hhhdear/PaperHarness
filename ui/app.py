"""Streamlit 展示端：改写控制台 / 流程 Trace 可视化 / Diff 对比 / 记忆库与成本看板。

启动：streamlit run ui/app.py
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config import get_settings
from harness.engine import RewriterEngine, build_llm_chain
from harness.trace import read_trace
from memory.db import MemoryDB
from memory.retriever import MemoryRetriever
from schemas import ExperienceItem, PreferenceItem, RewriteRequest, TermItem

st.set_page_config(page_title="学术论文智能改写 Agent Harness", page_icon="📝", layout="wide")

STAGE_ORDER = ["guardrails", "diagnose", "plan", "rewrite", "verify", "reflection", "deliver"]
STAGE_LABEL = {
    "guardrails": "① Guardrails 输入护栏",
    "diagnose": "② Diagnose 段落诊断",
    "plan": "③ Plan 策略规划",
    "rewrite": "④ Rewrite 分段改写",
    "verify": "⑤ Verify 四层校验",
    "reflection": "↻ Reflection 定向返工",
    "deliver": "⑥ Deliver 交付汇总",
}
STATUS_ICON = {"ok": "🟢", "failed": "🔴", "blocked": "🟡", "degraded": "🟠"}


@st.cache_resource
def get_context():
    settings = get_settings()
    db = MemoryDB(settings.db_path, seed_dir=settings.data_dir)
    db.init()
    retriever = MemoryRetriever(db, settings)
    return settings, db, retriever


def load_examples():
    path = ROOT / "data" / "examples.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_rewrite(text: str, mode: str, extra_terms: list[str], max_rounds: int, llm_cfg: dict):
    settings, db, retriever = get_context()
    overrides = {k: v for k, v in llm_cfg.items() if v}
    if overrides:
        settings = settings.override(**overrides)
    engine = RewriterEngine(settings, db, retriever, build_llm_chain(settings, mode))
    req = RewriteRequest(
        text=text, mode=mode, extra_locked_terms=extra_terms, max_reflection_rounds=max_rounds
    )
    return engine.run(req)


def sentence_lines(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；])|(?<=[.!?])\s+", text.strip())
    return [p for p in (s.strip() for s in parts) if p]


def diff_html(a: str, b: str) -> str:
    table = difflib.HtmlDiff(wrapcolumn=60).make_table(
        sentence_lines(a), sentence_lines(b),
        fromdesc="原文", todesc="改写后", context=False,
    )
    css = """
    <style>
    body { font-family: 'Segoe UI', sans-serif; font-size: 13px; }
    table.diff { width: 100%; border-collapse: collapse; }
    .diff_header { background:#444; color:#fff; padding:4px; }
    td { padding: 4px 6px; vertical-align: top; }
    .diff_add { background:#d4f7d4; }
    .diff_chg { background:#fff3c4; }
    .diff_sub { background:#ffd6d6; }
    </style>
    """
    return css + table


# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
settings, db, retriever = get_context()
with st.sidebar:
    st.title("⚙️ Harness 控制台")
    mode = st.radio(
        "运行模式",
        ["mock", "real"],
        index=0 if settings.llm_mode == "mock" else 1,
        format_func=lambda m: "🎭 Mock 演示模式（无 Key 可跑通）" if m == "mock" else "🤖 真实 LLM 模式（自动降级链）",
    )
    with st.expander("🔑 LLM 配置（real 模式生效）", expanded=False):
        llm_key = st.text_input("OPENAI_API_KEY", value="", type="password",
                                help="留空则读取 .env / 环境变量")
        llm_base = st.text_input("OPENAI_BASE_URL", value=settings.openai_base_url or "")
        llm_model = st.text_input("OPENAI_MODEL", value=settings.openai_model)
        st.caption(
            f"当前生效：model=`{settings.openai_model}`，"
            f"base_url=`{settings.openai_base_url}`，"
            f"key={'✅ 已配置' if settings.openai_api_key else '❌ 未配置'}"
        )
        st.caption("持久配置：复制 .env.example 为 .env 后填写，重启生效")
    max_rounds = st.slider("最大 Reflection 返工轮次", 0, 4, settings.max_reflection_rounds)
    st.divider()
    st.caption("Embedding 检索后端")
    st.info(f"`{retriever.backend}`：{retriever.backend_note}")
    st.caption("校验阈值")
    st.json(
        {
            "长度容差": settings.length_tolerance,
            "语义相似度阈值": settings.semantic_sim_threshold,
            "LLM机械度上限": settings.llm_mech_max,
            "LLM保真度下限": settings.llm_fidelity_min,
        },
        expanded=False,
    )

st.title("📝 基于自研 Agent Harness 的学术论文智能改写系统")
st.caption(
    "六阶段状态机 `Guardrails → Diagnose → Plan → Rewrite → Verify → Deliver` "
    "+ Reflection 定向返工闭环 ｜ RAG + SQLite 跨会话记忆 ｜ Turn/Stage 双级 Trace"
)

tab_run, tab_trace, tab_diff, tab_mem = st.tabs(
    ["✍️ 改写控制台", "🔍 流程状态与 Trace", "🆚 原文 vs 改写 Diff", "🧠 记忆库与成本看板"]
)

# --------------------------------------------------------------------------- #
# Tab 1: 改写控制台
# --------------------------------------------------------------------------- #
with tab_run:
    examples = load_examples()
    c1, c2, c3 = st.columns([1, 1, 4])
    if c1.button("📄 填入中文示例", use_container_width=True):
        st.session_state["input_text"] = "\n\n".join(examples.get("zh", {}).get("paragraphs", []))
    if c2.button("📄 填入英文示例", use_container_width=True):
        st.session_state["input_text"] = "\n\n".join(examples.get("en", {}).get("paragraphs", []))
    if c3.button("🧹 清空", use_container_width=True):
        st.session_state["input_text"] = ""

    text = st.text_area(
        "论文段落输入（空行分段）",
        key="input_text",
        height=260,
        placeholder="粘贴需要改写的论文段落，空行分隔多段……",
    )
    extra_terms_raw = st.text_input("附加锁定术语（逗号分隔，改写时逐字保留）", value="")
    extra_terms = [t.strip() for t in re.split(r"[,，]", extra_terms_raw) if t.strip()]

    if st.button("🚀 执行 Harness 改写", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("请先输入或填入示例段落。")
        else:
            llm_cfg = {
                "openai_api_key": llm_key,
                "openai_base_url": llm_base,
                "openai_model": llm_model,
            }
            with st.spinner("六阶段流水线执行中（含 Reflection 闭环）……"):
                state = run_rewrite(text, mode, extra_terms, max_rounds, llm_cfg)
            st.session_state["last_state"] = state.model_dump(mode="json")
            st.session_state["last_session"] = state.session_id

    last = st.session_state.get("last_state")
    if last:
        status = last["status"]
        color = {"delivered": "🟢", "partial": "🟠", "blocked": "🟡", "error": "🔴"}.get(status, "⚪")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("运行状态", f"{color} {status}")
        m2.metric("Reflection 返工", f"{last['reflection_count']} 轮")
        m3.metric("总 Token", last["total_usage"]["total_tokens"])
        m4.metric("估算成本", f"${last['total_cost']:.5f}")
        m5.metric("总耗时", f"{last['total_latency_ms']:.0f} ms")
        if last.get("error"):
            st.error(last["error"])
        for note in last.get("degradation_notes", []):
            st.warning(f"降级链事件：{note}")

        st.subheader("各段落诊断与改写结果")
        for r in last["results"]:
            diag = last["diagnoses"].get(r["paragraph_id"], {})
            plan = last["plans"].get(r["paragraph_id"], {})
            icon = "✅" if r["passed"] else "⚠️"
            with st.expander(
                f"{icon} {r['paragraph_id']} ｜ 尝试 {r['attempts']} 轮 ｜ {diag.get('summary', '')}",
                expanded=True,
            ):
                st.markdown(f"**诊断来源**：`{diag.get('source', 'llm')}`")
                st.markdown(f"**策略组合**：{'、'.join(plan.get('strategies', []))}")
                st.markdown(f"**诊断摘要**：{diag.get('summary', '')}")
                for issue in diag.get("issues", []):
                    st.markdown(f"- `{issue['severity']}` {issue['detail']}：{', '.join(issue.get('evidence', []))}")
                st.markdown("**最终改写**：")
                st.success(r["final_text"])
                last_report = r["verify_reports"][-1] if r["verify_reports"] else None
                if last_report:
                    st.markdown("**四层校验（最终轮）**：")
                    for c in last_report["checks"]:
                        st.markdown(f"- {'✅' if c['passed'] else '❌'} `{c['name']}`：{c['detail']}")

# --------------------------------------------------------------------------- #
# Tab 2: Trace 可视化
# --------------------------------------------------------------------------- #
with tab_trace:
    sid = st.session_state.get("last_session")
    sid_input = st.text_input("会话 ID（默认展示最近一次运行）", value=sid or "")
    events = read_trace(settings.runs_dir, sid_input) if sid_input else None
    if not events:
        st.info("暂无 Trace。请先在「改写控制台」执行一次改写。")
    else:
        rows = []
        for e in events:
            rows.append(
                {
                    "阶段": STAGE_LABEL.get(e.stage.value, e.stage.value),
                    "Turn": e.turn_id,
                    "状态": f"{STATUS_ICON.get(e.status.value, '⚪')} {e.status.value}",
                    "耗时(ms)": e.latency_ms,
                    "Token": e.token_usage.total_tokens,
                    "输入摘要": e.input_summary,
                    "输出摘要": e.output_summary,
                    "失败/降级原因": e.error_or_reason,
                }
            )
        df = pd.DataFrame(rows)

        st.subheader("六阶段流转时间线（Turn/Stage 双级）")
        cols = st.columns(len(df))
        for col, (_, row) in zip(cols, df.iterrows()):
            col.markdown(
                f"<div style='border:1px solid #666;border-radius:8px;padding:8px;text-align:center'>"
                f"<b>{row['阶段']}</b><br/>Turn {row['Turn']}<br/>{row['状态']}<br/>"
                f"{row['耗时(ms)']:.0f} ms · {row['Token']} tok</div>",
                unsafe_allow_html=True,
            )

        st.subheader("各阶段耗时 / Token 分布")
        chart_df = df[["阶段", "耗时(ms)", "Token"]].set_index("阶段")
        c1, c2 = st.columns(2)
        c1.bar_chart(chart_df[["耗时(ms)"]])
        c2.bar_chart(chart_df[["Token"]])

        st.subheader("JSONL Trace 明细")
        st.dataframe(df, use_container_width=True, hide_index=True)
        trace_file = settings.runs_dir / f"trace_{sid_input}.jsonl"
        st.caption(f"落盘文件：`{trace_file}`")
        st.download_button(
            "⬇️ 下载 Trace JSONL",
            data=trace_file.read_text(encoding="utf-8"),
            file_name=trace_file.name,
            mime="application/jsonl",
        )

# --------------------------------------------------------------------------- #
# Tab 3: Diff 对比
# --------------------------------------------------------------------------- #
with tab_diff:
    if not last:
        st.info("暂无运行结果。请先执行一次改写。")
    else:
        results = last["results"]
        pid = st.selectbox("选择段落", [r["paragraph_id"] for r in results])
        r = next(x for x in results if x["paragraph_id"] == pid)
        version_labels = [f"第 {v['attempt']} 轮" for v in r["versions"]]
        vidx = st.selectbox("对比版本", range(len(version_labels)),
                            format_func=lambda i: version_labels[i],
                            index=len(version_labels) - 1)
        chosen = r["versions"][vidx]
        report = next((rep for rep in r["verify_reports"] if rep["attempt"] == chosen["attempt"]), None)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**原文**")
            st.text_area("原文", r["original"], height=200, disabled=True, label_visibility="collapsed")
        with c2:
            st.markdown(f"**改写（第 {chosen['attempt']} 轮）**")
            st.text_area("改写", chosen["text"], height=200, disabled=True, label_visibility="collapsed")
        if report:
            badge = "✅ 通过" if report["passed"] else "❌ 打回"
            st.markdown(
                f"本轮校验：**{badge}** ｜ 机械度 {report.get('llm_mechanical_score')} ｜ "
                f"保真度 {report.get('llm_fidelity_score')} ｜ {report.get('llm_comments', '')}"
            )
            if report.get("suggestions"):
                st.markdown("返工意见：" + "；".join(report["suggestions"]))
        st.subheader("高亮差异（句级 Diff）")
        components.html(diff_html(r["original"], chosen["text"]), height=420, scrolling=True)

# --------------------------------------------------------------------------- #
# Tab 4: 记忆库与成本看板
# --------------------------------------------------------------------------- #
with tab_mem:
    st.subheader("📊 历史运行成本看板")
    summary = db.metrics_summary()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("累计运行", summary["total_runs"])
    k2.metric("校验通过率", f"{summary['pass_rate']:.0%}")
    k3.metric("平均返工轮次", f"{summary['avg_reflection_rounds']:.2f}")
    k4.metric("累计 Token", int(summary["total_tokens"]))
    k5.metric("累计成本", f"${summary['total_cost']:.5f}")
    runs = db.list_runs(limit=30)
    if runs:
        st.dataframe(pd.DataFrame(runs), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🧠 SQLite 记忆库管理")
    mt1, mt2, mt3 = st.tabs(["学术术语库", "用户改写偏好", "历史改写经验"])

    with mt1:
        terms = db.list_terms()
        st.dataframe(
            pd.DataFrame([t.model_dump() for t in terms]), use_container_width=True, hide_index=True
        )
        with st.form("add_term"):
            c1, c2, c3, c4 = st.columns(4)
            t_term = c1.text_input("术语")
            t_std = c2.text_input("标准写法（可同原词）")
            t_lang = c3.selectbox("语言", ["zh", "en"])
            t_locked = c4.checkbox("强制锁定", value=True)
            if st.form_submit_button("➕ 新增术语") and t_term:
                db.add_term(TermItem(term=t_term, standard_form=t_std or t_term,
                                     locked=t_locked, lang=t_lang))
                retriever.invalidate()
                st.success(f"已新增术语「{t_term}」")
                st.rerun()
        del_id = st.number_input("删除术语 ID", min_value=0, step=1, key="del_term")
        if st.button("🗑️ 删除术语") and del_id:
            db.delete_term(int(del_id))
            retriever.invalidate()
            st.rerun()

    with mt2:
        prefs = db.list_preferences()
        st.dataframe(
            pd.DataFrame([p.model_dump() for p in prefs]), use_container_width=True, hide_index=True
        )
        with st.form("add_pref"):
            c1, c2 = st.columns(2)
            p_key = c1.text_input("偏好键（如 tone / avoid_first_person）")
            p_val = c2.text_input("偏好值")
            if st.form_submit_button("➕ 设置偏好") and p_key:
                db.set_preference(PreferenceItem(key=p_key, value=p_val))
                st.success(f"已设置偏好 {p_key}={p_val}")
                st.rerun()

    with mt3:
        exps = db.list_experiences()
        for e in exps:
            with st.expander(f"[{e.id}] {e.title}（质量 {e.quality}）"):
                st.markdown(f"**原文**：{e.source_text}")
                st.markdown(f"**改写**：{e.rewritten_text}")
                st.caption(f"tags: {e.tags} ｜ {e.note}")
        with st.form("add_exp"):
            e_title = st.text_input("经验标题")
            e_src = st.text_area("原文示例")
            e_dst = st.text_area("高质量改写示例")
            e_tags = st.text_input("标签（空格分隔）")
            if st.form_submit_button("➕ 新增经验") and e_src and e_dst:
                db.add_experience(ExperienceItem(title=e_title, source_text=e_src,
                                                 rewritten_text=e_dst, tags=e_tags))
                retriever.invalidate()
                st.success("已新增经验")
                st.rerun()
