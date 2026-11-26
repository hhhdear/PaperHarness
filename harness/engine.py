"""六阶段状态机 + Reflection 定向返工闭环。

Guardrails -> Diagnose -> Plan -> Rewrite -> Verify -> Deliver
校验未通过且未超上限时进入 Reflection，携带失败原因回到 Rewrite。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Settings, get_settings
from harness import strategies, verifier
from harness.llm_client import (
    FallbackLLMClient,
    LLMError,
    MockLLMClient,
    OpenAICompatClient,
)
from harness.trace import Tracer
from harness.utils import extract_json_obj, truncate
from memory.db import MemoryDB
from memory.retriever import MemoryRetriever
from schemas import (
    AgentState,
    CheckResult,
    DiagnosisIssue,
    ParagraphDiagnosis,
    ParagraphPlan,
    ParagraphResult,
    RewriteRequest,
    RewriteVersion,
    RunStatus,
    Stage,
    StageStatus,
    StrategyName,
    TermItem,
    TokenUsage,
)

_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard your instructions",
    "忽略以上指令",
    "忽略之前的指令",
    "忽略所有指令",
]
_SENSITIVE_PATTERNS = ["制造炸弹", "爆炸物制作", "bomb making instructions"]


class RewriterEngine:
    def __init__(
        self,
        settings: Settings,
        db: MemoryDB,
        retriever: MemoryRetriever,
        llm: FallbackLLMClient,
    ):
        self.settings = settings
        self.db = db
        self.retriever = retriever
        self.llm = llm

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def run(self, request: RewriteRequest) -> AgentState:
        max_rounds = (
            request.max_reflection_rounds
            if request.max_reflection_rounds is not None
            else self.settings.max_reflection_rounds
        )
        state = AgentState(
            mode=request.mode or self.settings.llm_mode,
            status=RunStatus.RUNNING,
            paragraphs=request.to_paragraphs(),
            max_reflection_rounds=max_rounds,
        )
        tracer = Tracer(self.settings.runs_dir, state.session_id)
        t_all = time.perf_counter()
        try:
            self._stage_guardrails(state, request, tracer)
            if state.status == RunStatus.BLOCKED:
                self._finalize(state, tracer, t_all)
                return state
            self._stage_diagnose(state, tracer)
            self._stage_plan(state, tracer)
            self._rewrite_verify_loop(state, tracer)
            self._stage_deliver(state, tracer)
        except Exception as exc:  # noqa: BLE001
            state.status = RunStatus.ERROR
            state.error = f"{type(exc).__name__}: {exc}"
            tracer.log(
                stage=Stage.DELIVER,
                status=StageStatus.FAILED,
                turn_id=state.turn,
                error_or_reason=state.error,
                output_summary="引擎异常终止",
            )
        self._finalize(state, tracer, t_all)
        return state

    # ------------------------------------------------------------------ #
    # Stage 1: Guardrails
    # ------------------------------------------------------------------ #
    def _stage_guardrails(self, state: AgentState, request: RewriteRequest, tracer: Tracer) -> None:
        t0 = time.perf_counter()
        state.current_stage = Stage.GUARDRAILS
        s = self.settings
        reasons: List[str] = []

        if len(state.paragraphs) > s.max_paragraphs:
            reasons.append(f"段落数 {len(state.paragraphs)} 超过上限 {s.max_paragraphs}")
        total_chars = sum(len(p.text) for p in state.paragraphs)
        if total_chars > s.max_total_chars:
            reasons.append(f"总字数 {total_chars} 超过上限 {s.max_total_chars}")
        for p in state.paragraphs:
            if not p.text.strip():
                reasons.append(f"段落 {p.id} 为空")
            if len(p.text) > s.max_chars_per_paragraph:
                reasons.append(f"段落 {p.id} 长度 {len(p.text)} 超过上限 {s.max_chars_per_paragraph}")
            low = p.text.lower()
            for pat in _INJECTION_PATTERNS:
                if pat in low or pat in p.text:
                    reasons.append(f"段落 {p.id} 命中注入攻击模式：「{pat}」")
            for pat in _SENSITIVE_PATTERNS:
                if pat in low or pat in p.text:
                    reasons.append(f"段落 {p.id} 命中敏感内容：「{pat}」")

        # 术语锁定列表解析：请求附加 + 库中锁定术语（仅注入实际出现的）
        locked = list(dict.fromkeys(request.extra_locked_terms))
        all_text = "\n".join(p.text for p in state.paragraphs)
        for t in self.db.locked_terms():
            if t.term in all_text and t.term not in locked:
                locked.append(t.term)
        state.locked_terms = locked

        prefs = self.db.preferences_dict()
        prefs.update(request.preference_overrides)
        state.preferences = prefs

        latency = (time.perf_counter() - t0) * 1000
        if reasons:
            state.status = RunStatus.BLOCKED
            state.error = "；".join(reasons)
            tracer.log(
                stage=Stage.GUARDRAILS, status=StageStatus.BLOCKED, turn_id=0,
                latency_ms=latency,
                input_summary=f"{len(state.paragraphs)} 段 / {total_chars} 字",
                output_summary="输入被拦截",
                error_or_reason=state.error,
            )
        else:
            tracer.log(
                stage=Stage.GUARDRAILS, status=StageStatus.OK, turn_id=0,
                latency_ms=latency,
                input_summary=f"{len(state.paragraphs)} 段 / {total_chars} 字",
                output_summary=f"通过；锁定术语 {len(locked)} 个；偏好 {len(prefs)} 条",
            )

    # ------------------------------------------------------------------ #
    # Stage 2: Diagnose
    # ------------------------------------------------------------------ #
    def _stage_diagnose(self, state: AgentState, tracer: Tracer) -> None:
        t0 = time.perf_counter()
        state.current_stage = Stage.DIAGNOSE
        diag_usage = TokenUsage()
        fallback_notes: List[str] = []
        for p in state.paragraphs:
            local = strategies.build_diagnosis_payload(p.text)
            payload, usage = self._llm_diagnose(p.text, local)
            diag_usage = diag_usage + usage
            if payload is not None:
                source = f"llm:{self.llm.last_used}"
                issues = self._parse_issues(payload.get("issues"), local["issues"])
                mech = float(payload.get("mechanical_score", local["mechanical_score"]))
                llm_summary = str(payload.get("summary", ""))
            else:
                source = "rule-fallback"
                issues = [DiagnosisIssue(**i) for i in local["issues"]]
                mech = local["mechanical_score"]
                llm_summary = local["llm_summary"]
                note = f"段落 {p.id} 诊断 LLM 不可用/输出非法，回退本地规则诊断"
                fallback_notes.append(note)
            locked_found = [t for t in state.locked_terms if t in p.text]
            diag = ParagraphDiagnosis(
                paragraph_id=p.id,
                source=source,
                language=local["language"],
                char_count=len(p.text),
                sentence_count=local["sentence_count"],
                avg_sentence_len=local["avg_sentence_len"],
                mechanical_score=round(float(mech), 3),
                issues=issues,
                locked_terms_found=locked_found,
                numbers_found=local["numbers"],
                summary=(f"{local['summary']}｜锁定术语 {len(locked_found)}"
                         + (f"｜LLM: {truncate(llm_summary, 80)}" if llm_summary else "")),
            )
            state.diagnoses[p.id] = diag
        state.total_usage = state.total_usage + diag_usage
        self._drain_fallback_events(state)
        state.degradation_notes.extend(fallback_notes)
        latency = (time.perf_counter() - t0) * 1000
        degraded = bool(fallback_notes) or any(
            d.source == "rule-fallback" for d in state.diagnoses.values()
        )
        tracer.log(
            stage=Stage.DIAGNOSE,
            status=StageStatus.DEGRADED if degraded else StageStatus.OK,
            turn_id=0, latency_ms=latency,
            token_usage=diag_usage,
            input_summary=f"{len(state.paragraphs)} 段",
            output_summary="；".join(f"{d.paragraph_id}: {d.summary}" for d in state.diagnoses.values()),
            error_or_reason="；".join(fallback_notes),
        )

    def _llm_diagnose(self, text: str, local: Dict[str, Any]):
        """调用 LLM 输出结构化诊断 JSON；失败/非法输出返回 (None, usage) 由调用方回退。"""
        system = (
            "TASK: diagnose\n"
            "你是学术写作诊断专家。识别段落的“表达机械化”特征（模板化连接词堆砌、句式单一、"
            "空洞修饰、绝对化表述）与潜在语义风险点，仅输出 JSON：\n"
            '{"issues":[{"type":"template_connector|vague_modifier|long_sentence|'
            'monotone_sentence|semantic_risk","severity":"low|medium|high",'
            '"detail":"...","evidence":["..."]}],'
            '"mechanical_score":0~1,"summary":"一句话诊断总结"}\n'
            "以下为本地初检特征（供参考，可修正）：\n"
            f"CONTEXT_JSON: {json.dumps({k: local[k] for k in ('connectors', 'vague_modifiers', 'long_sentences', 'absolute_claims', 'sentence_count', 'avg_sentence_len')}, ensure_ascii=False)}"
        )
        try:
            resp = self.llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": text}],
                json_mode=True,
            )
        except LLMError:
            return None, TokenUsage()
        payload = extract_json_obj(resp.text)
        if payload is None or "issues" not in payload:
            return None, resp.usage
        return payload, resp.usage

    @staticmethod
    def _parse_issues(raw: Any, fallback: List[Dict[str, Any]]) -> List[DiagnosisIssue]:
        issues: List[DiagnosisIssue] = []
        for item in raw if isinstance(raw, list) else []:
            try:
                issues.append(DiagnosisIssue(**item))
            except Exception:  # noqa: BLE001 - 丢弃非法条目
                continue
        if not issues:
            issues = [DiagnosisIssue(**i) for i in fallback]
        return issues

    def _drain_fallback_events(self, state: AgentState) -> None:
        events = getattr(self.llm, "fallback_events", [])
        if events:
            state.degradation_notes.extend(events)
            events.clear()

    # ------------------------------------------------------------------ #
    # Stage 3: Plan（按需检索记忆注入）
    # ------------------------------------------------------------------ #
    def _stage_plan(self, state: AgentState, tracer: Tracer) -> None:
        t0 = time.perf_counter()
        state.current_stage = Stage.PLAN
        s = self.settings
        all_terms = {t.term: t for t in self.db.list_terms()}
        rag_notes: List[str] = []
        for p in state.paragraphs:
            diag = state.diagnoses[p.id]
            issue_types = {i.type for i in diag.issues}
            strats: List[StrategyName] = []
            if {"template_connector", "vague_modifier"} & issue_types:
                strats.append(StrategyName.DE_MECHANIZE)
            if {"long_sentence", "monotone_sentence"} & issue_types:
                strats.append(StrategyName.SENTENCE_RESTRUCTURE)
            if {"vague_modifier", "semantic_risk"} & issue_types:
                strats.append(StrategyName.CONCRETIZE)
            if diag.locked_terms_found or diag.numbers_found:
                strats.append(StrategyName.ACADEMIC_PRESERVE)
            if not strats:
                strats = [StrategyName.DE_MECHANIZE]

            # RAG 按需注入：仅当存在相关问题时检索，避免 Token 膨胀
            injected_terms: List[str] = list(diag.locked_terms_found)
            if "template_connector" in issue_types or "vague_modifier" in issue_types:
                hits = self.retriever.search_terms(p.text, top_k=s.retrieval_top_k_terms)
                for t, score in hits:
                    if t.term not in injected_terms and (t.term in p.text or score >= 0.35):
                        injected_terms.append(t.term)
            exp_ids: List[int] = []
            if diag.mechanical_score >= 0.3:
                for e, _ in self.retriever.search_experiences(p.text, top_k=s.retrieval_top_k_experiences):
                    if e.id is not None:
                        exp_ids.append(e.id)
            prefs = {k: v for k, v in state.preferences.items()
                     if k in {"tone", "avoid_first_person", "sentence_length", "connective_style"}}
            rag_notes.append(
                f"{p.id}: 策略 {[x.value for x in strats]}，注入术语 {len(injected_terms)}，经验 {len(exp_ids)}"
            )
            state.plans[p.id] = ParagraphPlan(
                paragraph_id=p.id,
                strategies=strats,
                rationale=f"诊断机械度 {diag.mechanical_score:.2f}，问题类型 {sorted(issue_types) or ['轻微']}",
                injected_terms=injected_terms,
                injected_experience_ids=exp_ids,
                injected_preferences=prefs,
            )
        latency = (time.perf_counter() - t0) * 1000
        tracer.log(
            stage=Stage.PLAN, status=StageStatus.OK, turn_id=0, latency_ms=latency,
            input_summary=f"诊断 {len(state.diagnoses)} 段；检索后端 {self.retriever.backend}",
            output_summary="；".join(rag_notes),
        )

    # ------------------------------------------------------------------ #
    # Stage 4/5 + Reflection 闭环
    # ------------------------------------------------------------------ #
    def _rewrite_verify_loop(self, state: AgentState, tracer: Tracer) -> None:
        results: Dict[str, ParagraphResult] = {
            p.id: ParagraphResult(paragraph_id=p.id, original=p.text) for p in state.paragraphs
        }
        suggestions: Dict[str, List[str]] = {p.id: [] for p in state.paragraphs}
        total_attempts = 1 + state.max_reflection_rounds

        for attempt in range(1, total_attempts + 1):
            state.turn = attempt
            pending = [p for p in state.paragraphs if not results[p.id].passed]
            if not pending:
                break

            # ---- REWRITE ----
            state.current_stage = Stage.REWRITE
            t0 = time.perf_counter()
            turn_usage = TokenUsage()
            rewrite_notes: List[str] = []
            for p in pending:
                resp = self._rewrite_one(state, p.id, p.text, attempt, suggestions[p.id])
                turn_usage = turn_usage + resp["usage"]
                results[p.id].versions.append(
                    RewriteVersion(attempt=attempt, text=resp["text"],
                                   usage=resp["usage"], latency_ms=resp["latency_ms"])
                )
                results[p.id].attempts = attempt
                rewrite_notes.append(f"{p.id}: {len(p.text)}→{len(resp['text'])} 字")
            state.total_usage = state.total_usage + turn_usage
            degraded = bool(getattr(self.llm, "fallback_events", []))
            self._drain_fallback_events(state)
            tracer.log(
                stage=Stage.REWRITE,
                status=StageStatus.DEGRADED if degraded else StageStatus.OK,
                turn_id=attempt, latency_ms=(time.perf_counter() - t0) * 1000,
                token_usage=turn_usage,
                input_summary=f"attempt {attempt}；{len(pending)} 段待改写；模型 {self.llm.last_used}",
                output_summary="；".join(rewrite_notes),
                error_or_reason="；".join(state.degradation_notes[-1:]) if degraded else "",
            )

            # ---- VERIFY ----
            state.current_stage = Stage.VERIFY
            t0 = time.perf_counter()
            v_usage = TokenUsage()
            fail_reasons: List[str] = []
            for p in pending:
                version_text = results[p.id].versions[-1].text
                report, usage = verifier.verify_paragraph(
                    paragraph_id=p.id,
                    attempt=attempt,
                    original=p.text,
                    rewritten=version_text,
                    locked_terms=[t for t in state.locked_terms if t in p.text],
                    retriever=self.retriever,
                    llm=self.llm,
                    settings=self.settings,
                )
                v_usage = v_usage + usage
                results[p.id].verify_reports.append(report)
                if report.passed:
                    results[p.id].passed = True
                    results[p.id].final_text = version_text
                else:
                    suggestions[p.id] = report.suggestions
                    fail_reasons.append(f"{p.id}: {'; '.join(report.blocking_reasons)}")
            state.total_usage = state.total_usage + v_usage
            tracer.log(
                stage=Stage.VERIFY,
                status=StageStatus.OK if not fail_reasons else StageStatus.FAILED,
                turn_id=attempt, latency_ms=(time.perf_counter() - t0) * 1000,
                token_usage=v_usage,
                input_summary=f"attempt {attempt}；校验 {len(pending)} 段（4 层检查）",
                output_summary=("全部通过" if not fail_reasons else f"{len(fail_reasons)} 段未通过"),
                error_or_reason="；".join(fail_reasons)[:600],
            )

            if not fail_reasons:
                break

            # ---- REFLECTION ----
            if attempt <= state.max_reflection_rounds:
                state.reflection_count += 1
                state.current_stage = Stage.REFLECTION
                tracer.log(
                    stage=Stage.REFLECTION, status=StageStatus.OK, turn_id=attempt,
                    latency_ms=0.0,
                    input_summary=f"第 {attempt} 轮校验失败 {len(fail_reasons)} 段",
                    output_summary=(f"进入第 {state.reflection_count}/{state.max_reflection_rounds}"
                                    f" 次定向返工，回传 {sum(len(suggestions[p.id]) for p in pending)} 条修改建议"),
                    error_or_reason="；".join(fail_reasons)[:400],
                )
            else:
                for p in pending:
                    results[p.id].final_text = results[p.id].versions[-1].text
                state.error = (f"达到最大返工次数 {state.max_reflection_rounds}，"
                               f"仍有 {len(fail_reasons)} 段未通过校验")
                tracer.log(
                    stage=Stage.REFLECTION, status=StageStatus.FAILED, turn_id=attempt,
                    latency_ms=0.0,
                    input_summary=f"第 {attempt} 轮校验失败",
                    output_summary="触发最大返工次数上限，终止闭环",
                    error_or_reason=state.error,
                )

        state.results = list(results.values())

    def _rewrite_one(
        self, state: AgentState, pid: str, text: str, attempt: int, suggestions: List[str]
    ) -> Dict[str, Any]:
        plan = state.plans[pid]
        all_terms = {t.term: t for t in self.db.list_terms()}
        ctx_terms = [all_terms[name] for name in plan.injected_terms if name in all_terms]
        context = {
            "attempt": attempt,
            "strategies": [s.value for s in plan.strategies],
            "terms": [t.model_dump() for t in ctx_terms],
            "locked_terms": [t for t in state.locked_terms if t in text],
            "avoid_first_person": state.preferences.get("avoid_first_person", "false").lower() == "true",
            "suggestions": suggestions,
        }
        strat_txt = "、".join(s.value for s in plan.strategies)
        system = (
            "TASK: rewrite\n"
            "你是学术论文改写引擎。按给定策略改写段落，保持语义、事实、数字、公式编号不变，"
            "锁定术语必须逐字保留，只输出改写后的段落文本。\n"
            f"策略组合: {strat_txt}\n"
            f"锁定术语: {json.dumps(context['locked_terms'], ensure_ascii=False)}\n"
            f"写作偏好: {json.dumps(plan.injected_preferences, ensure_ascii=False)}\n"
            + (f"上一轮返工意见: {json.dumps(suggestions, ensure_ascii=False)}\n" if suggestions else "")
            + f"CONTEXT_JSON: {json.dumps(context, ensure_ascii=False)}"
        )
        t0 = time.perf_counter()
        resp = self.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": text}]
        )
        latency = (time.perf_counter() - t0) * 1000
        out = resp.text.strip() or text
        return {"text": out, "usage": resp.usage, "latency_ms": latency}

    # ------------------------------------------------------------------ #
    # Stage 6: Deliver
    # ------------------------------------------------------------------ #
    def _stage_deliver(self, state: AgentState, tracer: Tracer) -> None:
        t0 = time.perf_counter()
        state.current_stage = Stage.DELIVER
        passed = sum(1 for r in state.results if r.passed)
        state.status = RunStatus.DELIVERED if passed == len(state.results) else RunStatus.PARTIAL
        tracer.log(
            stage=Stage.DELIVER,
            status=StageStatus.OK if state.status == RunStatus.DELIVERED else StageStatus.FAILED,
            turn_id=state.turn,
            latency_ms=(time.perf_counter() - t0) * 1000,
            input_summary=f"{len(state.results)} 段结果汇总",
            output_summary=(f"通过 {passed}/{len(state.results)}；返工 {state.reflection_count} 轮；"
                            f"Token {state.total_usage.total_tokens}"),
            error_or_reason=state.error or "",
        )

    # ------------------------------------------------------------------ #
    def _finalize(self, state: AgentState, tracer: Tracer, t_all: float) -> None:
        s = self.settings
        state.total_latency_ms = round((time.perf_counter() - t_all) * 1000, 2)
        state.total_cost = round(
            state.total_usage.prompt_tokens / 1000 * s.price_input_per_1k
            + state.total_usage.completion_tokens / 1000 * s.price_output_per_1k,
            8,
        )
        if state.status in (RunStatus.DELIVERED, RunStatus.PARTIAL):
            passed = sum(1 for r in state.results if r.passed)
            pass_rate = passed / len(state.results) if state.results else 0.0
            self.db.record_run(
                session_id=state.session_id, mode=state.mode, status=state.status.value,
                paragraphs=len(state.paragraphs), reflection_rounds=state.reflection_count,
                prompt_tokens=state.total_usage.prompt_tokens,
                completion_tokens=state.total_usage.completion_tokens,
                cost=state.total_cost, latency_ms=state.total_latency_ms,
                verify_pass_rate=pass_rate,
            )
        result_path = Path(s.runs_dir) / f"result_{state.session_id}.json"
        result_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
def build_llm_chain(settings: Settings, mode: str) -> FallbackLLMClient:
    mock = MockLLMClient(settings)
    if mode == "real":
        return FallbackLLMClient([OpenAICompatClient(settings), mock])
    return FallbackLLMClient([mock])


def build_default_engine(
    settings: Optional[Settings] = None, mode: Optional[str] = None
) -> RewriterEngine:
    settings = settings or get_settings()
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = MemoryDB(settings.db_path, seed_dir=settings.data_dir)
    db.init()
    retriever = MemoryRetriever(db, settings)
    llm = build_llm_chain(settings, mode or settings.llm_mode)
    return RewriterEngine(settings, db, retriever, llm)
