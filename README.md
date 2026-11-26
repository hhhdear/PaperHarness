# 基于自研 Agent Harness 的学术论文智能改写系统（MVP）

一个可本地运行、可直接演示的学术论文智能改写原型：围绕 **六阶段状态机 + Reflection 定向返工闭环** 构建，
集成 **RAG 检索增强 + SQLite 跨会话记忆**、**Turn/Stage 双级 JSONL Trace**、**四层校验体系** 与 **Token/成本统计**，
提供 **FastAPI 后端**、**Streamlit 演示端** 与 **离线 Pytest 测试套件**。

无需任何 API Key 即可完整演示多轮 Reflection 打回重改（内置 Mock LLM + TF-IDF 检索降级链）。

## 架构总览

```text
                ┌─────────────────────────── AgentState（Pydantic 全局状态）───────────────────────────┐
                │  session_id / paragraphs / diagnoses / plans / versions / verify / tokens / cost   │
                └──────────────────────────────────────────────────────────────────────────────────┘
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐            ┌───────────┐
│ Guardrails│→│ Diagnose  │→│   Plan    │→│  Rewrite  │→│  Verify   │──────────→──│  Deliver  │
│ 输入护栏  │  │ 段落诊断  │  │ 策略规划  │  │ 分段改写  │  │ 四层校验  │            │ 交付汇总  │
└───────────┘  └───────────┘  └────┬──────┘  └─────↑─────┘  └─────┬─────┘            └───────────┘
       │                           │                │              │ 未通过且未超上限
       │ 长度/注入/敏感词           │ RAG 按需注入    │              ▼
       │ 术语锁定解析               │ 术语/经验/偏好   │       ┌─────────────┐
       ▼                           ▼                │       │ Reflection  │── 携带失败原因+返工意见回传 Rewrite
  拦截/放行                 SQLite 记忆库            └───────┤ (≤ max 轮)  │
                       terms/preferences/experiences        └─────────────┘

Verify 四层校验：① 长度偏差 ② 术语/数字硬约束 ③ Embedding 语义相似度 ④ LLM 结构化评审(JSON)
降级链：LLM = OpenAI 兼容客户端 →(异常/超时)→ Mock 客户端；Embedding = BGE →(离线无缓存)→ TF-IDF
全程 Turn/Stage 双级 Trace 实时追加落盘 runs/trace_<session_id>.jsonl
```

## 目录结构

```text
.
├── config.py               # 全局配置（环境变量加载，可 override）
├── schemas.py              # AgentState / TraceEvent / VerifyReport / MemoryItem 等模型
├── memory/
│   ├── db.py               # SQLite 持久化（terms / preferences / experiences / runs）
│   └── retriever.py        # BGE Embedding 检索 + TF-IDF 降级检索
├── harness/
│   ├── llm_client.py       # OpenAI 兼容客户端 + Mock 演示客户端 + 降级链
│   ├── strategies.py       # 句式重构 / 机械特征剔除 / 表达具体化 / 学术风格保留
│   ├── verifier.py         # 长度偏差 / 硬约束 / 语义相似度 / LLM 结构化评审
│   ├── trace.py            # Turn/Stage 两级 JSONL Trace 记录器
│   └── engine.py           # 六阶段状态机 + Reflection 闭环
├── api/main.py             # FastAPI 服务
├── ui/app.py               # Streamlit 可视化（控制台 / Trace / Diff / 看板）
├── tests/test_harness.py   # Pytest 离线测试（13 项）
├── scripts/demo.py         # 一键演示脚本（引擎 + API 冒烟）
├── data/                   # 种子术语/偏好/经验 + 中英示例段落
├── runs/                   # JSONL Trace 与结果快照输出目录
├── requirements.txt
└── .env.example
```

## 快速开始

```bash
pip install -r requirements.txt

# 1) 跑测试（离线，约 1 秒）
python -m pytest tests -q

# 2) 一键演示（引擎闭环 + API 冒烟，无需启动服务器）
python scripts/demo.py

# 3) 启动 Streamlit 演示端（推荐演示入口）
python -m streamlit run ui/app.py

# 4) 启动 FastAPI 后端
python -m uvicorn api.main:app --reload --port 8000
# 接口文档: http://localhost:8000/docs
```

## 两种运行模式

| 模式 | 说明 |
|---|---|
| `mock`（默认） | 内置规则化 Mock LLM：第 1 轮故意"不彻底"改写 → Verify 打回 → Reflection 返工 → 第 2 轮通过，确定性演示完整闭环，零成本零依赖 |
| `real` | 走 OpenAI 兼容接口（`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`），任何异常自动降级回 Mock 并在 Trace 中记录降级事件 |

## LLM 配置（三种方式，优先级从高到低）

1. **Streamlit 侧边栏「🔑 LLM 配置」面板**（推荐演示用）：直接填 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`，
   仅对当前会话生效，面板下方实时显示当前生效配置；
2. **项目根目录 `.env` 文件**（推荐本地持久用）：启动时自动加载，**优先于系统环境变量**，修改后需重启服务；
3. **系统环境变量**：与 `.env.example` 中同名变量一一对应（会被 `.env` 覆盖）。

> 改了配置不生效？先跑 `python scripts/check_config.py` 自检：
> 它会告诉你 .env 是否被找到（注意文件名必须是 `.env` 而非 `.env.txt`）、当前生效的 mode/key/base_url/model（Key 打码显示）。

任意 OpenAI 兼容服务均可接入（DeepSeek、通义、Moonshot、本地 vLLM/Ollama 等），只需改 `OPENAI_BASE_URL` 与 `OPENAI_MODEL`，例如：

```ini
HARNESS_LLM_MODE=real
OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```

> Diagnose（段落诊断）阶段同样走 LLM 调用并计入 Token/成本：输出结构化 JSON 诊断；
> LLM 异常或输出非法 JSON 时自动回退本地规则诊断，Trace 中标记 `degraded`，诊断来源字段显示为 `rule-fallback`。

Embedding 检索：优先加载本地缓存的 `BAAI/bge-small-zh-v1.5`（`HARNESS_EMBEDDING_OFFLINE=0` 可允许联网下载），
离线无缓存时自动降级为纯 Python TF-IDF 稀疏向量余弦相似度，任何环境都能跑通。

## API 一览

| 接口 | 说明 |
|---|---|
| `POST /api/rewrite` | 提交段落/术语约束/偏好/模式，执行完整 Harness，返回结果与 Trace 路径 |
| `GET /api/trace/{session_id}` | 查询会话的 Turn/Stage Trace 明细 |
| `GET /api/memories?kind=...` | 查看术语/偏好/经验 |
| `POST /api/memories` | 新增术语/偏好/经验（自动失效检索缓存） |
| `GET /api/metrics` | 历史运行成本、平均返工轮次、校验通过率看板 |
| `GET /api/health` | 模式与 Embedding 后端状态 |

## Trace 格式（runs/trace_<session_id>.jsonl）

```json
{"event_id":"...","ts":1764056000.0,"session_id":"9f85c78390e5","turn_id":1,"stage":"verify",
 "status":"failed","latency_ms":12.5,"token_usage":{"prompt_tokens":380,"completion_tokens":96},
 "input_summary":"attempt 1；校验 2 段（4 层检查）","output_summary":"2 段未通过",
 "error_or_reason":"p1: [llm_review] 机械度 0.50(≤0.45) / 保真度 0.88(≥0.7) / LLM 判定 打回"}
```

## 关键设计

- **诊断摘要回传**：Reflection 不把整段历史塞回上下文，仅回传结构化失败原因与修改建议；
- **分段按需供给上下文**：Plan 阶段按段落检索 top-k 术语/经验注入，避免全量塞入导致 Token 膨胀；
- **成本控制**：每轮 Rewrite/Verify 统计 prompt/completion token，按单价估算单次与累计成本并写入 `runs` 表；
- **防死循环**：`max_reflection_rounds` 硬上限（默认 2），超限交付 `partial` 状态并记录原因；
- **强类型全链路**：`AgentState` / `TraceEvent` / `VerifyReport` 均为 Pydantic 模型，Trace 与结果快照可直接复现。

## 测试说明

`tests/test_harness.py` 覆盖：状态机正常流转、Reflection 触发与最大次数终止、Guardrails 拦截（注入/超长）、
LLM 降级链、Embedding 降级、长度偏差/锁定术语/语义相似度三类打回、SQLite 读写与检索、JSONL Trace 格式。

> 本机已知问题：若 `python -m pytest` 因 Python 安装目录下的 `py.py` 残留文件报 `EOFError`，
> 可用 `python -c "import sys; p=r'<Python310目录>'; sys.path.remove(p); sys.path.append(p); import pytest; raise SystemExit(pytest.main(['tests','-q']))"` 运行。

## 演示动线建议

1. Streamlit 侧边栏选 **Mock 演示模式** → 「填入中文示例」→「执行 Harness 改写」；
2. 「流程状态与 Trace」展示六阶段流转、Reflection 返工、耗时/Token 分布与 JSONL 明细；
3. 「原文 vs 改写 Diff」切换第 1/2 轮版本，展示打回原因与高亮差异；
4. 「记忆库与成本看板」新增一条锁定术语（如`混合专家模型`）→ 重新改写含该词段落，展示硬约束生效；
5. 侧边栏切到 **真实 LLM 模式**（无 Key）再跑一次，展示降级链事件记录。
