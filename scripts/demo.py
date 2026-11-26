"""一键演示脚本：引擎闭环 + FastAPI 接口冒烟（无需启动真实服务器）。

用法：python scripts/demo.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from fastapi.testclient import TestClient

from api.main import app
from harness.engine import build_default_engine
from schemas import RewriteRequest


def engine_demo() -> str:
    print("=" * 70)
    print("【1】引擎闭环演示（Mock 模式，离线）")
    print("=" * 70)
    examples = json.loads((ROOT / "data" / "examples.json").read_text(encoding="utf-8"))
    engine = build_default_engine(mode="mock")
    print(f"Embedding 后端: {engine.retriever.backend}（{engine.retriever.backend_note}）")
    state = engine.run(
        RewriteRequest(text="\n\n".join(examples["zh"]["paragraphs"]), mode="mock")
    )
    print(f"状态: {state.status.value} | Reflection 返工: {state.reflection_count} 轮 | "
          f"Token: {state.total_usage.total_tokens} | 成本: ${state.total_cost:.6f} | "
          f"耗时: {state.total_latency_ms:.0f} ms")
    for r in state.results:
        print("-" * 70)
        print(f"{r.paragraph_id}（尝试 {r.attempts} 轮，{'通过' if r.passed else '未通过'}）")
        print(f"原文: {r.original[:90]}...")
        print(f"终稿: {r.final_text[:90]}...")
        for rep in r.verify_reports:
            mark = "PASS" if rep.passed else "REJECT"
            print(f"  第 {rep.attempt} 轮校验: {mark} {'; '.join(rep.blocking_reasons)[:120]}")
    return state.session_id


def api_demo(session_id: str) -> None:
    print()
    print("=" * 70)
    print("【2】FastAPI 接口冒烟（TestClient，无需启动服务器）")
    print("=" * 70)
    client = TestClient(app)
    print("GET /api/health ->", client.get("/api/health").json())

    tr = client.get(f"/api/trace/{session_id}")
    events = tr.json()
    print(f"GET /api/trace/{session_id} -> {len(events)} 条事件: "
          f"{[e['stage'] for e in events]}")

    mem = client.get("/api/memories", params={"kind": "term"}).json()
    print(f"GET /api/memories?kind=term -> {len(mem['terms'])} 条术语")

    r = client.post("/api/memories", json={
        "kind": "term",
        "data": {"term": "混合专家模型", "standard_form": "混合专家模型",
                 "locked": True, "lang": "zh", "note": "demo 新增"},
    })
    print("POST /api/memories ->", r.json())

    metrics = client.get("/api/metrics").json()
    summary = {k: v for k, v in metrics.items() if k != "recent_runs"}
    print("GET /api/metrics ->", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    sid = engine_demo()
    api_demo(sid)
