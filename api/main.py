"""FastAPI 后端：改写执行 / Trace 查询 / 记忆管理 / 成本看板。

启动：uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from harness.engine import RewriterEngine, build_llm_chain
from harness.trace import read_trace
from memory.db import MemoryDB
from memory.retriever import MemoryRetriever
from schemas import (
    ExperienceItem,
    MemoryCreateRequest,
    MemoryKind,
    PreferenceItem,
    RewriteRequest,
    RewriteResponse,
    TermItem,
)

settings = get_settings()
db = MemoryDB(settings.db_path, seed_dir=settings.data_dir)
db.init()
retriever = MemoryRetriever(db, settings)
_lock = threading.Lock()

app = FastAPI(title="Academic Paper Rewriting Agent Harness", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "llm_mode": settings.llm_mode,
        "embedding_backend": retriever.backend,
        "embedding_note": retriever.backend_note,
    }


@app.post("/api/rewrite", response_model=RewriteResponse)
def rewrite(req: RewriteRequest) -> RewriteResponse:
    mode = req.mode or settings.llm_mode
    engine = RewriterEngine(settings, db, retriever, build_llm_chain(settings, mode))
    with _lock:
        state = engine.run(req)
    return RewriteResponse(
        session_id=state.session_id,
        status=state.status,
        results=state.results,
        reflection_count=state.reflection_count,
        total_usage=state.total_usage,
        total_cost=state.total_cost,
        total_latency_ms=state.total_latency_ms,
        degradation_notes=state.degradation_notes,
        trace_path=str(settings.runs_dir / f"trace_{state.session_id}.jsonl"),
        result_path=str(settings.runs_dir / f"result_{state.session_id}.json"),
        error=state.error,
    )


@app.get("/api/trace/{session_id}")
def get_trace(session_id: str) -> List[Dict[str, Any]]:
    events = read_trace(settings.runs_dir, session_id)
    if events is None:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 的 Trace 不存在")
    return [e.model_dump(mode="json") for e in events]


@app.get("/api/memories")
def list_memories(kind: Optional[MemoryKind] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if kind in (None, MemoryKind.TERM):
        out["terms"] = [t.model_dump() for t in db.list_terms()]
    if kind in (None, MemoryKind.PREFERENCE):
        out["preferences"] = [p.model_dump() for p in db.list_preferences()]
    if kind in (None, MemoryKind.EXPERIENCE):
        out["experiences"] = [e.model_dump() for e in db.list_experiences()]
    return out


@app.post("/api/memories")
def create_memory(req: MemoryCreateRequest) -> Dict[str, Any]:
    try:
        if req.kind == MemoryKind.TERM:
            item_id = db.add_term(TermItem(**req.data))
        elif req.kind == MemoryKind.PREFERENCE:
            item_id = db.set_preference(PreferenceItem(**req.data))
        else:
            item_id = db.add_experience(ExperienceItem(**req.data))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"记忆写入失败: {exc}") from exc
    retriever.invalidate()
    return {"ok": True, "kind": req.kind.value, "id": item_id}


@app.get("/api/metrics")
def metrics() -> Dict[str, Any]:
    summary = db.metrics_summary()
    summary["recent_runs"] = db.list_runs(limit=20)
    return summary
