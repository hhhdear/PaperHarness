"""Turn / Stage 两级 PipelineTrace 记录器：实时追加写入 JSONL。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from schemas import Stage, StageStatus, TokenUsage, TraceEvent


class Tracer:
    def __init__(self, runs_dir: Path | str, session_id: str):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.path = self.runs_dir / f"trace_{session_id}.jsonl"

    def log(
        self,
        *,
        stage: Stage,
        status: StageStatus,
        turn_id: int = 0,
        latency_ms: float = 0.0,
        token_usage: Optional[TokenUsage] = None,
        input_summary: str = "",
        output_summary: str = "",
        error_or_reason: str = "",
    ) -> TraceEvent:
        event = TraceEvent(
            session_id=self.session_id,
            turn_id=turn_id,
            stage=stage,
            status=status,
            latency_ms=round(latency_ms, 2),
            token_usage=token_usage or TokenUsage(),
            input_summary=input_summary,
            output_summary=output_summary,
            error_or_reason=error_or_reason,
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return event


def read_trace(runs_dir: Path | str, session_id: str) -> Optional[List[TraceEvent]]:
    path = Path(runs_dir) / f"trace_{session_id}.jsonl"
    if not path.exists():
        return None
    events: List[TraceEvent] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(TraceEvent(**json.loads(line)))
    return events


def list_trace_sessions(runs_dir: Path | str) -> List[str]:
    runs = Path(runs_dir)
    if not runs.exists():
        return []
    return sorted(p.stem.replace("trace_", "") for p in runs.glob("trace_*.jsonl"))
