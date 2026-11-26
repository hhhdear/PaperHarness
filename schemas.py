"""强类型数据模型：AgentState / TraceEvent / VerifyReport / MemoryItem 等。

全部基于 Pydantic v2，可直接 JSON 序列化用于 Trace 落盘与 API 返回。
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field, model_validator


class Stage(str, Enum):
    GUARDRAILS = "guardrails"
    DIAGNOSE = "diagnose"
    PLAN = "plan"
    REWRITE = "rewrite"
    VERIFY = "verify"
    REFLECTION = "reflection"
    DELIVER = "deliver"


class StageStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    BLOCKED = "blocked"
    DEGRADED = "degraded"


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    DELIVERED = "delivered"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    ERROR = "error"


class StrategyName(str, Enum):
    SENTENCE_RESTRUCTURE = "句式重构"
    DE_MECHANIZE = "机械特征剔除"
    CONCRETIZE = "表达具体化"
    ACADEMIC_PRESERVE = "学术风格保留"


class MemoryKind(str, Enum):
    TERM = "term"
    PREFERENCE = "preference"
    EXPERIENCE = "experience"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class ParagraphInput(BaseModel):
    id: str
    text: str


class DiagnosisIssue(BaseModel):
    type: str
    severity: str
    detail: str
    evidence: List[str] = Field(default_factory=list)


class ParagraphDiagnosis(BaseModel):
    paragraph_id: str
    source: str = "llm"                # llm:<model> | rule-fallback（LLM 异常时降级）
    language: str = "zh"
    char_count: int = 0
    sentence_count: int = 0
    avg_sentence_len: float = 0.0
    mechanical_score: float = 0.0
    issues: List[DiagnosisIssue] = Field(default_factory=list)
    locked_terms_found: List[str] = Field(default_factory=list)
    numbers_found: List[str] = Field(default_factory=list)
    summary: str = ""


class ParagraphPlan(BaseModel):
    paragraph_id: str
    strategies: List[StrategyName] = Field(default_factory=list)
    rationale: str = ""
    injected_terms: List[str] = Field(default_factory=list)
    injected_experience_ids: List[int] = Field(default_factory=list)
    injected_preferences: Dict[str, str] = Field(default_factory=dict)


class RewriteVersion(BaseModel):
    attempt: int
    text: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0


class CheckResult(BaseModel):
    name: str
    passed: bool
    score: Optional[float] = None
    detail: str = ""
    blocking: bool = True


class VerifyReport(BaseModel):
    paragraph_id: str
    attempt: int
    passed: bool
    checks: List[CheckResult] = Field(default_factory=list)
    llm_mechanical_score: Optional[float] = None
    llm_fidelity_score: Optional[float] = None
    llm_comments: str = ""
    suggestions: List[str] = Field(default_factory=list)
    blocking_reasons: List[str] = Field(default_factory=list)


class ParagraphResult(BaseModel):
    paragraph_id: str
    original: str
    final_text: str = ""
    passed: bool = False
    attempts: int = 0
    versions: List[RewriteVersion] = Field(default_factory=list)
    verify_reports: List[VerifyReport] = Field(default_factory=list)


class AgentState(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    mode: str = "mock"
    status: RunStatus = RunStatus.CREATED
    current_stage: Optional[Stage] = None
    turn: int = 0
    reflection_count: int = 0
    max_reflection_rounds: int = 2
    paragraphs: List[ParagraphInput] = Field(default_factory=list)
    locked_terms: List[str] = Field(default_factory=list)
    preferences: Dict[str, str] = Field(default_factory=dict)
    diagnoses: Dict[str, ParagraphDiagnosis] = Field(default_factory=dict)
    plans: Dict[str, ParagraphPlan] = Field(default_factory=dict)
    results: List[ParagraphResult] = Field(default_factory=list)
    total_usage: TokenUsage = Field(default_factory=TokenUsage)
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    degradation_notes: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)


class TraceEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    ts: float = Field(default_factory=time.time)
    session_id: str
    turn_id: int = 0
    stage: Stage
    status: StageStatus
    latency_ms: float = 0.0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    input_summary: str = ""
    output_summary: str = ""
    error_or_reason: str = ""


class TermItem(BaseModel):
    id: Optional[int] = None
    term: str
    standard_form: str = ""
    locked: bool = False
    lang: str = "zh"
    note: str = ""


class PreferenceItem(BaseModel):
    id: Optional[int] = None
    key: str
    value: str
    note: str = ""


class ExperienceItem(BaseModel):
    id: Optional[int] = None
    title: str = ""
    source_text: str
    rewritten_text: str
    tags: str = ""
    quality: float = 0.8
    note: str = ""


class RewriteRequest(BaseModel):
    text: Optional[str] = None
    paragraphs: Optional[List[str]] = None
    mode: Optional[str] = None              # None = 使用服务端配置（HARNESS_LLM_MODE）
    extra_locked_terms: List[str] = Field(default_factory=list)
    preference_overrides: Dict[str, str] = Field(default_factory=dict)
    max_reflection_rounds: Optional[int] = None

    @model_validator(mode="after")
    def _check_input(self) -> "RewriteRequest":
        if not self.text and not self.paragraphs:
            raise ValueError("必须提供 text 或 paragraphs 之一")
        return self

    def to_paragraphs(self) -> List[ParagraphInput]:
        raw = self.paragraphs
        if not raw and self.text:
            raw = [p.strip() for p in self.text.split("\n") if p.strip()]
        return [ParagraphInput(id=f"p{i+1}", text=t) for i, t in enumerate(raw or [])]


class RewriteResponse(BaseModel):
    session_id: str
    status: RunStatus
    results: List[ParagraphResult]
    reflection_count: int
    total_usage: TokenUsage
    total_cost: float
    total_latency_ms: float
    degradation_notes: List[str] = Field(default_factory=list)
    trace_path: str = ""
    result_path: str = ""
    error: Optional[str] = None


class MemoryCreateRequest(BaseModel):
    kind: MemoryKind
    data: Dict[str, Any]
