"""全局配置：从环境变量加载，支持 dataclasses.replace 方式在测试中覆盖。"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RUNS_DIR = BASE_DIR / "runs"


def _env_str(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Harness 全局设置（冻结 dataclass，可用 .override() 派生）。"""

    # LLM
    llm_mode: str = "mock"            # mock | real
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 30.0
    llm_temperature: float | None = None    # None = 不传递，用模型默认值（兼容性最好）

    # Embedding
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_offline: bool = True     # True = 仅本地缓存，绝不联网

    # Harness 阈值
    max_reflection_rounds: int = 2
    length_tolerance: float = 0.35
    semantic_sim_threshold: float = 0.55
    llm_mech_max: float = 0.45
    llm_fidelity_min: float = 0.70

    # Guardrails 上限
    max_paragraphs: int = 20
    max_chars_per_paragraph: int = 6000
    max_total_chars: int = 40000

    # 成本（USD / 1K tokens）
    price_input_per_1k: float = 0.00015
    price_output_per_1k: float = 0.0006

    # 检索
    retrieval_top_k_terms: int = 8
    retrieval_top_k_experiences: int = 2

    # 路径
    db_path: Path = DATA_DIR / "memory.db"
    runs_dir: Path = RUNS_DIR
    data_dir: Path = DATA_DIR

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_mode=_env_str("HARNESS_LLM_MODE", "mock") or "mock",
            openai_api_key=_env_str("OPENAI_API_KEY"),
            openai_base_url=_env_str("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=_env_str("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
            llm_timeout_s=_env_float("HARNESS_LLM_TIMEOUT_S", 30.0),
            llm_temperature=(
                float(os.environ["HARNESS_LLM_TEMPERATURE"])
                if os.environ.get("HARNESS_LLM_TEMPERATURE") not in (None, "")
                else None
            ),
            embedding_model=_env_str("HARNESS_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5") or "BAAI/bge-small-zh-v1.5",
            embedding_offline=_env_bool("HARNESS_EMBEDDING_OFFLINE", True),
            max_reflection_rounds=_env_int("HARNESS_MAX_REFLECTION_ROUNDS", 2),
            length_tolerance=_env_float("HARNESS_LENGTH_TOLERANCE", 0.35),
            semantic_sim_threshold=_env_float("HARNESS_SEMANTIC_SIM_THRESHOLD", 0.55),
            llm_mech_max=_env_float("HARNESS_LLM_MECH_MAX", 0.45),
            llm_fidelity_min=_env_float("HARNESS_LLM_FIDELITY_MIN", 0.70),
            max_paragraphs=_env_int("HARNESS_MAX_PARAGRAPHS", 20),
            max_chars_per_paragraph=_env_int("HARNESS_MAX_CHARS_PER_PARAGRAPH", 6000),
            max_total_chars=_env_int("HARNESS_MAX_TOTAL_CHARS", 40000),
            price_input_per_1k=_env_float("HARNESS_PRICE_INPUT_PER_1K", 0.00015),
            price_output_per_1k=_env_float("HARNESS_PRICE_OUTPUT_PER_1K", 0.0006),
            retrieval_top_k_terms=_env_int("HARNESS_TOP_K_TERMS", 8),
            retrieval_top_k_experiences=_env_int("HARNESS_TOP_K_EXPERIENCES", 2),
        )

    def override(self, **kw) -> "Settings":
        return replace(self, **kw)


def _load_dotenv(path: Path) -> None:
    """极简 .env 加载器：项目内 .env 优先于系统环境变量（显式配置应生效）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv(BASE_DIR / ".env")
    s = Settings.from_env()
    s.runs_dir.mkdir(parents=True, exist_ok=True)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
