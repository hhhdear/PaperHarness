"""配置自检：打印 .env 是否被发现、当前生效的 LLM 配置（Key 打码）。

用法：python scripts/check_config.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from config import BASE_DIR, get_settings


def mask(key: str | None) -> str:
    if not key:
        return "❌ 未配置"
    if len(key) <= 8:
        return "✅ 已配置（长度异常，请检查）"
    return f"✅ 已配置 {key[:4]}...{key[-4:]}"


def main() -> None:
    env_file = BASE_DIR / ".env"
    print(f"项目根目录: {BASE_DIR}")
    print(f".env 文件: {'✅ 找到 ' + str(env_file) if env_file.exists() else '❌ 未找到（请确认文件名是 .env 而不是 .env.txt，且位于项目根目录）'}")
    s = get_settings()
    print("-" * 60)
    print(f"HARNESS_LLM_MODE      = {s.llm_mode}")
    print(f"OPENAI_API_KEY        = {mask(s.openai_api_key)}")
    print(f"OPENAI_BASE_URL       = {s.openai_base_url}")
    print(f"OPENAI_MODEL          = {s.openai_model}")
    print(f"HARNESS_LLM_TIMEOUT_S = {s.llm_timeout_s}")
    print("-" * 60)
    if s.llm_mode == "real" and not s.openai_api_key:
        print("⚠️  real 模式但未配置 Key：每次调用会走降级链回退到 Mock。")
    if s.llm_mode == "mock":
        print("当前为 Mock 演示模式；如需真实 LLM，请将 .env 中 HARNESS_LLM_MODE 改为 real。")


if __name__ == "__main__":
    main()
