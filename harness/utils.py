"""通用小工具：Token 估算、文本截断、JSON 抽取。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

_CJK_RE = re.compile(r"[一-鿿]")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def estimate_tokens(text: str) -> int:
    """粗略 Token 估算：CJK 每字 1 token，英文每词约 1.3 token。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    words = len(_WORD_RE.findall(text))
    return int(cjk + words * 1.3) + 4


def truncate(text: str, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def extract_json_obj(raw: str) -> Optional[Dict[str, Any]]:
    """从 LLM 输出中稳健抽取第一个 JSON 对象；失败返回 None。"""
    m = _JSON_RE.search(raw or "")
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
