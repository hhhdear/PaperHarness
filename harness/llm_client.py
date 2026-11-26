"""LLM 客户端层：OpenAI 兼容客户端 + 内置 Mock 演示客户端 + 降级链。

- MockLLMClient：零依赖、确定性，复用 strategies.py 的规则化改写，
  并在第 1 轮故意"不彻底"以演示 Reflection 打回重改。
- OpenAICompatClient：真实模式；任何异常抛 LLMError 交给降级链处理。
- FallbackLLMClient：按顺序尝试多个客户端，记录降级事件。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from config import Settings
from harness import strategies
from harness.utils import estimate_tokens
from schemas import StrategyName, TermItem, TokenUsage


class LLMError(RuntimeError):
    """LLM 调用失败（超时 / 网络 / 鉴权 / 解析），触发降级链。"""


class LLMResponse:
    def __init__(self, text: str, usage: TokenUsage, model: str):
        self.text = text
        self.usage = usage
        self.model = model

    def __repr__(self) -> str:
        return f"LLMResponse(model={self.model!r}, chars={len(self.text)})"


def _messages_text(messages: List[Dict[str, str]]) -> str:
    return "\n".join(m.get("content", "") for m in messages)


class MockLLMClient:
    """内置演示客户端：确定性规则化 LLM，无需任何 Key/网络。"""

    name = "mock-llm"

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings

    def chat(self, messages: List[Dict[str, str]], **_: Any) -> LLMResponse:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if "TASK: rewrite" in system:
            text = self._rewrite(system, user)
        elif "TASK: diagnose" in system:
            text = self._diagnose(user)
        elif "TASK: review" in system:
            text = self._review(user)
        else:
            text = "MOCK-LLM: unrecognized task."
        usage = TokenUsage(
            prompt_tokens=estimate_tokens(_messages_text(messages)),
            completion_tokens=estimate_tokens(text),
        )
        return LLMResponse(text=text, usage=usage, model=self.name)

    @staticmethod
    def _diagnose(user: str) -> str:
        payload = strategies.build_diagnosis_payload(user)
        return json.dumps(
            {
                "issues": payload["issues"],
                "mechanical_score": payload["mechanical_score"],
                "summary": payload["llm_summary"],
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _parse_context(system: str) -> Dict[str, Any]:
        m = re.search(r"CONTEXT_JSON:\s*(\{.*\})", system, re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return {}

    def _rewrite(self, system: str, paragraph: str) -> str:
        ctx = self._parse_context(system)
        attempt = int(ctx.get("attempt", 1))
        strategy_names = [StrategyName(s) for s in ctx.get("strategies", [])] or [
            StrategyName.DE_MECHANIZE
        ]
        terms = [TermItem(**t) for t in ctx.get("terms", [])]
        avoid_fp = bool(ctx.get("avoid_first_person", False))
        full = attempt >= 2
        rewritten, _changes = strategies.compose_rewrite(
            paragraph,
            strategy_names,
            terms=terms,
            avoid_first_person=avoid_fp,
            full=full,
        )
        return rewritten

    def _review(self, user: str) -> str:
        original, _, rewritten = user.partition("\n---REWRITTEN---\n")
        rewritten = rewritten or user
        feats = strategies.analyze_features(rewritten)
        n_sent = max(1, len(feats.sentences))
        mech = min(
            1.0,
            len(feats.connectors) * 0.25
            + len(feats.vague_modifiers) * 0.15
            + 0.1 * len(feats.absolute_claims),
        )
        fidelity = self._content_overlap(original, rewritten)
        mech_max = getattr(self.settings, "llm_mech_max", 0.45) if self.settings else 0.45
        fid_min = getattr(self.settings, "llm_fidelity_min", 0.70) if self.settings else 0.70
        passed = mech <= mech_max and fidelity >= fid_min
        suggestions: List[str] = []
        if feats.connectors:
            suggestions.append(f"仍存在模板化连接词：{'、'.join(feats.connectors)}，请全部替换")
        if feats.vague_modifiers:
            suggestions.append(f"仍存在空洞修饰：{'、'.join(v.strip() for v in feats.vague_modifiers)}")
        if feats.absolute_claims:
            suggestions.append(f"存在绝对化表述：{'、'.join(feats.absolute_claims)}，请缓和")
        if fidelity < fid_min:
            suggestions.append("语义保真度不足，请减小改写幅度")
        comments = (
            f"句子数 {n_sent}，机械化评分 {mech:.2f}，语义保真度 {fidelity:.2f}。"
            + ("达到发表级表达要求。" if passed else "仍存在机械化痕迹，建议返工。")
        )
        return json.dumps(
            {
                "mechanical_score": round(mech, 3),
                "fidelity_score": round(fidelity, 3),
                "passed": passed,
                "comments": comments,
                "suggestions": suggestions,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _content_overlap(a: str, b: str) -> float:
        tok = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")
        stop = set("的一是了在和我你他这那有与及或等")
        sa = {t.lower() for t in tok.findall(a) if t.lower() not in stop}
        sb = {t.lower() for t in tok.findall(b) if t.lower() not in stop}
        if not sa:
            return 1.0
        return len(sa & sb) / len(sa)


class OpenAICompatClient:
    name = "openai-compat"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.settings.openai_api_key:
            raise LLMError("未配置 OPENAI_API_KEY")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(f"openai SDK 未安装: {exc}") from exc
        self._client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            timeout=self.settings.llm_timeout_s,
            max_retries=0,
        )
        return self._client

    def chat(self, messages: List[Dict[str, str]], json_mode: bool = False, **_: Any) -> LLMResponse:
        client = self._ensure_client()
        kwargs: Dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.settings.llm_temperature is not None:
            kwargs["temperature"] = self.settings.llm_temperature
        try:
            resp = client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                **kwargs,
            )
        except Exception as exc:
            raise LLMError(f"OpenAI 调用失败: {type(exc).__name__}: {exc}") from exc
        text = (resp.choices[0].message.content or "").strip()
        usage = TokenUsage(
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0)
            or estimate_tokens(_messages_text(messages)),
            completion_tokens=getattr(resp.usage, "completion_tokens", 0) or estimate_tokens(text),
        )
        return LLMResponse(text=text, usage=usage, model=self.settings.openai_model)


class FallbackLLMClient:
    """按顺序尝试多个客户端，全部失败才抛 LLMError；记录降级事件。"""

    def __init__(self, clients: List[Any]):
        if not clients:
            raise ValueError("至少需要一个 LLM 客户端")
        self.clients = clients
        self.fallback_events: List[str] = []
        self.last_used: str = clients[0].name

    @property
    def name(self) -> str:
        return "fallback(" + "->".join(c.name for c in self.clients) + ")"

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResponse:
        errors: List[str] = []
        for client in self.clients:
            try:
                resp = client.chat(messages, **kwargs)
                self.last_used = client.name
                return resp
            except LLMError as exc:
                note = f"[{time.strftime('%H:%M:%S')}] {client.name} 失败，降级: {exc}"
                self.fallback_events.append(note)
                errors.append(note)
            except Exception as exc:
                note = f"[{time.strftime('%H:%M:%S')}] {client.name} 异常，降级: {type(exc).__name__}: {exc}"
                self.fallback_events.append(note)
                errors.append(note)
        raise LLMError("；".join(errors) or "所有 LLM 客户端均不可用")
