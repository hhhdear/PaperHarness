"""多层校验体系：

1. 长度偏差检测
2. 术语与数字硬约束检查
3. 语义相似度校验（Embedding 余弦相似度）
4. LLM 结构化评审（JSON：机械度 / 语义保真度 / 通过状态 / 返工意见）
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from config import Settings
from harness import strategies
from harness.llm_client import FallbackLLMClient
from harness.utils import truncate
from schemas import CheckResult, TokenUsage, VerifyReport


def _length_units(text: str, lang: str) -> int:
    if lang == "zh":
        return len(re.sub(r"\s", "", text))
    return len(re.findall(r"[A-Za-z0-9_']+", text))


def length_check(original: str, rewritten: str, lang: str, tolerance: float) -> CheckResult:
    """长度偏差检测：|1 - 新/旧| > tolerance 则打回。"""
    lo = max(1, _length_units(original, lang))
    lr = _length_units(rewritten, lang)
    ratio = lr / lo
    deviation = abs(1.0 - ratio)
    passed = deviation <= tolerance
    return CheckResult(
        name="length_deviation",
        passed=passed,
        score=round(ratio, 3),
        detail=f"长度比 {ratio:.2f}（原文 {lo} / 改写 {lr}，容差 ±{tolerance:.0%}）",
        blocking=True,
    )


def hard_constraint_check(
    original: str, rewritten: str, locked_terms: List[str]
) -> CheckResult:
    """术语与数字硬约束：锁定术语逐字保留，关键数字/单位/公式编号不丢失。"""
    problems: List[str] = []
    for term in locked_terms:
        if term and term not in rewritten:
            problems.append(f"锁定术语被篡改/丢失：「{term}」")
    for num in strategies.find_numbers(original):
        if num not in rewritten:
            problems.append(f"关键数字/编号丢失：「{num}」")
    passed = not problems
    return CheckResult(
        name="hard_constraint",
        passed=passed,
        score=1.0 if passed else 0.0,
        detail="；".join(problems) if problems else f"{len(locked_terms)} 个锁定术语、全部数字均保留",
        blocking=True,
    )


def semantic_check(original: str, rewritten: str, retriever: Any, threshold: float) -> CheckResult:
    """语义相似度校验：Embedding 余弦相似度低于阈值判定语义失真。"""
    try:
        sim = float(retriever.similarity(original, rewritten))
    except Exception as exc:  # noqa: BLE001 - 检索器异常不阻断，记为非阻断告警
        return CheckResult(
            name="semantic_similarity",
            passed=True,
            score=None,
            detail=f"相似度计算异常，跳过（{type(exc).__name__}）",
            blocking=False,
        )
    passed = sim >= threshold
    return CheckResult(
        name="semantic_similarity",
        passed=passed,
        score=round(sim, 4),
        detail=f"余弦相似度 {sim:.3f}（阈值 {threshold:.2f}，后端 {getattr(retriever, 'backend', '?')}）",
        blocking=True,
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_review_json(raw: str) -> Dict[str, Any]:
    m = _JSON_RE.search(raw or "")
    if not m:
        raise ValueError("评审输出中未找到 JSON")
    data = json.loads(m.group(0))
    return {
        "mechanical_score": float(data.get("mechanical_score", 0.0)),
        "fidelity_score": float(data.get("fidelity_score", 1.0)),
        "passed": bool(data.get("passed", False)),
        "comments": str(data.get("comments", "")),
        "suggestions": [str(s) for s in data.get("suggestions", [])],
    }


def llm_review_check(
    original: str,
    rewritten: str,
    llm: FallbackLLMClient,
    settings: Settings,
) -> tuple[CheckResult, Dict[str, Any], TokenUsage]:
    """LLM 结构化评审。返回 (CheckResult, 解析结果, TokenUsage)。"""
    system = (
        "TASK: review\n"
        "你是学术论文改写质量评审员。请对比【原文】与【改写文】，仅输出 JSON：\n"
        '{"mechanical_score":0~1,"fidelity_score":0~1,"passed":true/false,'
        '"comments":"...","suggestions":["..."]}\n'
        "mechanical_score 越高表示表达越机械化；fidelity_score 越高表示语义越保真。"
    )
    user = original + "\n---REWRITTEN---\n" + rewritten
    resp = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        json_mode=True,
    )
    usage = resp.usage
    try:
        review = parse_review_json(resp.text)
    except (ValueError, json.JSONDecodeError) as exc:
        return (
            CheckResult(
                name="llm_review",
                passed=False,
                score=None,
                detail=f"评审 JSON 解析失败：{exc}",
                blocking=True,
            ),
            {"mechanical_score": None, "fidelity_score": None, "comments": "", "suggestions": ["评审输出格式错误，请重新生成"]},
            usage,
        )
    mech_ok = review["mechanical_score"] <= settings.llm_mech_max
    fid_ok = review["fidelity_score"] >= settings.llm_fidelity_min
    passed = bool(review["passed"]) and mech_ok and fid_ok
    detail = (
        f"机械度 {review['mechanical_score']:.2f}(≤{settings.llm_mech_max}) / "
        f"保真度 {review['fidelity_score']:.2f}(≥{settings.llm_fidelity_min}) / "
        f"LLM 判定 {'通过' if review['passed'] else '打回'}"
    )
    return (
        CheckResult(name="llm_review", passed=passed, score=review["fidelity_score"],
                    detail=detail, blocking=True),
        review,
        usage,
    )


def verify_paragraph(
    *,
    paragraph_id: str,
    attempt: int,
    original: str,
    rewritten: str,
    locked_terms: List[str],
    retriever: Any,
    llm: FallbackLLMClient,
    settings: Settings,
) -> tuple[VerifyReport, TokenUsage]:
    """执行全部四层校验，返回 (VerifyReport, 本轮评审 TokenUsage)。"""
    lang = strategies.detect_language(original)
    checks: List[CheckResult] = [
        length_check(original, rewritten, lang, settings.length_tolerance),
        hard_constraint_check(original, rewritten, locked_terms),
        semantic_check(original, rewritten, retriever, settings.semantic_sim_threshold),
    ]
    review_check, review, review_usage = llm_review_check(original, rewritten, llm, settings)
    checks.append(review_check)

    blocking_reasons = [f"[{c.name}] {c.detail}" for c in checks if c.blocking and not c.passed]
    suggestions: List[str] = list(review.get("suggestions", []))
    for c in checks:
        if c.blocking and not c.passed and c.name != "llm_review":
            suggestions.append(c.detail)
    passed = not blocking_reasons
    report = VerifyReport(
        paragraph_id=paragraph_id,
        attempt=attempt,
        passed=passed,
        checks=checks,
        llm_mechanical_score=review.get("mechanical_score"),
        llm_fidelity_score=review.get("fidelity_score"),
        llm_comments=truncate(str(review.get("comments", "")), 200),
        suggestions=suggestions,
        blocking_reasons=blocking_reasons,
    )
    return report, review_usage
