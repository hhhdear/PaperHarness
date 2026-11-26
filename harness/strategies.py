"""可插拔改写策略的纯函数实现 + 段落诊断特征抽取。

该模块不依赖任何 LLM：Mock 客户端直接复用这些函数生成改写结果，
真实 LLM 模式下则把策略名与诊断摘要注入 Prompt。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from schemas import StrategyName, TermItem

_CJK_RE = re.compile(r"[一-鿿]")

# --------------------------------------------------------------------------- #
# 机械化特征词典
# --------------------------------------------------------------------------- #
ZH_CONNECTOR_ROTATIONS: Dict[str, List[str]] = {
    "首先": ["在方法层面，", "作为切入点，"],
    "其次": ["进一步地，", "在此之上，"],
    "再次": ["与此同时，"],
    "然后": ["随后，"],
    "最后": ["最终，"],
    "综上所述": ["综合上述实验结果"],
    "总而言之": ["概括而言"],
    "总的来说": ["整体而言"],
    "值得注意的是": ["需要指出的是"],
    "众所周知": ["已有研究普遍表明"],
    "与此同时": ["同期"],
}

EN_CONNECTOR_ROTATIONS: Dict[str, List[str]] = {
    "Firstly,": ["To begin with,", "As a first step,"],
    "Secondly,": ["Building on this,"],
    "Thirdly,": ["A further consideration is that"],
    "Moreover,": ["Beyond this,", "A further observation is that"],
    "Furthermore,": ["In addition to this,", "Complementing these findings,"],
    "In addition,": ["Complementing this,"],
    "In conclusion,": ["Taken together,"],
    "Last but not least,": ["Finally,"],
    "It is worth noting that": ["Notably,"],
    "All in all,": ["Overall,"],
    "To sum up,": ["In aggregate,"],
}

ZH_VAGUE_REPLACEMENTS: Dict[str, str] = {
    "非常重要": "重要",
    "十分重要": "重要",
    "非常": "",
    "十分": "",
    "极大地": "显著地",
    "极其": "显著",
    "很大程度上": "在相当程度上",
    "取得了很好的效果": "取得了稳定的性能提升",
    "很好的效果": "稳定的性能提升",
    "具有重要的意义": "具有明确的理论与应用意义",
    "深远的影响": "可验证的影响",
}

EN_VAGUE_REPLACEMENTS: Dict[str, str] = {
    "very important": "critical",
    "very ": "",
    "extremely ": "",
    "greatly ": "markedly ",
    "remarkably ": "",
    "It is well known that ": "Prior studies indicate that ",
    "achieved very good results": "achieved consistent performance gains",
}

ZH_ABSOLUTE_CLAIMS = ["完全解决", "彻底消除", "证明了其最优", "绝对优于", "百分之百"]
EN_ABSOLUTE_CLAIMS = ["completely solves", "always outperforms", "never fails", "proves to be optimal"]

# 数字 / 单位 / 公式编号 / 图表引用
NUMBER_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|％|GHz|MHz|kHz|Hz|ms|μs|s|km|m|cm|mm|kg|g|GB|MB|KB|TB|kW|W|V|mA|A|℃|°C|K|x|×|倍)?"
    r"|[（(]\d+[)）]"
    r"|(?:表|图|式|Table|Figure|Fig\.|Eq\.)\s*\d+",
    re.IGNORECASE,
)

ZH_SENT_END = re.compile(r"(?<=[。！？；])")
EN_SENT_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class FeatureReport:
    language: str
    sentences: List[str] = field(default_factory=list)
    connectors: List[str] = field(default_factory=list)
    vague_modifiers: List[str] = field(default_factory=list)
    absolute_claims: List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)
    avg_sentence_len: float = 0.0
    max_sentence_len: int = 0
    long_sentences: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def detect_language(text: str) -> str:
    cjk = len(_CJK_RE.findall(text))
    return "zh" if cjk >= max(4, len(text) * 0.05) else "en"


def split_sentences(text: str, lang: Optional[str] = None) -> List[str]:
    lang = lang or detect_language(text)
    parts = (ZH_SENT_END.split(text) if lang == "zh" else EN_SENT_END.split(text))
    return [p.strip() for p in parts if p.strip()]


def find_numbers(text: str) -> List[str]:
    found = [m.group(0).strip() for m in NUMBER_RE.finditer(text)]
    seen, out = set(), []
    for tok in found:
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def analyze_features(text: str) -> FeatureReport:
    lang = detect_language(text)
    sentences = split_sentences(text, lang)
    connectors = [k for k in (ZH_CONNECTOR_ROTATIONS if lang == "zh" else EN_CONNECTOR_ROTATIONS) if k in text]
    vague = [k for k in (ZH_VAGUE_REPLACEMENTS if lang == "zh" else EN_VAGUE_REPLACEMENTS) if k in text]
    claims = [c for c in (ZH_ABSOLUTE_CLAIMS if lang == "zh" else EN_ABSOLUTE_CLAIMS) if c.lower() in text.lower()]
    numbers = find_numbers(text)
    lens = [len(s) for s in sentences] or [0]
    long_limit = 90 if lang == "zh" else 160
    long_sents = [s for s in sentences if len(s) > long_limit]
    return FeatureReport(
        language=lang,
        sentences=sentences,
        connectors=connectors,
        vague_modifiers=vague,
        absolute_claims=claims,
        numbers=numbers,
        avg_sentence_len=sum(lens) / len(lens),
        max_sentence_len=max(lens),
        long_sentences=long_sents,
    )


def build_diagnosis_payload(text: str) -> Dict[str, Any]:
    """段落级"表达机械化"诊断（规则实现）。

    同时作为：a) Mock LLM 的诊断输出；b) 真实 LLM 诊断的参考上下文；
    c) LLM 异常/输出非法时引擎的兜底结果。
    """
    feats = analyze_features(text)
    issues: List[Dict[str, Any]] = []
    if feats.connectors:
        issues.append({
            "type": "template_connector", "severity": "high",
            "detail": f"模板化连接词堆砌 ×{len(feats.connectors)}",
            "evidence": feats.connectors,
        })
    if feats.vague_modifiers:
        issues.append({
            "type": "vague_modifier", "severity": "medium",
            "detail": f"空洞修饰 ×{len(feats.vague_modifiers)}",
            "evidence": [v.strip() for v in feats.vague_modifiers],
        })
    if feats.long_sentences:
        issues.append({
            "type": "long_sentence", "severity": "medium",
            "detail": f"超长句 ×{len(feats.long_sentences)}（最长 {feats.max_sentence_len} 字）",
            "evidence": [s[:40] for s in feats.long_sentences[:2]],
        })
    if feats.absolute_claims:
        issues.append({
            "type": "semantic_risk", "severity": "medium",
            "detail": "绝对化/夸大表述风险",
            "evidence": feats.absolute_claims,
        })
    openings = [s[:2] for s in feats.sentences if len(s) >= 2]
    if len(openings) >= 3 and len(set(openings)) <= max(1, len(openings) // 3):
        issues.append({
            "type": "monotone_sentence", "severity": "low",
            "detail": "句式开头高度重复，节奏单一",
            "evidence": sorted(set(openings))[:3],
        })
    mech = min(1.0, len(feats.connectors) * 0.18 + len(feats.vague_modifiers) * 0.12
               + len(feats.long_sentences) * 0.10 + len(feats.absolute_claims) * 0.08)
    summary = (f"{feats.language}｜{len(feats.sentences)} 句｜机械度 {mech:.2f}｜"
               f"连接词 {len(feats.connectors)} / 空洞修饰 {len(feats.vague_modifiers)} / "
               f"超长句 {len(feats.long_sentences)}｜数字 {len(feats.numbers)}")
    llm_summary = "；".join(i["detail"] for i in issues) or "表达自然，未发现明显机械化特征"
    return {
        "language": feats.language,
        "sentence_count": len(feats.sentences),
        "avg_sentence_len": round(feats.avg_sentence_len, 1),
        "connectors": feats.connectors,
        "vague_modifiers": [v.strip() for v in feats.vague_modifiers],
        "long_sentences": [s[:60] for s in feats.long_sentences],
        "absolute_claims": feats.absolute_claims,
        "numbers": feats.numbers,
        "issues": issues,
        "mechanical_score": round(mech, 3),
        "summary": summary,
        "llm_summary": llm_summary,
    }


# --------------------------------------------------------------------------- #
# 策略实现（返回 新文本 + 变更说明）
# --------------------------------------------------------------------------- #
def de_mechanize(text: str, lang: str, full: bool = True) -> Tuple[str, List[str]]:
    """机械特征剔除：替换模板化连接词与空洞修饰。

    full=False 时只处理一半种类的连接词（Mock 客户端用于演示 Reflection 打回）。
    """
    changes: List[str] = []
    rotations = ZH_CONNECTOR_ROTATIONS if lang == "zh" else EN_CONNECTOR_ROTATIONS
    present = [m for m in rotations if m in text]
    if not full:
        present = present[: max(1, (len(present) + 1) // 2)]
    for marker in present:
        alternatives = rotations[marker]
        idx = 0
        while marker in text:
            repl = alternatives[idx % len(alternatives)]
            if repl.endswith(("，", ",")):
                # 替换词自带逗号时吞掉原句紧跟的逗号，避免"xx，，"
                text = re.sub(re.escape(marker) + r"[，,]?", lambda _m: repl, text, count=1)
            else:
                text = text.replace(marker, repl, 1)
            idx += 1
        if idx:
            changes.append(f"替换模板化连接词「{marker}」×{idx}")
    vague_map = ZH_VAGUE_REPLACEMENTS if lang == "zh" else EN_VAGUE_REPLACEMENTS
    for vague, repl in vague_map.items():
        if vague in text:
            n = text.count(vague)
            text = text.replace(vague, repl)
            changes.append(f"剔除空洞修饰「{vague.strip()}」×{n}")
    return text, changes


def restructure_sentences(text: str, lang: str) -> Tuple[str, List[str]]:
    """句式重构：拆分超长句，打破单一节奏。"""
    changes: List[str] = []
    if lang == "zh":
        sentences = split_sentences(text, "zh")
        out: List[str] = []
        for s in sentences:
            cut = _best_split(s, "，", max_len=90, min_side=20)
            if cut:
                first, rest = cut
                out.append(first + "。")
                out.append(rest)
                changes.append(f"拆分超长句（{len(s)} 字 -> {len(first)}+{len(rest)} 字）")
            else:
                out.append(s)
        text = "".join(out)
        if not text.endswith(("。", "！", "？", "；")) and text:
            text += "。"
    else:
        sentences = split_sentences(text, "en")
        out = []
        for s in sentences:
            cut = _best_split(s, ", ", max_len=160, min_side=40)
            if cut:
                first, rest = cut
                out.append(first + ".")
                out.append(rest[0].upper() + rest[1:] if rest else rest)
                changes.append(f"Split long sentence ({len(s)} chars)")
            else:
                out.append(s)
        text = " ".join(out)
    return text, changes


def _best_split(sentence: str, sep: str, max_len: int, min_side: int):
    """在 sep 处寻找最接近句中点、且两侧都足够长的拆分点；找不到返回 None。"""
    if len(sentence) <= max_len or sep not in sentence:
        return None
    parts = sentence.split(sep)
    best = None
    for i in range(1, len(parts)):
        left = sep.join(parts[:i])
        right = sep.join(parts[i:])
        if len(left) < min_side or len(right) < min_side:
            continue
        score = abs(len(left) - len(sentence) / 2)
        if best is None or score < best[0]:
            best = (score, left.rstrip(sep.strip()), right)
    if best is None:
        return None
    return best[1], best[2]


def concretize(text: str, lang: str) -> Tuple[str, List[str]]:
    """表达具体化：缓和绝对化表述，给出可验证措辞（不虚构数字）。"""
    changes: List[str] = []
    if lang == "zh":
        pairs = {
            "完全解决": "针对性缓解",
            "彻底消除": "显著抑制",
            "绝对优于": "在本文设置下稳定优于",
            "很多": "相当数量",
            "大量实验": "多组对照实验",
        }
    else:
        pairs = {
            "completely solves": "substantially mitigates",
            "always outperforms": "consistently outperforms in our setting",
            "never fails": "rarely fails in our evaluation",
            "a lot of": "a substantial number of",
        }
    for src, dst in pairs.items():
        if src in text:
            text = text.replace(src, dst)
            changes.append(f"具体化/缓和表述「{src}」->「{dst}」")
    return text, changes


def academic_preserve(text: str, terms: List[TermItem]) -> Tuple[str, List[str]]:
    """学术风格保留：术语标准化（别名 -> 标准写法），不触碰锁定术语本身。"""
    changes: List[str] = []
    for t in terms:
        if t.standard_form and t.standard_form != t.term and t.term in text:
            text = text.replace(t.term, t.standard_form)
            changes.append(f"术语标准化「{t.term}」->「{t.standard_form}」")
    return text, changes


def apply_first_person_avoidance(text: str, lang: str) -> Tuple[str, List[str]]:
    changes: List[str] = []
    if lang == "zh":
        for src, dst in {"我们提出": "本文提出", "我们的方法": "本文方法", "我们实验": "本文实验"}.items():
            if src in text:
                text = text.replace(src, dst)
                changes.append(f"第一人称替换「{src}」->「{dst}」")
    else:
        for src, dst in {"We propose": "This paper proposes", "Our method": "The proposed method"}.items():
            if src in text:
                text = text.replace(src, dst)
                changes.append(f"First-person removal: {src}")
    return text, changes


# --------------------------------------------------------------------------- #
# 组合执行
# --------------------------------------------------------------------------- #
def compose_rewrite(
    text: str,
    strategies: List[StrategyName],
    *,
    terms: Optional[List[TermItem]] = None,
    avoid_first_person: bool = False,
    full: bool = True,
) -> Tuple[str, List[str]]:
    """按策略组合执行改写（Mock LLM 的核心实现）。"""
    lang = detect_language(text)
    all_changes: List[str] = []
    out = text
    for s in strategies:
        if s == StrategyName.DE_MECHANIZE:
            out, ch = de_mechanize(out, lang, full=full)
        elif s == StrategyName.SENTENCE_RESTRUCTURE:
            out, ch = restructure_sentences(out, lang)
        elif s == StrategyName.CONCRETIZE:
            out, ch = concretize(out, lang)
        elif s == StrategyName.ACADEMIC_PRESERVE:
            out, ch = academic_preserve(out, terms or [])
        else:
            ch = []
        all_changes.extend(ch)
    if avoid_first_person:
        out, ch = apply_first_person_avoidance(out, lang)
        all_changes.extend(ch)
    return out, all_changes
