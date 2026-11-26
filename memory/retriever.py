"""Embedding 向量化与相似度检索。

优先尝试本地缓存的 sentence-transformers BGE 模型；
不可用（离线 / 未装权重 / 加载失败）时平滑降级为纯 Python
字符 n-gram + 词元 TF-IDF 稀疏向量余弦相似度，保证任何环境可跑通。
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import Settings
from memory.db import MemoryDB
from schemas import ExperienceItem, TermItem

_CJK_RE = re.compile(r"[一-鿿]")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")


# --------------------------------------------------------------------------- #
# Embedder 实现
# --------------------------------------------------------------------------- #
class TfidfEmbedder:
    """零依赖降级 Embedder：字符 bigram + 词元 稀疏哈希向量（L2 归一化）。"""

    name = "tfidf-fallback"

    def _tokens(self, text: str) -> List[str]:
        text = text.lower()
        words = _WORD_RE.findall(text)
        compact = re.sub(r"\s+", "", text)
        bigrams = [compact[i:i + 2] for i in range(max(0, len(compact) - 1))]
        return words + bigrams

    def embed(self, texts: List[str]) -> List[Dict[str, float]]:
        vecs: List[Dict[str, float]] = []
        for text in texts:
            tf: Dict[str, float] = {}
            for tok in self._tokens(text):
                tf[tok] = tf.get(tok, 0.0) + 1.0
            norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
            vecs.append({k: v / norm for k, v in tf.items()})
        return vecs

    @staticmethod
    def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(k, 0.0) for k, v in a.items())


def _hf_cache_has_model(model_name: str) -> bool:
    """检查 HuggingFace 本地缓存中是否存在该模型（避免无谓的重型 import）。"""
    cache_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    hub = Path(cache_home) / "hub" / ("models--" + model_name.replace("/", "--"))
    if not hub.exists():
        return False
    return any(hub.rglob("*.safetensors")) or any(hub.rglob("*.bin"))


class BGEEmbedder:
    """sentence-transformers BGE 封装；仅在本地缓存可用时启用。"""

    name = "bge"

    def __init__(self, model_name: str, offline: bool = True):
        from sentence_transformers import SentenceTransformer  # 延迟导入

        self.model = SentenceTransformer(
            model_name, local_files_only=offline
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    @staticmethod
    def cosine(a: List[float], b: List[float]) -> float:
        return float(sum(x * y for x, y in zip(a, b)))


# --------------------------------------------------------------------------- #
# 记忆检索器
# --------------------------------------------------------------------------- #
class MemoryRetriever:
    """面向 Diagnose/Plan/Rewrite 阶段的按需检索（术语 + 经验）。"""

    def __init__(self, db: MemoryDB, settings: Settings):
        self.db = db
        self.settings = settings
        self._embedder = None
        self.backend = "tfidf-fallback"
        self.backend_note = ""
        self._term_cache: Dict[int, object] = {}
        self._exp_cache: Dict[int, object] = {}
        self._try_load_bge()

    def _try_load_bge(self) -> None:
        # 离线模式下本地缓存不存在时直接走降级，避免触发 torch 等重型依赖加载
        if self.settings.embedding_offline and not _hf_cache_has_model(self.settings.embedding_model):
            self._embedder = TfidfEmbedder()
            self.backend = "tfidf-fallback"
            self.backend_note = f"本地无 {self.settings.embedding_model} 缓存，降级 TF-IDF"
            return
        try:
            self._embedder = BGEEmbedder(
                self.settings.embedding_model, offline=self.settings.embedding_offline
            )
            self.backend = "bge"
            self.backend_note = f"BGE({self.settings.embedding_model})"
        except Exception as exc:  # noqa: BLE001 - 任何加载失败都降级
            self._embedder = TfidfEmbedder()
            self.backend = "tfidf-fallback"
            self.backend_note = f"BGE 不可用，降级 TF-IDF（{type(exc).__name__}）"

    @property
    def embedder(self):
        return self._embedder

    def similarity(self, a: str, b: str) -> float:
        va, vb = self._embedder.embed([a, b])
        return float(self._embedder.cosine(va, vb))

    # ------------------------------------------------------------------ #
    def _embed_cached(self, cache: Dict[int, object], item_id: int, text: str):
        if item_id not in cache:
            cache[item_id] = self._embedder.embed([text])[0]
        return cache[item_id]

    def search_terms(self, query: str, top_k: int = 8) -> List[Tuple[TermItem, float]]:
        terms = self.db.list_terms()
        if not terms:
            return []
        qv = self._embedder.embed([query])[0]
        scored = []
        for t in terms:
            text = f"{t.term} {t.standard_form} {t.note}"
            tv = self._embed_cached(self._term_cache, t.id, text)
            score = float(self._embedder.cosine(qv, tv))
            if t.term in query or (t.standard_form and t.standard_form in query):
                score += 1.0  # 精确命中优先
            scored.append((t, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_experiences(self, query: str, top_k: int = 2) -> List[Tuple[ExperienceItem, float]]:
        exps = self.db.list_experiences()
        if not exps:
            return []
        qv = self._embedder.embed([query])[0]
        scored = []
        for e in exps:
            text = f"{e.title} {e.tags} {e.source_text}"
            ev = self._embed_cached(self._exp_cache, e.id, text)
            score = float(self._embedder.cosine(qv, ev)) * max(0.1, e.quality)
            scored.append((e, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def invalidate(self) -> None:
        """记忆库写入后调用，清空嵌入缓存。"""
        self._term_cache.clear()
        self._exp_cache.clear()
