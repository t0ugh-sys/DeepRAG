"""Advanced retrieval utilities / 高级检索工具。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class FilterConfig:
    """Filter configuration / 过滤配置。"""

    doc_types: Optional[List[str]] = None  # ['pdf', 'markdown', 'text']
    date_from: Optional[str] = None  # 'YYYY-MM-DD'
    date_to: Optional[str] = None  # 'YYYY-MM-DD'
    min_score: Optional[float] = None
    max_results: Optional[int] = None
    paths: Optional[List[str]] = None  # path prefix or full path
    tags: Optional[List[str]] = None
    has_tables: Optional[bool] = None
    page_range: Optional[tuple[int, int]] = None  # (min, max)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict / 序列化为字典。"""
        return {
            "doc_types": self.doc_types,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "min_score": self.min_score,
            "max_results": self.max_results,
            "paths": self.paths,
            "tags": self.tags,
            "has_tables": self.has_tables,
            "page_range": self.page_range,
        }


@dataclass
class WeightConfig:
    """Weight configuration / 权重配置。"""

    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    reranker_enabled: bool = True
    mmr_lambda: float = 0.5

    def validate(self) -> None:
        """Validate weights / 校验权重。"""
        total = self.vector_weight + self.bm25_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError("vector_weight + bm25_weight must be 1.0 / 权重之和必须为 1.0")
        if not (0 <= self.vector_weight <= 1):
            raise ValueError("vector_weight must be in [0, 1] / vector_weight 需在 [0,1]")
        if not (0 <= self.bm25_weight <= 1):
            raise ValueError("bm25_weight must be in [0, 1] / bm25_weight 需在 [0,1]")
        if not (0 <= self.mmr_lambda <= 1):
            raise ValueError("mmr_lambda must be in [0, 1] / mmr_lambda 需在 [0,1]")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict / 序列化为字典。"""
        return {
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "reranker_enabled": self.reranker_enabled,
            "mmr_lambda": self.mmr_lambda,
        }


class AdvancedRetriever:
    """Advanced retriever / 高级检索器。"""

    def __init__(self) -> None:
        self.default_weights = WeightConfig()

    def dedupe_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate by path + chunk_id + text prefix / 结果去重。"""
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for r in results:
            meta = r.get("meta", {})
            key = (
                self._normalize_path(str(meta.get("path", ""))),
                meta.get("chunk_id"),
                (r.get("text") or "")[:200],
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        return deduped

    def filter_results(self, results: List[Dict[str, Any]], filter_config: FilterConfig) -> List[Dict[str, Any]]:
        """Filter results based on config / 按配置过滤结果。"""
        filtered = results

        if filter_config.doc_types:
            doc_types = {d.lower() for d in filter_config.doc_types if d}
            filtered = [
                r
                for r in filtered
                if str(r.get("meta", {}).get("doc_type", "")).lower() in doc_types
            ]

        if filter_config.paths:
            needles = [self._normalize_path(p) for p in filter_config.paths if p]
            filtered = [
                r
                for r in filtered
                if self._path_matches(r.get("meta", {}).get("path", ""), needles)
            ]

        if filter_config.min_score is not None:
            filtered = [r for r in filtered if r.get("score", 0) >= filter_config.min_score]

        if filter_config.has_tables is not None:
            filtered = [
                r
                for r in filtered
                if r.get("meta", {}).get("has_tables") == filter_config.has_tables
            ]

        if filter_config.page_range:
            min_page, max_page = filter_config.page_range
            filtered = [
                r
                for r in filtered
                if self._in_page_range(r.get("meta", {}).get("page"), min_page, max_page)
            ]

        if filter_config.tags:
            tags = {t.lower() for t in filter_config.tags if t}
            filtered = [
                r
                for r in filtered
                if tags.intersection({t.lower() for t in r.get("meta", {}).get("tags", [])})
            ]

        if filter_config.date_from or filter_config.date_to:
            filtered = [
                r
                for r in filtered
                if self._date_in_range(
                    r.get("meta", {}).get("date"),
                    filter_config.date_from,
                    filter_config.date_to,
                )
            ]

        if filter_config.max_results:
            filtered = filtered[: filter_config.max_results]

        return filtered

    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics / 统计信息。"""
        if not results:
            return {"total": 0, "avg_score": 0, "doc_types": {}, "paths": {}}

        total = len(results)
        avg_score = sum(r.get("score", 0) for r in results) / total
        doc_types: Dict[str, int] = {}
        paths: Dict[str, int] = {}
        for r in results:
            meta = r.get("meta", {})
            doc_type = str(meta.get("doc_type", "unknown")).lower()
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            path = str(meta.get("path", "unknown"))
            paths[path] = paths.get(path, 0) + 1

        return {
            "total": total,
            "avg_score": round(avg_score, 4),
            "doc_types": doc_types,
            "paths": paths,
        }

    def aggregate_by_document(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate by document path / 按文档聚合。"""
        agg: Dict[str, Dict[str, Any]] = {}
        for r in results:
            meta = r.get("meta", {})
            path = str(meta.get("path", "unknown"))
            bucket = agg.setdefault(path, {"path": path, "count": 0, "max_score": 0})
            bucket["count"] += 1
            bucket["max_score"] = max(bucket["max_score"], r.get("score", 0))
        return {"documents": list(agg.values())}

    def aggregate_by_type(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate by doc_type / 按类型聚合。"""
        agg: Dict[str, Dict[str, Any]] = {}
        for r in results:
            meta = r.get("meta", {})
            doc_type = str(meta.get("doc_type", "unknown")).lower()
            bucket = agg.setdefault(doc_type, {"doc_type": doc_type, "count": 0, "max_score": 0})
            bucket["count"] += 1
            bucket["max_score"] = max(bucket["max_score"], r.get("score", 0))
        return {"types": list(agg.values())}

    def _normalize_path(self, path: str) -> str:
        return path.replace("\\", "/").strip().lower()

    def _path_matches(self, path: str, needles: List[str]) -> bool:
        norm = self._normalize_path(str(path))
        return any(norm == n or norm.endswith(n) or n in norm for n in needles)

    def _date_in_range(self, date_value: Any, date_from: Optional[str], date_to: Optional[str]) -> bool:
        if not date_value:
            return False
        parsed = self._parse_date(str(date_value))
        if not parsed:
            return False
        if date_from:
            start = self._parse_date(date_from)
            if start and parsed < start:
                return False
        if date_to:
            end = self._parse_date(date_to)
            if end and parsed > end:
                return False
        return True

    def _parse_date(self, value: str) -> Optional[datetime]:
        value = value.strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    def _in_page_range(self, page: Any, min_page: int, max_page: int) -> bool:
        if page is None:
            return False
        try:
            page_num = int(page)
        except (TypeError, ValueError):
            return False
        return min_page <= page_num <= max_page


def create_retriever() -> AdvancedRetriever:
    """Factory helper / 创建检索器。"""
    return AdvancedRetriever()
