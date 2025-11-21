"""
文档管理增强模块

提供文档标签、分类、统计等功能
"""

import json
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib


class DocumentMetadata:
    """文档元数据管理"""
    
    def __init__(self, metadata_file: str = "data/documents_metadata.json"):
        self.metadata_file = metadata_file
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.load_metadata()
    
    def load_metadata(self):
        """加载元数据"""
        if Path(self.metadata_file).exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
    
    def save_metadata(self):
        """保存元数据"""
        Path(self.metadata_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def get_document_info(self, path: str) -> Optional[Dict[str, Any]]:
        """获取文档信息"""
        return self.metadata.get(path)
    
    def update_document_info(self, path: str, info: Dict[str, Any]):
        """更新文档信息"""
        if path not in self.metadata:
            self.metadata[path] = {
                "path": path,
                "created_at": datetime.now().isoformat(),
                "tags": [],
                "category": "未分类",
                "description": "",
                "custom_fields": {}
            }
        
        self.metadata[path].update(info)
        self.metadata[path]["updated_at"] = datetime.now().isoformat()
        self.save_metadata()
    
    def delete_document_info(self, path: str):
        """删除文档信息"""
        if path in self.metadata:
            del self.metadata[path]
            self.save_metadata()
    
    def add_tags(self, path: str, tags: List[str]):
        """添加标签"""
        if path not in self.metadata:
            self.update_document_info(path, {})
        
        current_tags = set(self.metadata[path].get("tags", []))
        current_tags.update(tags)
        self.metadata[path]["tags"] = list(current_tags)
        self.save_metadata()
    
    def remove_tags(self, path: str, tags: List[str]):
        """移除标签"""
        if path in self.metadata:
            current_tags = set(self.metadata[path].get("tags", []))
            current_tags.difference_update(tags)
            self.metadata[path]["tags"] = list(current_tags)
            self.save_metadata()
    
    def set_category(self, path: str, category: str):
        """设置分类"""
        if path not in self.metadata:
            self.update_document_info(path, {})
        
        self.metadata[path]["category"] = category
        self.save_metadata()
    
    def get_all_tags(self) -> List[str]:
        """获取所有标签"""
        all_tags = set()
        for doc in self.metadata.values():
            all_tags.update(doc.get("tags", []))
        return sorted(list(all_tags))
    
    def get_all_categories(self) -> List[str]:
        """获取所有分类"""
        categories = set()
        for doc in self.metadata.values():
            categories.add(doc.get("category", "未分类"))
        return sorted(list(categories))
    
    def search_by_tags(self, tags: List[str], match_all: bool = False) -> List[Dict[str, Any]]:
        """按标签搜索文档"""
        results = []
        for path, doc in self.metadata.items():
            doc_tags = set(doc.get("tags", []))
            search_tags = set(tags)
            
            if match_all:
                # 必须包含所有标签
                if search_tags.issubset(doc_tags):
                    results.append(doc)
            else:
                # 包含任一标签
                if search_tags & doc_tags:
                    results.append(doc)
        
        return results
    
    def search_by_category(self, category: str) -> List[Dict[str, Any]]:
        """按分类搜索文档"""
        return [
            doc for doc in self.metadata.values()
            if doc.get("category") == category
        ]


class DocumentStatistics:
    """文档统计"""
    
    def __init__(self, metadata: DocumentMetadata):
        self.metadata = metadata
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """获取总体统计"""
        docs = self.metadata.metadata
        
        if not docs:
            return {
                "total_documents": 0,
                "total_tags": 0,
                "total_categories": 0,
                "avg_tags_per_doc": 0.0
            }
        
        total_tags_count = sum(len(doc.get("tags", [])) for doc in docs.values())
        
        return {
            "total_documents": len(docs),
            "total_tags": len(self.metadata.get_all_tags()),
            "total_categories": len(self.metadata.get_all_categories()),
            "avg_tags_per_doc": round(total_tags_count / len(docs), 2)
        }
    
    def get_category_distribution(self) -> Dict[str, int]:
        """获取分类分布"""
        distribution = defaultdict(int)
        for doc in self.metadata.metadata.values():
            category = doc.get("category", "未分类")
            distribution[category] += 1
        return dict(distribution)
    
    def get_tag_distribution(self) -> List[Dict[str, Any]]:
        """获取标签分布"""
        tag_count = defaultdict(int)
        for doc in self.metadata.metadata.values():
            for tag in doc.get("tags", []):
                tag_count[tag] += 1
        
        # 按使用次数排序
        sorted_tags = sorted(tag_count.items(), key=lambda x: x[1], reverse=True)
        return [{"tag": tag, "count": count} for tag, count in sorted_tags]
    
    def get_recent_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的文档"""
        docs = list(self.metadata.metadata.values())
        
        # 按更新时间排序
        docs.sort(key=lambda x: x.get("updated_at", x.get("created_at", "")), reverse=True)
        
        return docs[:limit]
    
    def get_documents_by_date(self) -> Dict[str, int]:
        """按日期统计文档数量"""
        date_count = defaultdict(int)
        for doc in self.metadata.metadata.values():
            created_at = doc.get("created_at", "")
            if created_at:
                date = created_at.split("T")[0]  # 提取日期部分
                date_count[date] += 1
        
        return dict(sorted(date_count.items()))


class DocumentClassifier:
    """文档自动分类器"""
    
    # 预定义的分类规则
    CATEGORY_RULES = {
        "技术文档": ["api", "sdk", "开发", "技术", "代码", "编程", "架构"],
        "产品文档": ["产品", "功能", "需求", "PRD", "roadmap"],
        "运营文档": ["运营", "活动", "推广", "营销", "用户"],
        "设计文档": ["设计", "UI", "UX", "原型", "交互"],
        "测试文档": ["测试", "test", "QA", "bug"],
        "会议记录": ["会议", "纪要", "讨论", "决策"],
        "报告": ["报告", "分析", "总结", "汇报"],
    }
    
    @classmethod
    def auto_classify(cls, path: str, content: str = "") -> str:
        """自动分类文档"""
        path_lower = path.lower()
        content_lower = content.lower()
        
        # 基于路径和内容的关键词匹配
        for category, keywords in cls.CATEGORY_RULES.items():
            for keyword in keywords:
                if keyword in path_lower or keyword in content_lower:
                    return category
        
        # 基于文件扩展名
        if path.endswith('.pdf'):
            return "PDF文档"
        elif path.endswith(('.md', '.markdown')):
            return "Markdown文档"
        elif path.endswith(('.doc', '.docx')):
            return "Word文档"
        elif path.endswith(('.xls', '.xlsx')):
            return "Excel文档"
        
        return "未分类"
    
    @classmethod
    def suggest_tags(cls, path: str, content: str = "") -> List[str]:
        """建议标签"""
        suggested = set()
        path_lower = path.lower()
        content_lower = content.lower()
        
        # 从分类规则中提取标签
        for category, keywords in cls.CATEGORY_RULES.items():
            for keyword in keywords:
                if keyword in path_lower or keyword in content_lower:
                    suggested.add(keyword)
        
        # 基于文件类型
        if path.endswith('.pdf'):
            suggested.add('PDF')
        elif path.endswith(('.md', '.markdown')):
            suggested.add('Markdown')
        
        return list(suggested)[:5]  # 最多返回5个标签


class DocumentManager:
    """文档管理器"""
    
    def __init__(self, metadata_file: str = "data/documents_metadata.json"):
        self.metadata = DocumentMetadata(metadata_file)
        self.statistics = DocumentStatistics(self.metadata)
        self.classifier = DocumentClassifier()
    
    def add_document(self, path: str, auto_classify: bool = True, auto_tag: bool = True, content: str = ""):
        """添加文档"""
        info = {
            "path": path,
            "created_at": datetime.now().isoformat()
        }
        
        if auto_classify:
            info["category"] = self.classifier.auto_classify(path, content)
        
        if auto_tag:
            info["tags"] = self.classifier.suggest_tags(path, content)
        
        self.metadata.update_document_info(path, info)
    
    def update_document(self, path: str, **kwargs):
        """更新文档信息"""
        self.metadata.update_document_info(path, kwargs)
    
    def delete_document(self, path: str):
        """删除文档"""
        self.metadata.delete_document_info(path)
    
    def get_document(self, path: str) -> Optional[Dict[str, Any]]:
        """获取文档信息"""
        return self.metadata.get_document_info(path)
    
    def list_documents(self, category: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """列出文档"""
        if category:
            return self.metadata.search_by_category(category)
        elif tags:
            return self.metadata.search_by_tags(tags)
        else:
            return list(self.metadata.metadata.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "overall": self.statistics.get_overall_stats(),
            "category_distribution": self.statistics.get_category_distribution(),
            "tag_distribution": self.statistics.get_tag_distribution(),
            "recent_documents": self.statistics.get_recent_documents()
        }


# 全局实例
_global_manager: Optional[DocumentManager] = None


def get_document_manager() -> DocumentManager:
    """获取全局文档管理器实例"""
    global _global_manager
    if _global_manager is None:
        _global_manager = DocumentManager()
    return _global_manager
