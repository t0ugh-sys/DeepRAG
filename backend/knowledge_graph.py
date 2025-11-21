"""
知识图谱模块

提供实体识别、关系抽取、图谱构建、图谱检索等功能
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
import json
from pathlib import Path
import re
from collections import defaultdict


@dataclass
class Entity:
    """实体"""
    name: str
    type: str  # 实体类型：person, organization, location, concept, etc.
    mentions: List[str]  # 提及的文本
    doc_paths: List[str]  # 出现的文档
    attributes: Dict[str, Any]  # 属性


@dataclass
class Relation:
    """关系"""
    subject: str  # 主体实体
    predicate: str  # 关系类型
    object: str  # 客体实体
    confidence: float  # 置信度
    source_doc: str  # 来源文档


@dataclass
class Triple:
    """三元组"""
    head: str
    relation: str
    tail: str
    weight: float = 1.0


class EntityExtractor:
    """实体抽取器"""
    
    # 预定义实体模式
    ENTITY_PATTERNS = {
        "技术": [
            r"RAG", r"LLM", r"GPT", r"BERT", r"Transformer",
            r"向量数据库", r"Milvus", r"FAISS", r"Elasticsearch",
            r"Docker", r"Kubernetes", r"FastAPI", r"Vue"
        ],
        "概念": [
            r"检索增强生成", r"语义检索", r"混合检索", r"重排序",
            r"查询改写", r"文档分块", r"嵌入模型"
        ],
        "组件": [
            r"API", r"接口", r"服务", r"模块", r"系统", r"框架"
        ],
        "操作": [
            r"部署", r"配置", r"安装", r"优化", r"监控", r"测试"
        ]
    }
    
    @classmethod
    def extract_entities(cls, text: str, doc_path: str = "") -> List[Entity]:
        """从文本中抽取实体"""
        entities = []
        
        for entity_type, patterns in cls.ENTITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_name = match.group()
                    entities.append(Entity(
                        name=entity_name,
                        type=entity_type,
                        mentions=[entity_name],
                        doc_paths=[doc_path] if doc_path else [],
                        attributes={}
                    ))
        
        return entities
    
    @classmethod
    def extract_from_documents(cls, documents: List[Dict[str, Any]]) -> Dict[str, Entity]:
        """从文档集合中抽取实体"""
        entity_dict = {}
        
        for doc in documents:
            text = doc.get("text", "")
            doc_path = doc.get("path", "")
            
            entities = cls.extract_entities(text, doc_path)
            
            for entity in entities:
                if entity.name in entity_dict:
                    # 合并实体
                    existing = entity_dict[entity.name]
                    existing.mentions.extend(entity.mentions)
                    if doc_path and doc_path not in existing.doc_paths:
                        existing.doc_paths.append(doc_path)
                else:
                    entity_dict[entity.name] = entity
        
        return entity_dict


class RelationExtractor:
    """关系抽取器"""
    
    # 预定义关系模式
    RELATION_PATTERNS = [
        # (模式, 关系类型)
        (r"(.+?)是(.+?)的(.+)", "属于"),
        (r"(.+?)用于(.+)", "用于"),
        (r"(.+?)包括(.+)", "包含"),
        (r"(.+?)支持(.+)", "支持"),
        (r"(.+?)基于(.+)", "基于"),
        (r"(.+?)依赖(.+)", "依赖"),
        (r"(.+?)集成(.+)", "集成"),
    ]
    
    @classmethod
    def extract_relations(cls, text: str, entities: List[str], doc_path: str = "") -> List[Relation]:
        """从文本中抽取关系"""
        relations = []
        
        # 简单的模式匹配
        for pattern, relation_type in cls.RELATION_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    subject = groups[0].strip()
                    obj = groups[1].strip()
                    
                    # 检查是否是已知实体
                    if subject in entities and obj in entities:
                        relations.append(Relation(
                            subject=subject,
                            predicate=relation_type,
                            object=obj,
                            confidence=0.8,
                            source_doc=doc_path
                        ))
        
        return relations


class KnowledgeGraph:
    """知识图谱"""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # entity -> [(relation, target)]
    
    def add_entity(self, entity: Entity):
        """添加实体"""
        if entity.name in self.entities:
            # 合并
            existing = self.entities[entity.name]
            existing.mentions.extend(entity.mentions)
            existing.doc_paths.extend(entity.doc_paths)
            existing.doc_paths = list(set(existing.doc_paths))
        else:
            self.entities[entity.name] = entity
    
    def add_relation(self, relation: Relation):
        """添加关系"""
        self.relations.append(relation)
        self.adjacency[relation.subject].append((relation.predicate, relation.object))
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """获取实体"""
        return self.entities.get(name)
    
    def get_neighbors(self, entity_name: str, relation_type: Optional[str] = None) -> List[str]:
        """获取邻居实体"""
        neighbors = []
        for rel, target in self.adjacency.get(entity_name, []):
            if relation_type is None or rel == relation_type:
                neighbors.append(target)
        return neighbors
    
    def find_path(self, start: str, end: str, max_depth: int = 3) -> List[List[str]]:
        """查找两个实体之间的路径"""
        if start not in self.entities or end not in self.entities:
            return []
        
        paths = []
        visited = set()
        
        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            
            if current == end:
                paths.append(path.copy())
                return
            
            if current in visited:
                return
            
            visited.add(current)
            
            for _, neighbor in self.adjacency.get(current, []):
                path.append(neighbor)
                dfs(neighbor, path, depth + 1)
                path.pop()
            
            visited.remove(current)
        
        dfs(start, [start], 0)
        return paths
    
    def get_subgraph(self, entity_name: str, depth: int = 2) -> Dict[str, Any]:
        """获取以某个实体为中心的子图"""
        if entity_name not in self.entities:
            return {"entities": [], "relations": []}
        
        visited_entities = set()
        subgraph_relations = []
        
        def bfs(start: str, max_depth: int):
            queue = [(start, 0)]
            visited = {start}
            
            while queue:
                current, d = queue.pop(0)
                visited_entities.add(current)
                
                if d >= max_depth:
                    continue
                
                for rel, target in self.adjacency.get(current, []):
                    subgraph_relations.append({
                        "subject": current,
                        "predicate": rel,
                        "object": target
                    })
                    
                    if target not in visited:
                        visited.add(target)
                        queue.append((target, d + 1))
        
        bfs(entity_name, depth)
        
        return {
            "entities": [
                {
                    "name": e,
                    "type": self.entities[e].type,
                    "doc_count": len(self.entities[e].doc_paths)
                }
                for e in visited_entities
            ],
            "relations": subgraph_relations
        }
    
    def search_entities(self, query: str, entity_type: Optional[str] = None) -> List[Entity]:
        """搜索实体"""
        results = []
        query_lower = query.lower()
        
        for entity in self.entities.values():
            if entity_type and entity.type != entity_type:
                continue
            
            if query_lower in entity.name.lower():
                results.append(entity)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        entity_type_count = defaultdict(int)
        for entity in self.entities.values():
            entity_type_count[entity.type] += 1
        
        relation_type_count = defaultdict(int)
        for relation in self.relations:
            relation_type_count[relation.predicate] += 1
        
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entity_types": dict(entity_type_count),
            "relation_types": dict(relation_type_count),
            "avg_degree": len(self.relations) / len(self.entities) if self.entities else 0
        }
    
    def export_to_json(self, filepath: str):
        """导出为 JSON"""
        data = {
            "entities": [
                {
                    "name": e.name,
                    "type": e.type,
                    "mentions": e.mentions,
                    "doc_paths": e.doc_paths,
                    "attributes": e.attributes
                }
                for e in self.entities.values()
            ],
            "relations": [
                {
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "confidence": r.confidence,
                    "source_doc": r.source_doc
                }
                for r in self.relations
            ]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def import_from_json(self, filepath: str):
        """从 JSON 导入"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for e_data in data.get("entities", []):
            entity = Entity(
                name=e_data["name"],
                type=e_data["type"],
                mentions=e_data.get("mentions", []),
                doc_paths=e_data.get("doc_paths", []),
                attributes=e_data.get("attributes", {})
            )
            self.add_entity(entity)
        
        for r_data in data.get("relations", []):
            relation = Relation(
                subject=r_data["subject"],
                predicate=r_data["predicate"],
                object=r_data["object"],
                confidence=r_data.get("confidence", 1.0),
                source_doc=r_data.get("source_doc", "")
            )
            self.add_relation(relation)


class GraphEnhancedRetriever:
    """图谱增强检索器"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
    
    def expand_query_with_graph(self, query: str) -> List[str]:
        """使用知识图谱扩展查询"""
        expanded_terms = [query]
        
        # 查找查询中的实体
        entities = self.kg.search_entities(query)
        
        for entity in entities:
            # 添加邻居实体
            neighbors = self.kg.get_neighbors(entity.name)
            expanded_terms.extend(neighbors[:3])  # 最多添加 3 个邻居
        
        return list(set(expanded_terms))
    
    def get_related_entities(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """获取与查询相关的实体"""
        entities = self.kg.search_entities(query)
        
        # 按文档数量排序
        entities.sort(key=lambda e: len(e.doc_paths), reverse=True)
        
        return [
            {
                "name": e.name,
                "type": e.type,
                "doc_count": len(e.doc_paths),
                "doc_paths": e.doc_paths[:5]
            }
            for e in entities[:top_k]
        ]
    
    def find_entity_connections(self, entity1: str, entity2: str) -> Dict[str, Any]:
        """查找两个实体之间的连接"""
        paths = self.kg.find_path(entity1, entity2)
        
        return {
            "entity1": entity1,
            "entity2": entity2,
            "connected": len(paths) > 0,
            "paths": paths[:3],  # 最多返回 3 条路径
            "shortest_path_length": min(len(p) for p in paths) if paths else None
        }


class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()
    
    def build_from_documents(self, documents: List[Dict[str, Any]]) -> KnowledgeGraph:
        """从文档构建知识图谱"""
        kg = KnowledgeGraph()
        
        # 抽取实体
        print("抽取实体...")
        entities = self.entity_extractor.extract_from_documents(documents)
        for entity in entities.values():
            kg.add_entity(entity)
        
        # 抽取关系
        print("抽取关系...")
        entity_names = list(entities.keys())
        for doc in documents:
            text = doc.get("text", "")
            doc_path = doc.get("path", "")
            relations = self.relation_extractor.extract_relations(text, entity_names, doc_path)
            for relation in relations:
                kg.add_relation(relation)
        
        print(f"构建完成: {len(kg.entities)} 个实体, {len(kg.relations)} 个关系")
        return kg


# 全局知识图谱实例
_global_kg: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    """获取全局知识图谱实例"""
    global _global_kg
    if _global_kg is None:
        _global_kg = KnowledgeGraph()
    return _global_kg


def set_knowledge_graph(kg: KnowledgeGraph):
    """设置全局知识图谱实例"""
    global _global_kg
    _global_kg = kg
