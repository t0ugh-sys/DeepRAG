"""
多轮对话管理模块
支持对话历史、上下文管理
"""
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Message:
    """对话消息"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    sources: List[Dict[str, Any]] = None  # 引用来源
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Conversation:
    """对话会话"""
    id: str
    title: str
    messages: List[Message]
    created_at: float
    updated_at: float
    namespace: str = "default"
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "namespace": self.namespace,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        messages = [Message(**msg) for msg in data.get("messages", [])]
        return cls(
            id=data["id"],
            title=data["title"],
            messages=messages,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            namespace=data.get("namespace", "default"),
            metadata=data.get("metadata")
        )


class ConversationManager:
    """对话管理器"""
    
    def __init__(self, storage_dir: str = "data/conversations"):
        """
        初始化对话管理器
        
        Args:
            storage_dir: 对话存储目录
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def create_conversation(
        self, 
        title: str = "新对话", 
        namespace: str = "default"
    ) -> Conversation:
        """创建新对话"""
        conv_id = f"conv_{int(time.time() * 1000)}"
        now = time.time()
        
        conversation = Conversation(
            id=conv_id,
            title=title,
            messages=[],
            created_at=now,
            updated_at=now,
            namespace=namespace
        )
        
        self._save_conversation(conversation)
        return conversation
    
    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """获取对话"""
        file_path = self.storage_dir / f"{conv_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return Conversation.from_dict(data)
    
    def add_message(
        self, 
        conv_id: str, 
        role: str, 
        content: str,
        sources: List[Dict[str, Any]] = None
    ) -> Optional[Conversation]:
        """添加消息到对话"""
        conversation = self.get_conversation(conv_id)
        if not conversation:
            return None
        
        message = Message(
            role=role,
            content=content,
            timestamp=time.time(),
            sources=sources or []
        )
        
        conversation.messages.append(message)
        conversation.updated_at = time.time()
        
        # 自动更新标题（使用第一条用户消息）
        if role == "user" and len(conversation.messages) == 1:
            conversation.title = content[:50] + ("..." if len(content) > 50 else "")
        
        self._save_conversation(conversation)
        return conversation
    
    def list_conversations(
        self, 
        namespace: str = "default", 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """列出对话列表"""
        conversations = []
        
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if data.get("namespace") == namespace:
                    # 只返回摘要信息
                    conversations.append({
                        "id": data["id"],
                        "title": data["title"],
                        "created_at": data["created_at"],
                        "updated_at": data["updated_at"],
                        "message_count": len(data.get("messages", []))
                    })
            except Exception:
                continue
        
        # 按更新时间倒序排列
        conversations.sort(key=lambda x: x["updated_at"], reverse=True)
        return conversations[:limit]
    
    def delete_conversation(self, conv_id: str) -> bool:
        """删除对话"""
        file_path = self.storage_dir / f"{conv_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def get_context_messages(
        self, 
        conv_id: str, 
        max_messages: int = 10
    ) -> List[Dict[str, str]]:
        """
        获取对话上下文（用于 LLM）
        
        Args:
            conv_id: 对话 ID
            max_messages: 最大消息数
            
        Returns:
            格式化的消息列表 [{"role": "user", "content": "..."}]
        """
        conversation = self.get_conversation(conv_id)
        if not conversation:
            return []
        
        # 取最近的 N 条消息
        recent_messages = conversation.messages[-max_messages:]
        
        return [
            {"role": msg.role, "content": msg.content}
            for msg in recent_messages
        ]
    
    def _save_conversation(self, conversation: Conversation) -> None:
        """保存对话到文件"""
        file_path = self.storage_dir / f"{conversation.id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversation.to_dict(), f, ensure_ascii=False, indent=2)
