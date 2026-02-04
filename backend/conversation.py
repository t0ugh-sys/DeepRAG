"""Conversation storage and management / 会话存储与管理。"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """Conversation message / 会话消息。"""

    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    sources: List[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Conversation:
    """Conversation entity / 会话实体。"""

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
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        messages = [Message(**msg) for msg in data.get("messages", [])]
        return cls(
            id=data["id"],
            title=data["title"],
            messages=messages,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            namespace=data.get("namespace", "default"),
            metadata=data.get("metadata"),
        )


class ConversationManager:
    """Conversation manager / 会话管理。"""

    def __init__(self, storage_dir: str = "data/conversations") -> None:
        """Initialize storage directory / 初始化存储目录。"""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_conversation(
        self,
        title: str = "New conversation / 新对话",
        namespace: str = "default",
    ) -> Conversation:
        """Create a new conversation / 新建会话。"""
        conv_id = f"conv_{int(time.time() * 1000)}"
        now = time.time()

        conversation = Conversation(
            id=conv_id,
            title=title,
            messages=[],
            created_at=now,
            updated_at=now,
            namespace=namespace,
        )

        self._save_conversation(conversation)
        return conversation

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Get a conversation by id / 根据 ID 获取会话。"""
        file_path = self.storage_dir / f"{conv_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return Conversation.from_dict(data)

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        sources: List[Dict[str, Any]] = None,
    ) -> Optional[Conversation]:
        """Add a message / 添加消息。"""
        conversation = self.get_conversation(conv_id)
        if not conversation:
            return None

        message = Message(
            role=role,
            content=content,
            timestamp=time.time(),
            sources=sources or [],
        )

        conversation.messages.append(message)
        conversation.updated_at = time.time()

        # Use the first user message as title / 首条用户消息作为标题
        if role == "user" and len(conversation.messages) == 1:
            conversation.title = content[:50] + ("..." if len(content) > 50 else "")

        self._save_conversation(conversation)
        return conversation

    def list_conversations(
        self,
        namespace: str = "default",
        limit: int = 50,
        query: str | None = None,
    ) -> List[Dict[str, Any]]:
        """List conversations / 获取会话列表。"""
        conversations: List[Dict[str, Any]] = []

        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data.get("namespace") == namespace:
                    if query:
                        query_lower = query.lower()
                        title = str(data.get("title", "")).lower()
                        content = " ".join([m.get("content", "") for m in data.get("messages", [])]).lower()
                        if query_lower not in title and query_lower not in content:
                            continue
                    conversations.append(
                        {
                            "id": data["id"],
                            "title": data["title"],
                            "created_at": data["created_at"],
                            "updated_at": data["updated_at"],
                            "message_count": len(data.get("messages", [])),
                        }
                    )
            except Exception:
                continue

        conversations.sort(key=lambda x: x["updated_at"], reverse=True)
        return conversations[:limit]

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation / 删除会话。"""
        file_path = self.storage_dir / f"{conv_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def get_context_messages(
        self,
        conv_id: str,
        max_messages: int = 10,
    ) -> List[Dict[str, str]]:
        """Get recent messages for context / 获取上下文消息。"""
        conversation = self.get_conversation(conv_id)
        if not conversation:
            return []

        # Keep the latest N messages / 只保留最近 N 条消息
        recent_messages = conversation.messages[-max_messages:]

        return [{"role": msg.role, "content": msg.content} for msg in recent_messages]

    def _save_conversation(self, conversation: Conversation) -> None:
        """Persist a conversation / 保存会话。"""
        file_path = self.storage_dir / f"{conversation.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conversation.to_dict(), f, ensure_ascii=False, indent=2)
