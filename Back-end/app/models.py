# models.py
from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP, func, Numeric, String, DateTime
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import TSVECTOR
from pgvector.sqlalchemy import Vector
import uuid
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    source_url = Column(Text, nullable=True)    # 文件路径或URL
    source_type = Column(String, nullable=False)  # pdf / url
    created_at = Column(TIMESTAMP, server_default=func.now())

    chunks = relationship("DocChunk", back_populates="document", cascade="all, delete-orphan")


class DocChunk(Base):
    __tablename__ = "doc_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    tokens = Column(Integer, nullable=True)   # token 数
    page_number = Column(Integer, nullable=True)  # 对于 PDF 有意义
    tsv = Column(TSVECTOR, nullable=True)   # BM25 / 全文检索用
    created_at = Column(TIMESTAMP, server_default=func.now())

    document = relationship("Document", back_populates="chunks")
    embedding = relationship("Embedding", back_populates="chunk", uselist=False, cascade="all, delete-orphan")


class Embedding(Base):
    __tablename__ = "embeddings"

    chunk_id = Column(Integer, ForeignKey("doc_chunks.id", ondelete="CASCADE"), primary_key=True)
    embedding = Column(Vector(1536))

    chunk = relationship("DocChunk", back_populates="embedding")

class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_query = Column(Text, nullable=False)

    model = Column(String, nullable=True)                 # 聊天模型
    model_tokens_input = Column(Integer, nullable=True)   # prompt tokens
    model_tokens_output = Column(Integer, nullable=True)  # completion tokens

    # 可选：embeddings 用量（拿不到就填估算或 0）
    embedding_model = Column(String, nullable=True)
    embedding_tokens_input = Column(Integer, nullable=True)

    cost_usd = Column(Numeric(12, 6), nullable=True)      # 费用估算

    retrieved_chunk_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)

    duration_ms = Column(Integer, nullable=True)
    created_at = Column(String, server_default=func.now().cast(String))  # 或 TIMESTAMPTZ 视你现有用法

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    messages = relationship("Message", backref="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)

    role = Column(String, nullable=False)  # "system" | "user" | "assistant"
    content = Column(Text, nullable=False)

    # RAG / 生成的元数据
    citations = Column(JSONB, nullable=True)          # e.g. [{"doc_title":..., "page":..., "chunk_id":..., "text_snippet":...}, ...]
    used_chunk_ids = Column(JSONB, nullable=True)     # e.g. ["uuid-1", "uuid-2", ...]
    meta = Column(JSONB, nullable=True)               # 任意扩展，比如融合权重、检索耗时

    # 计费与用量（估算或 usage）
    model = Column(String, nullable=True)
    tokens_input = Column(Integer, nullable=True)
    tokens_output = Column(Integer, nullable=True)
    cost_usd = Column(Numeric(12, 6), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


