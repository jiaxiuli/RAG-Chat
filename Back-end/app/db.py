from datetime import datetime
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from models import Base, Document, DocChunk, Embedding, Conversation, Message
from embeddings import get_embedding
from decimal import Decimal
from models import RequestLog
import re

# 替换成你的数据库配置
DATABASE_URL = "postgresql://rag_user:123456@localhost:5432/ragchat"

# 创建数据库引擎
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

PRICING = {
    # 聊天模型：美元/1k tokens
    "gpt-4o-mini": {
        "input_per_1k": Decimal("0.150"),   # 示例价，替换成真实价格
        "output_per_1k": Decimal("0.600"),
    },
    # Embeddings 模型
    "text-embedding-3-small": {
        "input_per_1k": Decimal("0.020"),   # 示例价
    },
    "text-embedding-3-large": {
        "input_per_1k": Decimal("0.130"),   # 示例价
    },
}

def clean_text(text):
    # 去掉非标准字符
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # 把多余空格合并
    text = re.sub(r"\s+", " ", text).strip()
    return text

def insert_document(db: Session, title: str, source_url: str, source_type: str):
    doc = Document(title=title, source_url=source_url, source_type=source_type)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc

def insert_chunk(db: Session, document_id: int, chunk_index: int, text: str, tokens: int, page_number: int = None):
    chunk = DocChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        tokens=tokens,
        page_number=page_number,
        # tsv=text  # ⚠️ 这里简化，tsvector 需要额外触发 `to_tsvector`
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk

def insert_embedding(db: Session, chunk_id: int, vector: list[float]):
    emb = Embedding(chunk_id=chunk_id, embedding=vector)
    db.add(emb)
    db.commit()
    db.refresh(emb)
    return emb

# 依赖：获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- 向量检索（Top N） ----------
def search_by_vector(db: Session, user_query: str, top_n: int = 60) -> List[Dict[str, Any]]:
    qvec = get_embedding(user_query)  # list[float] / numpy array 都可以
    # pgvector 的 SQLAlchemy 扩展支持 .cosine_distance()
    rows = (
        db.query(
            DocChunk.id.label("chunk_id"),
            DocChunk.document_id,
            DocChunk.chunk_index,
            DocChunk.page_number,
            DocChunk.tokens,
            DocChunk.text,
            Document.title,
            Document.source_url,
            Embedding.embedding.cosine_distance(qvec).label("vec_dist"),
        )
        .join(Embedding, Embedding.chunk_id == DocChunk.id)
        .join(Document, Document.id == DocChunk.document_id)
        .order_by(Embedding.embedding.cosine_distance(qvec))  # 距离越小越相似
        .limit(top_n)
        .all()
    )

    # 统一成你后续融合需要的字典结构（把“距离”转成“相似度”便于直觉）
    out = []
    for r in rows:
        r = r._asdict()
        dist = float(r.pop("vec_dist") or 0.0)
        sim = 1.0 - dist  # 余弦相似度 = 1 - 距离
        r["vec_sim"] = sim
        out.append(r)
    return out

# ---------- BM25/FTS 检索（Top M） ----------
def search_by_bm25(db: Session, user_query: str, top_m: int = 60) -> List[Dict[str, Any]]:
    # 用 simple 配置；如果你之后接入中文分词，再换成对应配置
    tsq = func.plainto_tsquery("simple", user_query)
    tsv = func.to_tsvector("simple", DocChunk.text)
    rank = func.ts_rank(tsv, tsq)  # BM25 风格的相关性分数

    rows = (
        db.query(
            DocChunk.id.label("chunk_id"),
            DocChunk.document_id,
            DocChunk.chunk_index,
            DocChunk.page_number,
            DocChunk.tokens,
            DocChunk.text,
            Document.title,
            Document.source_url,
            rank.label("bm25_score"),
        )
        .join(Document, Document.id == DocChunk.document_id)
        .filter(tsv.op("@@")(tsq))
        .order_by(desc(rank))
        .limit(top_m)
        .all()
    )

    out = []
    for r in rows:
        r = r._asdict()
        r["bm25_score"] = float(r["bm25_score"] or 0.0)
        out.append(r)
    return out

# ---------- RRF 融合 + 多样性控制 ----------
def fuse_rrf(
    vec_hits: List[Dict[str, Any]],
    bm_hits: List[Dict[str, Any]],
    final_k: int = 12,
    per_doc_limit: int = 3,
    k_rrf: int = 60,
) -> List[Dict[str, Any]]:
    # 生成 rank 索引
    def make_rank_map(hits: List[Dict[str, Any]], key="chunk_id") -> Dict[Any, int]:
        return {h[key]: i + 1 for i, h in enumerate(hits)}  # rank 从 1 开始

    rank_vec = make_rank_map(vec_hits)
    rank_bm = make_rank_map(bm_hits)

    # 累积分数
    merged: Dict[Any, Dict[str, Any]] = {}

    def _push(hit: Dict[str, Any], extra: Dict[str, float]):
        cid = hit["chunk_id"]
        if cid not in merged:
            merged[cid] = {**hit, "rrf_score": 0.0}
        merged[cid].update(extra)

    for h in vec_hits:
        r = rank_vec[h["chunk_id"]]
        _push(h, {"rrf_score": merged.get(h["chunk_id"], {}).get("rrf_score", 0.0) + 1.0 / (k_rrf + r)})

    for h in bm_hits:
        r = rank_bm[h["chunk_id"]]
        _push(h, {"rrf_score": merged.get(h["chunk_id"], {}).get("rrf_score", 0.0) + 1.0 / (k_rrf + r)})

    # 排序 + 每文档不超过 N 条
    by_score = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)

    picked: List[Dict[str, Any]] = []
    per_doc_cnt: Dict[Any, int] = {}
    for h in by_score:
        did = h["document_id"]
        if per_doc_cnt.get(did, 0) >= per_doc_limit:
            continue
        per_doc_cnt[did] = per_doc_cnt.get(did, 0) + 1
        picked.append(h)
        if len(picked) >= final_k:
            break

    return picked

# ---------- 一站式检索（对外） ----------
def search_chunks(
    db: Session,
    user_query: str,
    k: int = 12,
    vec_limit: int = 60,
    bm_limit: int = 60,
    per_doc_limit: int = 3,
) -> List[Dict[str, Any]]:
    vec_hits = search_by_vector(db, user_query, top_n=vec_limit)
    bm_hits  = search_by_bm25(db, user_query, top_m=bm_limit)
    final = fuse_rrf(vec_hits, bm_hits, final_k=k, per_doc_limit=per_doc_limit)
    return final

# ---------- 上下文与引用 ----------
def build_context_and_citations(hits: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    # 组上下文：同文档按 chunk_index 升序
    ordered = sorted(hits, key=lambda x: (x["document_id"], x["chunk_index"]))
    context = "\n---\n".join(h["text"] for h in ordered)

    # 去重后的引用：文档 + 页码
    citations = []
    seen = set()
    for h in ordered:
        key = (h["document_id"], h.get("page_number"))
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "title": h.get("title"),
            "page": h.get("page_number"),
            "url": h.get("source_url"),
        })
    return context, citations

def estimate_cost_usd(
    model: Optional[str],
    model_tokens_input: int,
    model_tokens_output: int,
    embedding_model: Optional[str],
    embedding_tokens_input: int,
) -> Decimal:
    """按 tokens 和单价估算费用（美元）"""
    total = Decimal("0")

    if model and model in PRICING:
        p = PRICING[model]
        inp = Decimal(model_tokens_input or 0) / Decimal(1000)
        outp = Decimal(model_tokens_output or 0) / Decimal(1000)
        total += inp * p.get("input_per_1k", Decimal("0"))
        total += outp * p.get("output_per_1k", Decimal("0"))

    if embedding_model and embedding_model in PRICING:
        e = PRICING[embedding_model]
        einp = Decimal(embedding_tokens_input or 0) / Decimal(1000)
        total += einp * e.get("input_per_1k", Decimal("0"))

    return total.quantize(Decimal("0.000001"))

def create_request_log(
    db: Session,
    *,
    user_query: str,
    model: Optional[str],
    model_tokens_input: int,
    model_tokens_output: int,
    retrieved_chunk_ids: Optional[List[Any]],
    duration_ms: int,
    embedding_model: Optional[str] = None,
    embedding_tokens_input: int = 0,
    cost_usd: Optional[Decimal] = None,
) -> RequestLog:
    if cost_usd is None:
        cost_usd = estimate_cost_usd(
            model=model,
            model_tokens_input=model_tokens_input,
            model_tokens_output=model_tokens_output,
            embedding_model=embedding_model,
            embedding_tokens_input=embedding_tokens_input,
        )

    log = RequestLog(
        user_query=user_query,
        model=model,
        model_tokens_input=model_tokens_input,
        model_tokens_output=model_tokens_output,
        embedding_model=embedding_model,
        embedding_tokens_input=embedding_tokens_input,
        cost_usd=cost_usd,
        retrieved_chunk_ids=[uuid for uuid in (retrieved_chunk_ids or [])],
        duration_ms=duration_ms,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log
# db.py
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from models import Document, DocChunk

def list_documents(
    db: Session,
    cursor: Optional[datetime] = None,   # 上一页最后一个文档的 created_at
    page_size: int = 20,
) -> Dict[str, Any]:
    """
    Cursor-based 分页：
      - 按 created_at DESC 排序
      - cursor: 上一页最后一条记录的 created_at
    """
    # 基础查询（聚合统计 chunk 数 + 最近 ingest 时间）
    base = (
        db.query(
            Document.id.label("id"),
            Document.title.label("title"),
            Document.source_url.label("source_url"),
            Document.source_type.label("source_type"),
            Document.created_at.label("created_at"),
            func.count(DocChunk.id).label("chunk_count"),
            func.max(DocChunk.created_at).label("last_ingested_at"),
        )
        .outerjoin(DocChunk, DocChunk.document_id == Document.id)
        .group_by(Document.id)
        .order_by(desc(Document.created_at))   # 按上传时间倒序
    )

    # 如果有 cursor → 只取 cursor 之前的
    if cursor:
        base = base.filter(Document.created_at < cursor)

    # 取一页
    rows = base.limit(page_size).all()
    items = [r._asdict() for r in rows]

    # 下一个 cursor（最后一条的 created_at）
    next_cursor = None
    if items:
        next_cursor = items[-1]["created_at"]

    return {
        "page_size": page_size,
        "items": items,
        "next_cursor": next_cursor,   # 前端下次请求时带上这个
        "has_more": len(items) == page_size,  # 是否可能还有更多
    }

def get_document_detail(
    db: Session,
    document_id,
    include_chunks: bool = True,
    chunk_limit: int = 20,
) -> Dict[str, Any]:
    """
    返回单个文档基本信息 + （可选）部分切片预览
    """
    doc = (
        db.query(Document)
        .filter(Document.id == document_id)
        .first()
    )
    if not doc:
        return {}

    result = {
        "id": doc.id,
        "title": doc.title,
        "source_url": doc.source_url,
        "source_type": getattr(doc, "source_type", None),
        "created_at": doc.created_at,
    }

    # 统计切片数量
    chunk_count = db.query(func.count(DocChunk.id)).filter(DocChunk.document_id == document_id).scalar() or 0
    result["chunk_count"] = chunk_count

    if include_chunks:
        chunks = (
            db.query(
                DocChunk.id.label("chunk_id"),
                DocChunk.chunk_index,
                DocChunk.page_number,
                DocChunk.tokens,
                DocChunk.text,
                DocChunk.created_at,
            )
            .filter(DocChunk.document_id == document_id)
            .order_by(DocChunk.chunk_index.asc())
            .limit(chunk_limit)
            .all()
        )
        result["chunks_preview"] = [c._asdict() for c in chunks]
        result["chunks_preview_limit"] = chunk_limit

    return result

def delete_document_and_related(db: Session, document_id):
    """
    删除文档及其所有相关 chunk/embedding
    """
    # 先查找文档
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # 找到所有 chunks
    chunks = db.query(DocChunk).filter(DocChunk.document_id == document_id).all()
    chunk_ids = [c.id for c in chunks]

    if chunk_ids:
        # 删除 embeddings
        db.query(Embedding).filter(Embedding.chunk_id.in_(chunk_ids)).delete(synchronize_session=False)
        # 删除 chunks
        db.query(DocChunk).filter(DocChunk.document_id == document_id).delete(synchronize_session=False)

    # 删除文档
    db.delete(doc)
    db.commit()

    return {"message": f"Document {document_id} and all related data have been deleted"}

def get_conversation(db: Session, conversation_id):
    return db.query(Conversation).filter(Conversation.id == conversation_id).first()

def create_conversation(db: Session, title: str = "New Chat") -> Conversation:
    conv = Conversation(title=title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv

def get_or_create_conversation(db: Session, conversation_id: Optional[str] = None, title: Optional[str] = None) -> Conversation:
    """
    如果传了 conversation_id 就取原会话，否则新建一个。
    """
    if conversation_id:
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv:
            return conv

    conv = Conversation(title=title or "New Chat")
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def add_message(
    db: Session,
    conversation_id,
    role: str,
    content: str,
    *,
    citations: Optional[List[Dict[str, Any]]] = None,
    used_chunk_ids: Optional[List[str]] = None,
    model: Optional[str] = None,
    tokens_input: Optional[int] = None,
    tokens_output: Optional[int] = None,
    cost_usd: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Message:
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        citations=citations,
        used_chunk_ids=used_chunk_ids,
        model=model,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        cost_usd=cost_usd,
        meta=meta or {},
    )
    db.add(msg)
    # 同时更新会话的 updated_at
    db.query(Conversation).filter(Conversation.id == conversation_id).update({Conversation.updated_at: func.now()})
    db.commit()
    db.refresh(msg)
    return msg

def list_messages(db: Session, conversation_id, limit: int = 100) -> List[Message]:
    return (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .limit(limit)
        .all()
    )


def list_conversations_cursor(
    db: Session,
    cursor: Optional[datetime] = None,   # 上一页最后一条的 updated_at
    page_size: int = 20,
) -> Dict[str, Any]:
    """
    返回最近更新的会话列表（不含消息），按 updated_at DESC。
    支持基于 updated_at 的 cursor 分页。
    """
    q = db.query(
        Conversation.id.label("id"),
        Conversation.title.label("title"),
        Conversation.created_at.label("created_at"),
        Conversation.updated_at.label("updated_at"),
    ).order_by(desc(Conversation.updated_at))

    if cursor:
        q = q.filter(Conversation.updated_at < cursor)

    rows = q.limit(page_size).all()
    items = [r._asdict() for r in rows]

    next_cursor = items[-1]["updated_at"] if items else None
    return {
        "page_size": page_size,
        "items": items,
        "next_cursor": next_cursor,
        "has_more": len(items) == page_size,
    }

def delete_conversation_and_history(db: Session, conversation_id) -> Dict[str, Any]:
    """
    删除一个会话及其所有消息（手动级联删除，避免对模型关系的依赖）
    """
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # 先删消息，再删会话（事务内一次提交）
    db.query(Message).filter(Message.conversation_id == conversation_id).delete(synchronize_session=False)
    db.delete(conv)
    db.commit()

    return {"message": f"Conversation {conversation_id} and all related messages have been deleted"}


