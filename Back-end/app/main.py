import asyncio
import os
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Depends, Query,WebSocket, WebSocketDisconnect, HTTPException
from sqlalchemy.orm import Session
from db import (engine, get_db, insert_document, insert_chunk, insert_embedding,
                search_chunks, build_context_and_citations, create_request_log,
                list_documents, get_document_detail, delete_document_and_related,
                add_message, list_messages, list_conversations_cursor, create_conversation,
                get_conversation, delete_conversation_and_history)
from db import estimate_cost_usd, PRICING
from ingest import parse_pdf, chunk_text
from embeddings import get_embedding
from models import Base
from chat import select_chunks_by_budget, build_prompt, call_llm_and_parse
from chat import stream_rag_answer
import time
from typing import Optional, List, Dict, Any
import uuid
from fastapi.middleware.cors import CORSMiddleware

# 创建数据库表
Base.metadata.create_all(bind=engine)

app = FastAPI()

origins = [
    "http://localhost:3000",   # 你的前端本地端口
    "http://127.0.0.1:5173",   # 另一个常见前端端口
    # 你也可以加生产环境的域名，比如 "https://your-frontend.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # 允许的源
    allow_credentials=True,
    allow_methods=["*"],            # 允许的 HTTP 方法（GET/POST/DELETE 等）
    allow_headers=["*"],            # 允许的 HTTP 头
)

@app.post("/upload/")
def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    pdf_str = parse_pdf(file.file)
    chunks = chunk_text(pdf_str)
    print(chunks)
    doc = insert_document(db, title=file.filename, source_url="", source_type="pdf")
    emb = []
    for i, chunk in enumerate(chunks):
        print(chunk)
        chunk_obj = insert_chunk(db, doc.id, i, chunk, 0, 1)
        vector = get_embedding(chunk)
        insert_embedding(db, chunk_obj.id, vector)

    return {"message": f"文件 {file.filename} 已成功入库，共 {len(chunks)} 个切片"}

@app.get("/search")
def search_api(q: str = Query(..., description="用户问题"),
               k: int = 12,
               db: Session = Depends(get_db)):
    hits = search_chunks(db, user_query=q, k=k)
    context, citations = build_context_and_citations(hits)
    return {
        "query": q,
        "k": k,
        "hits": hits,                 # 每条含：chunk_id/doc_id/chunk_index/page/text/title/source_url/rrf_score
        "citations": citations,       # 前端的“来源卡片”
        "context_preview": context[:500]
    }

@app.get("/query")
def qa_api(
    q: str = Query(..., description="用户问题"),
    k: int = 12,
    context_budget: int = 3000,
    model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",   # 你实际用哪个就写哪个
    db: Session = Depends(get_db),
):
    start = time.perf_counter()

    # 1) 检索（融合）
    hits = search_chunks(db, user_query=q, k=k)

    # 2) 在预算内选片段
    used_chunks = select_chunks_by_budget(hits, context_token_budget=context_budget)

    # 3) 组装 Prompt 并调用模型
    prompt = build_prompt(q, used_chunks)
    result = call_llm_and_parse(q, prompt, model=model)

    # 4) 解析 usage（tokens）
    #    这里 result["raw"] 是原始字符串；如果你在 qa.call_llm_and_parse 里也返回了 resp 对象，就直接读 resp.usage
    #    我们示例里不返回 resp 对象，因此额外提供一个钩子（可选）：
    model_tokens_input = 0
    model_tokens_output = 0
    try:
        # 如果你改写了 call_llm_and_parse 返回 usage，可在此读取
        usage = result.get("usage") or {}  # e.g., {"prompt_tokens":..., "completion_tokens":...}
        if usage:
            model_tokens_input = int(usage.get("prompt_tokens") or 0)
            model_tokens_output = int(usage.get("completion_tokens") or 0)
    except Exception:
        pass

    # 如果没有 usage，但你愿意估算，可用 tiktoken 对 prompt 估算一下输入 tokens：
    if model_tokens_input == 0:
        try:
            from chat import estimate_tokens
            model_tokens_input = estimate_tokens(prompt)
        except Exception:
            model_tokens_input = 0

    # embeddings 用量：如果你在 embeddings.get_embedding 里能返回 usage，请在检索阶段记录并带出来。
    # 这里先用“选入上下文的文本粗估”作为兜底（可选）
    embedding_tokens_input = 0
    try:
        from chat import estimate_tokens
        # 估算：把用户 query 和（用于检索的）query 一起算；这里先用 query 粗估
        embedding_tokens_input = estimate_tokens(q)
    except Exception:
        pass

    duration_ms = int(round((time.perf_counter() - start) * 1000))

    # 收集选入上下文的 chunk_id，用于回放
    retrieved_chunk_ids = [h["chunk_id"] for h in used_chunks]

    # 5) 落库
    log = create_request_log(
        db,
        user_query=q,
        model=model,
        model_tokens_input=model_tokens_input,
        model_tokens_output=model_tokens_output,
        retrieved_chunk_ids=retrieved_chunk_ids,
        duration_ms=duration_ms,
        embedding_model=embedding_model,
        embedding_tokens_input=embedding_tokens_input,
    )

    # 6) 返回给前端
    return {
        "query": q,
        "model": model,
        "context_budget": context_budget,
        "used_chunks": [
            {
                "chunk_id": h["chunk_id"],
                "document_id": h["document_id"],
                "chunk_index": h["chunk_index"],
                "page": h.get("page_number"),
                "title": h.get("title"),
                "url": h.get("source_url"),
                "tokens": h.get("tokens"),
            } for h in used_chunks
        ],
        "answer": result.get("parsed", {}).get("answer"),
        "citations": result.get("parsed", {}).get("citations", []),
        "confidence": result.get("parsed", {}).get("confidence"),
        "metrics": {
            "duration_ms": duration_ms,
            "model_tokens_input": model_tokens_input,
            "model_tokens_output": model_tokens_output,
            "embedding_tokens_input": embedding_tokens_input,
            # cost_usd 可由后端估算，也可返回 DB 里存的
        },
        "log_id": str(log.id),
    }
@app.get("/documents")
def documents_api(
        cursor: Optional[datetime] = Query(None, description="上一页最后一个文档的 created_at"),
        page_size: int = Query(20, ge=1, le=100),
        db: Session = Depends(get_db),
):
    """
    列出已上传的文件（分页 + 可搜索）
    响应示例：
    {
      "page":1,"page_size":20,"total":3,
      "items":[
        {
          "id":"...","title":"xxx.pdf","source_url":"",
          "source_type":"pdf","created_at":"...",
          "chunk_count":42,"last_ingested_at":"..."
        }
      ]
    }
    """
    return list_documents(db, cursor=cursor, page_size=page_size)

@app.get("/documents/{document_id}")
def document_detail_api(
    document_id: str,
    include_chunks: bool = Query(True, description="是否包含切片预览"),
    chunk_limit: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """
    获取单个文档详情（可包含部分切片预览）
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document_id (must be UUID)")

    detail = get_document_detail(
        db,
        document_id=doc_uuid,
        include_chunks=include_chunks,
        chunk_limit=chunk_limit,
    )
    if not detail:
        raise HTTPException(status_code=404, detail="Document not found")

    return detail

@app.delete("/documents/{document_id}")
def delete_document_api(document_id: str, db: Session = Depends(get_db)):
    import uuid
    try:
        doc_uuid = uuid.UUID(document_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document_id (must be UUID)")

    result = delete_document_and_related(db, doc_uuid)
    return result


@app.get("/conversations/{conversation_id}/messages")
def get_history(conversation_id: str, db: Session = Depends(get_db)):
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid conversation_id")
    msgs = list_messages(db, conv_uuid, limit=200)
    return [{"id": str(m.id), "role": m.role, "content": m.content, "created_at": m.created_at, "citations": m.citations} for m in msgs]

# WebSocket：流式聊天
@app.websocket("/ws/chat")
async def ws_chat(
    websocket: WebSocket,
    k: int = 12,
    context_budget: int = 3000,
    model: str = "gpt-4o-mini",
):
    """
    Client sends user messages like:
      {
        "type": "user_message",
        "conversation_id": "<uuid or null or ''>",
        "content": "..."
      }

    Server behavior:
      - if conversation_id is null/empty: create a new conversation, ACK with its id, then handle the message
      - if conversation_id is a valid UUID: append to that conversation (404 if not found)
      - streams {"type":"delta","text": "..."} chunks
      - sends one final {"type":"done", ...} with answer & metrics
    """
    await websocket.accept()
    # Initial connection ACK (no conversation chosen yet)
    await websocket.send_json({"type": "ack"})

    # Track the last conversation used on this socket (optional)
    current_conv_id: Optional[uuid.UUID] = None

    try:
        while True:
            data = await websocket.receive_json()
            if not isinstance(data, dict) or data.get("type") != "user_message":
                await websocket.send_json({"type": "error", "message": "Invalid message payload"})
                continue

            user_text = (data.get("content") or "").strip()
            if not user_text:
                await websocket.send_json({"type": "error", "message": "Empty message"})
                continue

            # ---- Conversation selection/creation per message ----
            payload_conv_raw = (data.get("conversation_id") or "").strip() if isinstance(data.get("conversation_id"), str) else data.get("conversation_id")
            conv = None

            if payload_conv_raw:
                # Validate & fetch existing conversation
                try:
                    conv_uuid = uuid.UUID(str(payload_conv_raw))
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Invalid conversation_id"})
                    continue

                with next(get_db()) as db:
                    conv = get_conversation(db, conv_uuid)

                if not conv:
                    await websocket.send_json({"type": "error", "message": "Conversation not found"})
                    continue

                # Optional locking: disallow switching mid-connection
                # if current_conv_id and current_conv_id != conv.id:
                #     await websocket.send_json({"type": "error", "message": "This socket is bound to another conversation"})
                #     continue

                current_conv_id = conv.id
                # ACK with the conversation we will use
                await websocket.send_json({"type": "ack_conversation", "conversation_id": str(conv.id)})

            else:
                # Create a new conversation
                with next(get_db()) as db:
                    conv = create_conversation(db, title=user_text[:40] or "New Chat")
                current_conv_id = conv.id
                await websocket.send_json({"type": "ack_create_conversation", "conversation_id": str(conv.id)})

            # ---- Persist user message ----
            with next(get_db()) as db:
                _ = add_message(db, conv.id, role="user", content=user_text)

            # ---- Retrieval + selection within budget ----
            with next(get_db()) as db:
                hits = search_chunks(db, user_query=user_text, k=k)
            used_chunks = select_chunks_by_budget(hits, context_token_budget=context_budget)
            used_chunk_ids = [str(h["chunk_id"]) for h in used_chunks]

            # (optional) simple citations from retrieved chunks
            citations_payload = [
                {
                    "doc_title": h.get("title") or "",
                    "page": h.get("page_number"),
                    "chunk_id": str(h["chunk_id"]),
                }
                for h in used_chunks[:3]
            ]

            # ---- Stream generation ----
            start = time.perf_counter()
            full_text = ""
            prompt_tokens_est = 0
            completion_tokens_est = 0

            try:
                gen = stream_rag_answer(user_text, used_chunks, model=model)
            except Exception as e:
                await websocket.send_json({"type": "error", "message": f"Error creating stream: {e}"})
                continue

            try:
                for piece in gen:
                    t = piece.get("type")
                    if t == "delta":
                        txt = piece.get("text", "")
                        if txt:
                            full_text += txt
                            # Normal send:
                            await websocket.send_json({"type": "delta", "text": txt})
                            # Or throttled send (if you added helpers):
                            # await send_rate_limited(websocket, txt, cps=60, min_chunk=8, tick_ms=50)

                    elif t == "done_meta":
                        prompt_tokens_est = int(piece.get("prompt_tokens", 0))
                        completion_tokens_est = int(piece.get("completion_tokens", 0))

                    elif t == "error":
                        await websocket.send_json({"type": "error", "message": piece.get("message", "stream error")})
                        full_text = ""
                        break
            except Exception as e:
                await websocket.send_json({"type": "error", "message": f"Streaming error: {e}"})
                continue

            duration_ms = int(round((time.perf_counter() - start) * 1000))

            # ---- Persist assistant message & cost ----
            with next(get_db()) as db:
                cost = estimate_cost_usd(
                    model=model,
                    model_tokens_input=prompt_tokens_est,
                    model_tokens_output=completion_tokens_est,
                    embedding_model=None,
                    embedding_tokens_input=0,
                )
                assistant_msg = add_message(
                    db,
                    conv.id,
                    role="assistant",
                    content=full_text,
                    citations=citations_payload,
                    used_chunk_ids=used_chunk_ids,
                    model=model,
                    tokens_input=prompt_tokens_est,
                    tokens_output=completion_tokens_est,
                    cost_usd=float(cost),
                    meta={"duration_ms": duration_ms, "k": k, "context_budget": context_budget},
                )

            # ---- Final "done" ----
            await websocket.send_json({
                "type": "done",
                "conversation_id": str(conv.id),
                "answer": full_text,
                "citations": citations_payload,
                "used_chunks": used_chunk_ids,
                "message_id": str(assistant_msg.id),
                "duration_ms": duration_ms,
                "tokens_input": prompt_tokens_est,
                "tokens_output": completion_tokens_est,
                "cost_usd": float(cost),
            })

    except WebSocketDisconnect:
        return
@app.get("/conversations")
def list_conversations_api(
    cursor: Optional[datetime] = Query(None, description="上一页最后一个会话的 updated_at"),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    返回会话列表（不包含消息），按更新时间倒序。
    响应:
    {
      "page_size": 20,
      "items": [
        {"id":"...","title":"New Chat","created_at":"...","updated_at":"..."},
        ...
      ],
      "next_cursor":"2025-10-07T01:23:45.678901",
      "has_more": true
    }
    """
    return list_conversations_cursor(db, cursor=cursor, page_size=page_size)

@app.delete("/conversations/{conversation_id}")
def delete_conversation_api(conversation_id: str, db: Session = Depends(get_db)):
    """
    删除一个会话及其全部聊天记录
    """
    # 校验 UUID
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid conversation_id (must be UUID)")

    result = delete_conversation_and_history(db, conv_uuid)
    return result

@app.get("/conversations/{conversation_id}/history")
def get_conversation_history_api(
    conversation_id: str,
    limit: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """
    Return chat history for a conversation in the same structure you use on the websocket:
      - user messages:  {type:"user", content:"...", message_id:"...", created_at: "..."}
      - assistant msgs: {type:"done", answer:"...", citations:[...], used_chunks:[...],
                         message_id:"...", duration_ms:..., tokens_input:..., tokens_output:..., cost_usd:...,
                         created_at:"..."}
    Ordered oldest -> newest.
    """
    # Validate conversation_id
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid conversation_id (must be UUID)")

    # Ensure conversation exists
    conv = get_conversation(db, conv_uuid)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Load messages (oldest -> newest)
    msgs = list_messages(db, conv_uuid, limit=limit)

    # Map DB rows -> websocket-like payloads
    items: List[Dict[str, Any]] = []
    for m in msgs:
        base = {
            "message_id": str(m.id),
            "created_at": m.created_at,
        }
        if m.role == "user":
            items.append({
                **base,
                "type": "user",
                "content": m.content or "",
            })
        elif m.role == "assistant":
            items.append({
                **base,
                "type": "done",
                "answer": m.content or "",
                "citations": m.citations or [],
                "used_chunks": m.used_chunk_ids or [],
                "duration_ms": (m.meta or {}).get("duration_ms"),
                "tokens_input": m.tokens_input or 0,
                "tokens_output": m.tokens_output or 0,
                "cost_usd": float(m.cost_usd) if m.cost_usd is not None else 0.0,
            })
        else:
            # If you ever store "system" messages, you can choose how to expose them
            items.append({
                **base,
                "type": "system",
                "content": m.content or "",
            })

    return {
        "conversation_id": str(conv.id),
        "count": len(items),
        "items": items,
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

