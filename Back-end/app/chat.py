# qa.py
from typing import List, Dict, Any, Tuple, Generator
import json, re
from dotenv import load_dotenv
from dataclasses import dataclass
from openai import OpenAI

load_dotenv()  # 加载 .env 文件

client = OpenAI()

# ------- 工具：token 估算（优先 tiktoken，退化到字符近似） -------
def estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        # 退化：英文约 4 chars ~= 1 token；中文可按 2 chars ~= 1 token 粗估
        # 这里统一用保守估算：1 token ≈ 3.8 chars
        return max(1, int(len(text or "") / 3.8))

# ------- 从 hits 里挑选不超预算的片段 -------
def select_chunks_by_budget(
    hits: List[Dict[str, Any]],
    context_token_budget: int = 3000
) -> List[Dict[str, Any]]:
    """
    hits: 已融合排序好的结果（高分在前），元素包含：
          text, tokens(可空), title, page_number, chunk_id, document_id, chunk_index
    逻辑：从头往后累加，直到超过预算为止
    """
    picked = []
    used = 0
    for h in hits:
        tok = int(h.get("tokens") or 0)
        if tok <= 0:
            # 兜底估算：片段文本 + 1 行引用头
            header = f"[DOC: {h.get('title') or ''} | page:{h.get('page_number')} | chunk_id:{h.get('chunk_id')}]"
            tok = estimate_tokens(header) + estimate_tokens(h.get("text") or "")
        if used + tok > context_token_budget:
            break
        used += tok
        picked.append(h)
    return picked

PLAIN_TEXT_SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. "
    "Answer the user's question strictly based on the provided context. "
    "Output plain text only — no JSON, no Markdown code blocks, no backticks, and no formatting tags. "
    "If the information is insufficient, reply: 'Not enough information to answer.'"
)

def build_prompt(user_query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Assemble a readable prompt without JSON output instructions.
    """
    parts = []
    for h in chunks:
        header = f"[DOC: {h.get('title', '')} | page: {h.get('page_number')} | chunk_id: {h.get('chunk_id')}]"
        parts.append(header + "\n" + (h.get("text") or ""))

    context = "\n\n".join(parts)
    return f"Context:\n{context}\n\nUser question: {user_query}\n\nPlease provide a direct plain-text answer."


# ------- 调用 LLM 并健壮地解析 JSON -------
def call_llm_and_parse(user_query: str, prompt: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": PLAIN_TEXT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=False,  # 非流式才会带 usage
    )

    raw = resp.choices[0].message.content or ""
    usage = {
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) if hasattr(resp, "usage") else 0,
        "completion_tokens": getattr(resp.usage, "completion_tokens", 0) if hasattr(resp, "usage") else 0,
        "total_tokens": getattr(resp.usage, "total_tokens", 0) if hasattr(resp, "usage") else 0,
        # 有的 SDK 直接 resp.usage 是 dict，也兼容一下：
    }
    try:
        # 兼容 dict 形态
        if isinstance(resp.usage, dict):
            usage["prompt_tokens"] = resp.usage.get("prompt_tokens", 0)
            usage["completion_tokens"] = resp.usage.get("completion_tokens", 0)
            usage["total_tokens"] = resp.usage.get("total_tokens", 0)
    except Exception:
        pass

    # 只取最外层 JSON
    candidate = raw.strip()
    if not candidate.startswith("{"):
        m = re.search(r"\{.*\}\s*$", raw, flags=re.S)
        if m:
            candidate = m.group(0)

    try:
        parsed = json.loads(candidate)
    except Exception:
        parsed = {"answer": raw, "citations": [], "confidence": 0.0}

    return {"raw": raw, "parsed": parsed, "usage": usage}

# def stream_rag_answer(
#     question: str,
#     hits: List[Dict[str, Any]],
#     model: str = "gpt-4o-mini",
# ) -> Generator[Dict[str, Any], None, None]:
#     """
#     生成标准化的流式分片：
#       - 多次 yield: {"type":"delta","text":"..."}
#       - 最后一次 yield: {"type":"done","full":"...","prompt_tokens":int,"completion_tokens":int}
#     """
#     prompt = build_prompt(question, hits)
#     prompt_tokens_est = estimate_tokens(prompt)
#
#     parts: List[str] = []
#
#     try:
#         stream = client.chat.completions.create(
#             model=model,
#             temperature=0.2,
#             messages=[
#                 {"role": "system", "content": PLAIN_TEXT_SYSTEM_PROMPT},
#                 {"role": "user", "content": prompt},
#             ],
#             stream=True,
#         )
#
#         for chunk in stream:
#             # OpenAI v1：chunk.choices[0].delta 是 ChoiceDelta
#             delta = chunk.choices[0].delta
#             text = getattr(delta, "content", None)
#             if text:
#                 parts.append(text)
#                 # 标准化事件：仅传 dict
#                 yield {"type": "delta", "text": text}
#
#     except Exception as e:
#         # 出错时用 error 事件告知上层
#         yield {"type": "error", "message": str(e)}
#         return  # 结束生成器
#
#     # 结束：计算完整文本与 tokens，并一次性给出
#     full = "".join(parts)
#     completion_tokens_est = estimate_tokens(full)
#     yield {
#         "type": "done",
#         "full": full,
#         "prompt_tokens": prompt_tokens_est,
#         "completion_tokens": completion_tokens_est,
#     }
#

# Regular expressions to remove code fences and stray backticks
FENCE_RE = re.compile(r"^```(\w+)?\s*$")
INLINE_TICKS_RE = re.compile(r"`{1,3}")

def _sanitize_delta(text: str, fence_state: Dict[str, bool]) -> str:
    """
    Remove markdown code fences like ```json and stray backticks.
    fence_state tracks whether we are inside a fenced block.
    """
    if not text:
        return ""

    out_lines = []
    for line in text.splitlines(keepends=True):
        # Detect start or end of a code fence
        if FENCE_RE.match(line.strip()):
            fence_state["in_fence"] = not fence_state.get("in_fence", False)
            continue
        # Skip content inside fences
        if fence_state.get("in_fence", False):
            continue
        # Remove inline backticks
        out_lines.append(INLINE_TICKS_RE.sub("", line))
    return "".join(out_lines)


def stream_rag_answer(
    question: str,
    hits: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
) -> Generator[Dict[str, Any], None, None]:

    prompt = build_prompt(question, hits)
    prompt_tokens_est = estimate_tokens(prompt)
    parts: List[str] = []
    fence_state = {"in_fence": False}

    try:
        stream = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": PLAIN_TEXT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            raw = getattr(delta, "content", None)
            if not raw:
                continue

            clean = _sanitize_delta(raw, fence_state)
            if not clean:
                continue

            parts.append(clean)
            yield {"type": "delta", "text": clean}

    except Exception as e:
        yield {"type": "error", "message": str(e)}
        return

    # Just finish without sending text again
    full = "".join(parts).strip()
    completion_tokens_est = estimate_tokens(full)
    # Yield only metadata for server to handle
    yield {"type": "done_meta", "prompt_tokens": prompt_tokens_est, "completion_tokens": completion_tokens_est}
