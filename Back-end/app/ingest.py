# ingest.py
from typing import BinaryIO
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import tiktoken


def parse_pdf(file: BinaryIO) -> str:
    # reader = PdfReader(file_path)
    # text = []
    # for page in reader.pages:
    #     text.append(page.extract_text() or "")
    # return "\n".join(text)
    reader = PdfReader(file)  # 直接传文件对象
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def parse_url(url: str) -> str:
    resp = requests.get(url, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.get_text(separator="\n")

def chunk_text(text: str, max_tokens=300, overlap=50, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = tokens[start:end]
        chunks.append(encoding.decode(chunk))
        start += max_tokens - overlap
    return chunks

