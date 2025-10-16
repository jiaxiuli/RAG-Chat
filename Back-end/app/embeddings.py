# embedding.py
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件

client = OpenAI()
def get_embedding(text: str, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding
