"""
دالة واحدة لتوليد embedding، نستخدمها بمكانين: وقت الفهرسة (indexing)
ووقت البحث (query). لازم تكون نفس الدالة بالضبط بالمكانين،
عشان المتجهات تكون قابلة للمقارنة صح.
"""

import os
from openai import AzureOpenAI

_client = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
    return _client


def get_embedding(text: str) -> list[float]:
    client = _get_client()
    deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    response = client.embeddings.create(input=text, model=deployment)
    return response.data[0].embedding