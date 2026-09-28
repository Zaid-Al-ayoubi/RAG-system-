"""
طبقة الاسترجاع (retrieval). فيها مسارين:
- MOCK: بحث بدائي محلي على chunks.json، يشتغل فوراً بدون Azure
- REAL: hybrid + semantic search فعلي على Azure AI Search

باقي التطبيق (Flask) بينادي search_documents() بس، وما يعرف أصلاً
إذا كان الجواب جاي من mock أو real.
"""

import os
import json
from azure.core.exceptions import HttpResponseError

USE_MOCK = os.environ.get("USE_MOCK_SERVICES", "true").lower() == "true"

_MOCK_CHUNKS = None  # نخزنهم بالذاكرة بعد أول تحميل


def _load_mock_chunks() -> list[dict]:
    global _MOCK_CHUNKS
    if _MOCK_CHUNKS is None:
        with open("chunks.json", "r", encoding="utf-8") as f:
            _MOCK_CHUNKS = json.load(f)
    return _MOCK_CHUNKS


def _mock_search(query: str, top_k: int) -> list[dict]:
    # بحث بدائي: بيعد كم كلمة من السؤال ظهرت حرفياً بكل chunk
    chunks = _load_mock_chunks()
    query_words = [w for w in query.split() if len(w) > 1]

    scored = []
    for chunk in chunks:
        score = sum(1 for w in query_words if w in chunk["content"])
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def _real_search(query: str, top_k: int) -> list[dict]:
    # الاستيراد جوا الدالة عشان mock mode يشتغل بدون مكتبات Azure
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.models import VectorizedQuery
    from embedding_client import get_embedding

    client = SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]),
    )

    # نفس دالة الـ embedding المستخدمة وقت الفهرسة بالضبط
    vector_query = VectorizedQuery(
        vector=get_embedding(query),
        k_nearest_neighbors=top_k,
        fields="content_vector",
    )

    def _run(semantic: bool) -> list[dict]:
        kwargs = {
            "search_text": query,          # الشق الـ keyword
            "vector_queries": [vector_query],  # الشق الـ vector
            "top": top_k,
        }
        if semantic:
            kwargs["query_type"] = "semantic"
            kwargs["semantic_configuration_name"] = "default-semantic-config"

        # list() هون ضروري: الخطأ بيطلع أثناء التكرار (lazy)
        return [
            {
                "chunk_id": r["chunk_id"],
                "content": r["content"],
                "source_document": r["source_document"],
                "section": r["section"],
            }
            for r in client.search(**kwargs)
        ]

    try:
        return _run(semantic=True)
    except HttpResponseError as e:
        if "semantic" in str(e).lower():
            # الـ semantic config مش متاح على الـ index → نكمل hybrid عادي
            return _run(semantic=False)
        raise


def search_documents(query: str, top_k: int = 3) -> list[dict]:
    if USE_MOCK:
        return _mock_search(query, top_k)
    return _real_search(query, top_k)