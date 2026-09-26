"""
طبقة الاسترجاع (retrieval). فيها مسارين:
- MOCK: بحث بدائي محلي على chunks.json، يشتغل فوراً بدون Azure
- REAL: hybrid + semantic search فعلي على Azure AI Search

باقي التطبيق (Flask) بينادي search_documents() بس، وما يعرف أصلاً
إذا كان الجواب جاي من mock أو real. هيك لما تنحل مشكلة الفوترة،
منبدل قيمة USE_MOCK_SERVICES بـ.env وخلص، بدون ما نلمس Flask ولا الواجهة.
"""

import os
import json

USE_MOCK = os.environ.get("USE_MOCK_SERVICES", "true").lower() == "true"

_MOCK_CHUNKS = None  # نخزنهم بالذاكرة بعد أول تحميل، ما نعيد قراءة الملف كل مرة


def _load_mock_chunks() -> list[dict]:
    global _MOCK_CHUNKS
    if _MOCK_CHUNKS is None:
        with open("chunks.json", "r", encoding="utf-8") as f:
            _MOCK_CHUNKS = json.load(f)
    return _MOCK_CHUNKS


def _mock_search(query: str, top_k: int) -> list[dict]:
    # بحث بدائي: بيعد كم كلمة من السؤال ظهرت حرفياً بكل chunk
    # هاد مش semantic search حقيقي أبداً - غرضه الوحيد نختبر منطق
    # الـFlask والذاكرة والواجهة بشكل واقعي بدون انتظار Azure
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
    # الاستيراد جوا الدالة (مش أعلى الملف) عشان الـmock mode
    # يقدر يشتغل حتى لو مكتبات Azure مش مركبة أو الاتصال مش جاهز
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.models import VectorizedQuery
    from embedding_client import get_embedding

    client = SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]),
    )

    query_vector = get_embedding(query)  # نفس دالة الـembedding المستخدمة وقت الفهرسة بالضبط

    vector_query = VectorizedQuery(
        vector=query_vector, k_nearest_neighbors=top_k, fields="content_vector"
    )

    results = client.search(
        search_text=query,               # الشق الـkeyword من الـhybrid search
        vector_queries=[vector_query],   # الشق الـvector من الـhybrid search
        query_type="semantic",           # يفعّل إعادة الترتيب الدلالي فوق الاثنين
        semantic_configuration_name="default-semantic-config",
        top=top_k,
    )

    return [
        {
            "chunk_id": r["chunk_id"],
            "content": r["content"],
            "source_document": r["source_document"],
            "section": r["section"],
        }
        for r in results
    ]


def search_documents(query: str, top_k: int = 3) -> list[dict]:
    if USE_MOCK:
        return _mock_search(query, top_k)
    return _real_search(query, top_k)