"""
1. يقرأ chunks.json
2. يولّد embedding لكل chunk عبر Azure OpenAI
3. ينشئ index على Azure AI Search (hybrid + semantic search)
4. يرفع كل الـchunks مع متجهاتها
"""

import os
import json
import hashlib

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
)
from embedding_client import get_embedding  # نفس الدالة المستخدمة وقت البحث لاحقاً

load_dotenv()  # يقرأ القيم من ملف .env المجاور للسكريبت

# --- إعدادات Azure AI Search ---
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
INDEX_NAME = os.environ["AZURE_SEARCH_INDEX_NAME"]

# لازم يطابق فعلياً موديل الـembedding يلي عندك:
# text-embedding-3-small / text-embedding-ada-002 -> 1536
# text-embedding-3-large -> 3072
EMBEDDING_DIMENSIONS = 1536


def make_safe_key(chunk_id: str) -> str:
    # Azure AI Search بيسمح بس بأحرف إنجليزية/أرقام/-/_/=
    # الـchunk_id عندنا فيه عربي، فبنحوله لـhash إنجليزي فريد وآمن
    return hashlib.md5(chunk_id.encode("utf-8")).hexdigest()


def create_index_if_not_exists():
    index_client = SearchIndexClient(
        endpoint=SEARCH_ENDPOINT, credential=AzureKeyCredential(SEARCH_KEY)
    )

    # Azure AI Search ما بيسمح بتعديل تعريف حقل موجود مسبقاً (حتى لو التغيير
    # بسيط)، فبنحذف الـindex القديم (لو موجود) ونعيد إنشاءه من الصفر.
    # آمن هون لأنه بياناتنا كلها بـchunks.json ومنرفعها تاني بنفس السكريبت
    try:
        index_client.delete_index(INDEX_NAME)
        print(f"تم حذف الـindex القديم '{INDEX_NAME}' لإعادة إنشائه بالتعريف الجديد.")
    except Exception:
        pass  # الـindex مش موجود أصلاً، عادي نكمل

    fields = [
        # id: المفتاح الفريد الآمن (hash إنجليزي)
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        # chunk_id: نفس المعرف الأصلي بالعربي، لكن كحقل عادي للعرض فقط
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),
        # content: النص الفعلي، هاد اللي منسوي عليه keyword search
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(
            name="source_document",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(name="section", type=SearchFieldDataType.String, filterable=True),
        SimpleField(
            name="sections_covered",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
        ),
        # content_vector: المتجه الرقمي، هاد اللي منسوي عليه vector search
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="default-vector-profile",
        ),
    ]

    # بنعرّف خوارزمية HNSW وبنربطها بـ"بروفايل" اسمه default-vector-profile
    # الحقل content_vector فوق بيشاور على نفس اسم البروفايل هاد
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
        profiles=[
            VectorSearchProfile(
                name="default-vector-profile",
                algorithm_configuration_name="default-hnsw",
            )
        ],
    )

    # semantic configuration: بيحدد للـranker وين يركز أثناء إعادة الترتيب
    semantic_config = SemanticConfiguration(
        name="default-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")],
            keywords_fields=[SemanticField(field_name="section")],
        ),
    )
    semantic_search = SemanticSearch(configurations=[semantic_config])

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    # create_or_update_index: لو الـindex موجود بيحدثه، لو مش موجود بينشئه
    index_client.create_or_update_index(index)
    print(f"الـindex '{INDEX_NAME}' جاهز.")


def upload_chunks(chunks_path: str = "chunks.json"):
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_KEY),
    )

    documents = []
    for chunk in chunks:
        embedding = get_embedding(chunk["content"])  # نداء API لكل chunk لحاله
        documents.append(
            {
                "id": make_safe_key(chunk["chunk_id"]),
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "source_document": chunk["source_document"],
                "section": chunk["section"],
                "sections_covered": chunk["sections_covered"],
                "content_vector": embedding,
            }
        )
        print(f"جهزت embedding لـ {chunk['chunk_id']}")

    # upload_documents بيعمل upsert: لو الـid موجود بيحدثه، لو جديد بيضيفه
    result = search_client.upload_documents(documents=documents)
    success_count = sum(1 for r in result if r.succeeded)
    print(f"\nتم رفع {success_count}/{len(documents)} مستند بنجاح.")

    if success_count < len(documents):
        for r in result:
            if not r.succeeded:
                print(f"فشل: {r.key} -> {r.error_message}")


if __name__ == "__main__":
    create_index_if_not_exists()
    upload_chunks()