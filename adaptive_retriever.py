"""
مرحلة 2: الاسترجاع التكيّفي (Adaptive Retrieval).

يترجم قرار مرحلة 1 (query_classifier) لاستدعاء Azure AI Search فعلي،
بأربع أوضاع: vector فقط، keyword فقط، hybrid، metadata-filtered.

ملاحظة مهمة: هاد ملف مستقل تماماً عن search_service.py (يلي بيستخدمه
الـagent من الأسبوع الماضي عبر tools.py). ما بنلمسه، عشان نظام الـAgentic
RAG يضل شغال بدون أي تأثير، والتقييم الجديد يكون مستقل 100% زي ما
طلبت المهمة صراحة ("should not evaluate Agentic tool selection").
"""

import os
from embedding_client import get_embedding
from reranker import rerank_and_filter

# نجيب عدد مرشحين أكبر من top_k الفعلي المطلوب (قبل التصفية)، عشان
# يكون عند مرحلة إعادة الترتيب شي فعلي تختار منه - لو جبنا top_k بالظبط
# من الأساس، ما رح يكون فيه فايدة حقيقية من خطوة التصفية
CANDIDATE_MULTIPLIER = 3

# أسماء المستندات الفعلية بالـindex - لازم تطابق أسماء الملفات المرفوعة
# بـbuild_index.py بالضبط. بما إنها 5 مستندات ثابتة ومعروفة مسبقاً،
# منحل اسمها بمقارنة كلمات بسيطة بدل صيغة بحث جزئي أعقد بـAzure
KNOWN_DOCUMENTS = [
    "1_سياسة_الإجازات_والدوام_الموسعة.docx",
    "2_سلم_الرواتب_والمكافآت_الموسع.docx",
    "3_التأمين_الطبي_والخدمات_الموسع.docx",
    "4_الأمن_السيبراني_واستخدام_الأجهزة_الموسع.docx",
    "5_المشاريع_والتطوير_التقني_الموسع.docx",
]


def resolve_document_name(free_text: str) -> str | None:
    """
    يحول نص حر (زي "سياسة الأمن السيبراني" يلي يرجعه المصنّف) لاسم الملف
    الفعلي بالـindex، عبر عدّ الكلمات المشتركة. يرجع None لو ما لقى
    تطابق معقول - أفضل نعترف بالفشل من نطبق فلتر غلط يرجع صفر نتائج بصمت.
    """
    if not free_text:
        return None

    free_text_words = set(free_text.replace("_", " ").split())

    best_match = None
    best_score = 0
    for doc_name in KNOWN_DOCUMENTS:
        doc_words = set(doc_name.replace(".docx", "").replace("_", " ").split())
        overlap = len(free_text_words & doc_words)
        if overlap > best_score:
            best_score = overlap
            best_match = doc_name

    return best_match if best_score > 0 else None


def _get_search_client():
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    return SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]),
    )


def _format_results(results) -> list[dict]:
    return [
        {
            "chunk_id": r["chunk_id"],
            "content": r["content"],
            "source_document": r["source_document"],
            "section": r["section"],
        }
        for r in results
    ]


def retrieve_hybrid(query: str, top_k: int = 3) -> list[dict]:
    from azure.search.documents.models import VectorizedQuery

    client = _get_search_client()
    query_vector = get_embedding(query)
    vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=top_k, fields="content_vector")

    use_semantic = os.environ.get("AZURE_SEARCH_USE_SEMANTIC", "true").lower() == "true"
    kwargs = {"search_text": query, "vector_queries": [vector_query], "top": top_k}
    if use_semantic:
        kwargs["query_type"] = "semantic"
        kwargs["semantic_configuration_name"] = "default-semantic-config"

    return _format_results(client.search(**kwargs))


def retrieve_vector_only(query: str, top_k: int = 3) -> list[dict]:
    from azure.search.documents.models import VectorizedQuery

    client = _get_search_client()
    query_vector = get_embedding(query)
    vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=top_k, fields="content_vector")

    # search_text=None: بدون شق keyword نهائياً، بس مقارنة المتجهات
    return _format_results(client.search(search_text=None, vector_queries=[vector_query], top=top_k))


def retrieve_keyword_only(query: str, top_k: int = 3) -> list[dict]:
    client = _get_search_client()
    # بدون vector_queries نهائياً - بس مطابقة BM25 النصية التقليدية
    return _format_results(client.search(search_text=query, top=top_k))


def retrieve_metadata_filtered(query: str, document_filter: str, top_k: int = 3) -> list[dict]:
    from azure.search.documents.models import VectorizedQuery

    resolved_name = resolve_document_name(document_filter)
    if resolved_name is None:
        # ما قدرنا نحل اسم المستند - أفضل نرجع لhybrid عادي بدل ما نطبق
        # فلتر غلط يرجع صفر نتائج بصمت (نفس فلسفة fallback بمرحلة 1)
        return retrieve_hybrid(query, top_k)

    client = _get_search_client()
    query_vector = get_embedding(query)
    vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=top_k, fields="content_vector")

    results = client.search(
        search_text=query,
        vector_queries=[vector_query],
        filter=f"source_document eq '{resolved_name}'",
        top=top_k,
    )
    return _format_results(results)


def adaptive_retrieve(original_query: str, classification: dict, top_k: int = 3) -> list[dict]:
    """
    نقطة الدخول الرئيسية لهاي المرحلة. تاخد السؤال الأصلي + نتيجة
    مرحلة 1 كاملة، وترجع القطع المسترجعة - مع تكرار الاسترجاع لكل
    sub_query لو موجودة (comparison / multi_source).
    """
    strategy = classification["retrieval_strategy"]
    sub_queries = classification.get("sub_queries")
    metadata_filter = classification.get("metadata_filter")

    strategy_fn = {
        "hybrid": retrieve_hybrid,
        "vector": retrieve_vector_only,
        "keyword": retrieve_keyword_only,
    }

    # لو عنا أسئلة فرعية، نكرر الاسترجاع لكل وحدة لحالها ونجمع النتائج
    queries_to_run = sub_queries if sub_queries else [original_query]
    candidate_k = top_k * CANDIDATE_MULTIPLIER

    all_results = []
    for sub_query in queries_to_run:
        if strategy == "metadata_filtered":
            candidates = retrieve_metadata_filtered(sub_query, metadata_filter, candidate_k)
        else:
            # .get() مع hybrid كافتراضي آمن لو رجعت استراتيجية غير متوقعة
            fn = strategy_fn.get(strategy, retrieve_hybrid)
            candidates = fn(sub_query, candidate_k)

        # مرحلة 5: إعادة الترتيب والتصفية - نطبقها على كل استراتيجية
        # بدون استثناء (الـsemantic ranker يرتب لكن ما يصفي، فمحتاجين
        # طبقة تصفية صريحة فوقه دايماً)
        filtered = rerank_and_filter(sub_query, candidates, top_k=top_k)

        for r in filtered:
            # نوسم كل نتيجة بالسؤال الفرعي يلي جابها - مهم لاحقاً لتوثيق
            # "Retrieved documents/chunks" بجدول حالات الاختبار
            r["matched_sub_query"] = sub_query

        all_results.extend(filtered)

    return all_results
