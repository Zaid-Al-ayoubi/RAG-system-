"""
مرحلة 6: الاسترجاع متعدد الخطوات (Multi-Hop Retrieval) - Type B حقيقي.

بعكس مرحلة 3 (إعادة الصياغة، رد فعل على فشل)، هاي مرحلة **مخطط لها
مسبقاً** لأسئلة type=="multi_step": نعرف من التصنيف إنه محتاجة أكتر
من قفزة استرجاع، ونبني كل قفزة بناءً على نتيجة اللي قبلها مباشرة.
"""

import os
from openai import AzureOpenAI
from adaptive_retriever import adaptive_retrieve

MAX_HOPS = 3

HOP_SYSTEM_PROMPT = """أنت نظام استرجاع متعدد الخطوات (multi-hop retrieval).
عندك سؤال أصلي معقد، ومعلومات مجمّعة من خطوة/خطوات استرجاع سابقة.

مهمتك: حدد هل المعلومات المجمّعة لحد الآن كافية للإجابة الكاملة على
السؤال الأصلي.
- لو كافية: أرجع فقط الكلمة DONE (بدون أي نص إضافي).
- لو ناقصة: استخرج القيمة أو الحقيقة المحددة اللي ظهرت بالمعلومات
  المجمّعة (مثال: اسم منصة، اسم فريق، رقم محدد) واستخدمها بالذات
  لصياغة سؤال بحث جديد ومحدد للخطوة الجاية - لا تصوغ سؤالاً عاماً،
  استخدم القيمة المكتشفة فعلياً بالنص.

أرجع فقط: إما DONE، أو نص سؤال البحث الجديد. بدون أي شرح إضافي."""


def _client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def formulate_next_hop(original_query: str, accumulated_chunks: list[dict], hop_number: int) -> str | None:
    """يرجع سؤال الخطوة الجاية، أو None لو المعلومات المجمّعة كافية فعلاً"""
    client = _client()
    chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

    context_str = "\n\n".join(
        f"[من: {c['source_document']} - {c['section']}]\n{c['content']}" for c in accumulated_chunks
    )

    response = client.chat.completions.create(
        model=chat_deployment,
        messages=[
            {"role": "system", "content": HOP_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"السؤال الأصلي: {original_query}\n\nالمعلومات المجمّعة لحد الآن (بعد {hop_number} خطوة/خطوات):\n{context_str}",
            },
        ],
        temperature=0.1,
    )

    result = (response.choices[0].message.content or "").strip()
    if not result or result.upper().startswith("DONE"):
        return None
    return result


def multi_hop_retrieve(original_query: str, top_k: int = 3, max_hops: int = MAX_HOPS) -> dict:
    """
    ينفذ حلقة hop فعلية: استرجع → شوف النتيجة → صيغ سؤال الخطوة الجاية
    بناءً عليها → استرجع تاني. يتوقف لما الموديل يقرر إنه المعلومات
    كافية (DONE) أو لما يوصل max_hops.
    """
    hop_log = []
    accumulated_chunks = []
    seen_chunk_ids = set()
    current_hop_query = original_query

    # استراتيجية hybrid بسيطة بدون sub_queries لكل قفزة - القفزات هون
    # هي بديل الأسئلة الفرعية، مش بالإضافة لها
    simple_classification = {"retrieval_strategy": "hybrid", "sub_queries": None, "metadata_filter": None}

    for hop_num in range(1, max_hops + 1):
        hop_chunks = adaptive_retrieve(current_hop_query, simple_classification, top_k)

        for c in hop_chunks:
            c["hop_number"] = hop_num
            c["hop_query"] = current_hop_query
            if c["chunk_id"] not in seen_chunk_ids:
                seen_chunk_ids.add(c["chunk_id"])
                accumulated_chunks.append(c)

        hop_log.append({"hop": hop_num, "query": current_hop_query, "chunks_found": len(hop_chunks)})

        if hop_num < max_hops:
            next_query = formulate_next_hop(original_query, accumulated_chunks, hop_num)
            if next_query is None:
                break  # الموديل قرر إنه المعلومات كافية - نوقف الحلقة بدري
            current_hop_query = next_query

    return {"chunks": accumulated_chunks, "hops": hop_log}
