"""
ينسّق المراحل 1-4 سوا:
تصنيف → استرجاع تكيّفي → فحص كفاية → (لو ناقص) إعادة صياغة → استرجاع تاني.

حد أقصى 3 محاولات كلية:
- محاولة 1: التصنيف الأصلي زي ما هو
- محاولة 2: نفس الاستراتيجية، بس بصياغة سؤال جديدة
- محاولة 3: صياغة جديدة + تصعيد الاستراتيجية كمان
"""

from query_classifier import classify_query
from adaptive_retriever import adaptive_retrieve
from query_rewriter import rewrite_query
from quality_checker import check_sufficiency
from multi_hop_retriever import multi_hop_retrieve

MAX_ATTEMPTS = 3

# خريطة تصعيد الاستراتيجية للمحاولة الثالثة - لو الأولى فشلت، نجرب بديل
# منطقي: hybrid وvector بيتبادلوا، وkeyword/metadata_filtered بيرجعوا
# لhybrid (الأعم والأشمل) كخط دفاع أخير
STRATEGY_ESCALATION = {
    "hybrid": "vector",
    "vector": "hybrid",
    "keyword": "hybrid",
    "metadata_filtered": "hybrid",
}


def retrieve_with_quality_loop(original_query: str, top_k: int = 3) -> dict:
    """
    نقطة الدخول الرئيسية. بترجع dict فيه:
    - chunks: آخر مجموعة قطع (الأفضل يلي توفرت)
    - sufficient: هل السياق النهائي كافٍ فعلاً
    - attempts: سجل كامل لكل محاولة (مفيد لجدول حالات الاختبار الرسمي)
    - classification: نتيجة التصنيف الأصلية (قبل أي تعديل)
    """
    original_classification = classify_query(original_query)
    current_classification = dict(original_classification)
    current_query = original_query

    attempt_log = []
    chunks = []

    for attempt_num in range(1, MAX_ATTEMPTS + 1):
        # مرحلة 6: multi_step مخطط له مسبقاً من التصنيف، فبس بالمحاولة
        # الأولى منستخدم حلقة الـhops الحقيقية بدل الاسترجاع العادي.
        # لو فشلت (ناقصة)، محاولات إعادة الصياغة 2 و3 بترجع للمسار
        # العادي (adaptive_retrieve) - نفس فلسفة "أبسط تدخل أول" يلي
        # اتفقنا عليها بمرحلة 3
        extra_log = {}
        if attempt_num == 1 and original_classification["query_type"] == "multi_step":
            hop_result = multi_hop_retrieve(current_query, top_k)
            chunks = hop_result["chunks"]
            extra_log["hops"] = hop_result["hops"]
        else:
            chunks = adaptive_retrieve(current_query, current_classification, top_k)

        quality = check_sufficiency(current_query, chunks, query_type=original_classification["query_type"])

        attempt_entry = {
            "attempt": attempt_num,
            "query_used": current_query,
            "strategy_used": current_classification["retrieval_strategy"],
            "chunks_count": len(chunks),
            "sufficient": quality["sufficient"],
            "reasoning": quality["reasoning"],
        }
        attempt_entry.update(extra_log)
        attempt_log.append(attempt_entry)

        if quality["sufficient"]:
            return {
                "chunks": chunks,
                "sufficient": True,
                "attempts": attempt_log,
                "classification": original_classification,
            }

        if attempt_num < MAX_ATTEMPTS:
            # نجهز المحاولة الجاية: صياغة جديدة دايماً، وسؤال مُبسّط
            # (نلغي sub_queries القديمة عشان الصياغة الجديدة تُستخدم فعلياً،
            # مش تُتجاهل لصالح تقسيم قديم)
            current_query = rewrite_query(current_query, chunks)
            current_classification = dict(current_classification)
            current_classification["sub_queries"] = None

            if attempt_num == 2:
                # المحاولة الثالثة بس: نصعّد الاستراتيجية كمان، مو الصياغة بس
                old_strategy = current_classification["retrieval_strategy"]
                current_classification["retrieval_strategy"] = STRATEGY_ESCALATION.get(old_strategy, "hybrid")

    # استنفدنا كل المحاولات المسموحة وما لقينا سياق كافٍ
    return {
        "chunks": chunks,
        "sufficient": False,
        "attempts": attempt_log,
        "classification": original_classification,
    }
