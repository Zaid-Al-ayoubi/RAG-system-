"""
نقطة الدخول الموحّدة للـAdaptive RAG (مراحل 1-4 + 7).

⚠️ نسخة أولية للاختبار end-to-end - مراحل 5 (إعادة الترتيب الصريح
لغير hybrid) و6 (multi-hop تسلسلي حقيقي) لسا ما انبنوا كوحدات منفصلة.
"""

from retrieval_pipeline import retrieve_with_quality_loop
from answer_generator import generate_grounded_answer


def answer_query(query: str, top_k: int = 3, history: list[dict] | None = None) -> dict:
    pipeline_result = retrieve_with_quality_loop(query, top_k)
    answer_result = generate_grounded_answer(query, pipeline_result, history=history)

    return {
        "query": query,
        "answer": answer_result["answer"],
        "sources": answer_result["sources"],
        "grounded": answer_result["grounded"],
        "classification": pipeline_result["classification"],
        "attempts": pipeline_result["attempts"],
    }