"""
يقيّم نتائج test_adaptive_rag.py عبر 9 مقاييس RAG رسمية.
(نسخة معدّلة - انظر التعليقات المعلّمة بـ [تعديل] لشرح كل تغيير)
"""

import json
import re
from pathlib import Path


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?%?", text or ""))


def _extract_keywords(text: str, min_len: int = 4) -> set[str]:
    stopwords = {
        "من", "في", "على", "عن", "إلى", "هذا", "هذه", "التي", "الذي",
        "ما", "لا", "هو", "هي", "مع", "أو", "و", "ثم", "كل", "بعض",
        "عند", "بعد", "قبل", "بين", "حسب", "عبر", "خلال", "دون",
    }
    words = re.findall(r"[\u0600-\u06FF]{3,}", text or "")
    return {w for w in words if w not in stopwords and len(w) >= min_len}


def _answer_correctness(expected_info: str, answer: str) -> dict:
    if not expected_info or not answer:
        return {"score": 0.0, "numbers_matched": 0, "numbers_total": 0, "note": "معلومات ناقصة"}

    expected_numbers = _extract_numbers(expected_info)
    answer_numbers = _extract_numbers(answer)
    numbers_matched = len(expected_numbers & answer_numbers)
    numbers_total = len(expected_numbers)
    numbers_score = numbers_matched / numbers_total if numbers_total > 0 else 1.0

    expected_kw = _extract_keywords(expected_info)
    answer_kw = _extract_keywords(answer)
    kw_matched = len(expected_kw & answer_kw)
    kw_total = len(expected_kw)
    kw_score = kw_matched / kw_total if kw_total > 0 else 1.0

    final_score = 0.7 * numbers_score + 0.3 * kw_score

    return {
        "score": round(final_score, 3),
        "numbers_matched": numbers_matched,
        "numbers_total": numbers_total,
        "keywords_matched": kw_matched,
        "keywords_total": kw_total,
    }


def compute_metrics(results: list[dict]) -> dict:
    cases_with_expected_docs = [r for r in results if r.get("expected_documents")]
    cases_without_expected_docs = [r for r in results if not r.get("expected_documents")]

    # --- 1. Retrieval Recall ---
    recall_scores = []
    for r in cases_with_expected_docs:
        expected = set(r["expected_documents"])
        actual = set(r.get("actual_documents", []))
        if expected:
            recall_scores.append(len(expected & actual) / len(expected))
    retrieval_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    # --- 2. Precision@K ---
    precision_scores = []
    for r in cases_with_expected_docs:
        expected = set(r["expected_documents"])
        sources = r.get("sources", [])
        if sources:
            relevant = sum(1 for s in sources if s["source_document"] in expected)
            precision_scores.append(relevant / len(sources))
    precision_at_k = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0

    # --- 3. Context Relevance ---
    # [تعديل] بنفس منطق Recall/Precision بالضبط: نقيس بس على الحالات يلي
    # فعلياً عندها مستندات متوقعة (بمعنى فعلياً لازم يلاقي شي). حالة 7
    # (info_not_in_kb) مستثناة هون تماماً، مش بس من العد الإيجابي - لأنه
    # "لم يجد شيئاً" هو سلوكها الصحيح، مش نتيجة نقيّمها بمقياس "هل السياق
    # كان ذا صلة" أصلاً (ما كان متوقع يكون فيه سياق ذو صلة من الأساس)
    context_relevant_count = sum(
        1 for r in cases_with_expected_docs
        if r.get("attempts") and r["attempts"][-1].get("sufficient", False)
    )
    context_relevance = (
        context_relevant_count / len(cases_with_expected_docs) if cases_with_expected_docs else 0.0
    )

    # --- 4. Answer Correctness ---
    correctness_scores = []
    correctness_details = []
    for r in results:
        if not r.get("expected_documents"):
            score = 1.0 if not r.get("grounded", False) else 0.0
            correctness_details.append({"case_id": r["case_id"], "score": score, "note": "Info NOT in KB"})
        else:
            calc = _answer_correctness(r.get("expected_info", ""), r.get("answer", ""))
            score = calc["score"]
            correctness_details.append({"case_id": r["case_id"], **calc})
        correctness_scores.append(score)
    answer_correctness = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0

    # --- 5. Groundedness ---
    grounded_count = sum(1 for r in results if r.get("grounded", False))
    groundable_cases = [r for r in results if r.get("expected_documents")]
    grounded_expected_count = sum(1 for r in groundable_cases if r.get("grounded", False))
    groundedness = grounded_expected_count / len(groundable_cases) if groundable_cases else 0.0

    # --- 6. Retrieval Failure Rate ---
    failures = sum(1 for r in results if r.get("expect_sufficient") and not r.get("grounded", False))
    retrieval_failure_rate = failures / len(results) if results else 0.0

    # --- 7. Query-Rewriting Success Rate ---
    # [تعديل] "نجاح" إعادة الصياغة يعني "وصلنا للنتيجة الصحيحة المتوقعة"،
    # مش حرفياً "grounded=True" دايماً. لحالة 7، النتيجة الصحيحة هي
    # grounded=False (اعتراف صادق) - فهاي لازم تُحسب نجاح، مش فشل.
    rewrite_cases = [r for r in results if r.get("rewrite_triggered", False)]
    rewrite_success = sum(
        1 for r in rewrite_cases
        if r.get("grounded", False) == r.get("expect_sufficient", True)
    )
    query_rewriting_success_rate = (
        rewrite_success / len(rewrite_cases) if rewrite_cases else None
    )

    # --- 8. Avg Retrieval Attempts ---
    attempts_list = [r.get("num_attempts", 1) for r in results]
    avg_attempts = sum(attempts_list) / len(attempts_list) if attempts_list else 0.0

    # --- 9. Avg Latency ---
    latencies = [r.get("latency_seconds", 0) for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "retrieval_recall": round(retrieval_recall, 3),
        "precision_at_k": round(precision_at_k, 3),
        "context_relevance": round(context_relevance, 3),
        "answer_correctness": round(answer_correctness, 3),
        "groundedness": round(groundedness, 3),
        "retrieval_failure_rate": round(retrieval_failure_rate, 3),
        "query_rewriting_success_rate": (
            round(query_rewriting_success_rate, 3) if query_rewriting_success_rate is not None else None
        ),
        "avg_retrieval_attempts": round(avg_attempts, 2),
        "avg_latency_seconds": round(avg_latency, 2),
        "_meta": {
            "total_cases": len(results),
            "cases_with_expected_docs": len(cases_with_expected_docs),
            "cases_without_expected_docs": len(cases_without_expected_docs),
            "rewrite_triggered_cases": len(rewrite_cases),
            "failures": failures,
        },
        "_answer_correctness_details": correctness_details,
    }


def print_report(metrics: dict):
    print("\n" + "=" * 70)
    print("📊 تقرير تقييم Adaptive RAG")
    print("=" * 70)

    print(f"\n🎯 الحالات:")
    print(f"   إجمالي: {metrics['_meta']['total_cases']}")
    print(f"   بمستندات متوقعة: {metrics['_meta']['cases_with_expected_docs']}")
    print(f"   بدون مستندات (Info NOT in KB): {metrics['_meta']['cases_without_expected_docs']}")
    print(f"   حالات rewrite trigger: {metrics['_meta']['rewrite_triggered_cases']}")

    print(f"\n📈 مقاييس الاسترجاع:")
    print(f"   Retrieval Recall:          {metrics['retrieval_recall']:.1%}")
    print(f"   Precision@K:               {metrics['precision_at_k']:.1%}")
    print(f"   Context Relevance:         {metrics['context_relevance']:.1%}")

    print(f"\n📈 مقاييس التوليد:")
    print(f"   Answer Correctness:        {metrics['answer_correctness']:.1%}")
    print(f"   Groundedness:              {metrics['groundedness']:.1%}")

    print(f"\n📈 مقاييس الأداء:")
    print(f"   Retrieval Failure Rate:    {metrics['retrieval_failure_rate']:.1%}")
    qrs = metrics["query_rewriting_success_rate"]
    print(f"   Query-Rewriting Success:   {qrs:.1%}" if qrs is not None else "   Query-Rewriting Success:   N/A")
    print(f"   Avg Retrieval Attempts:    {metrics['avg_retrieval_attempts']}")
    print(f"   Avg Latency:               {metrics['avg_latency_seconds']}s")

    print(f"\n📋 تفاصيل Answer Correctness:")
    print(f"   {'حالة':<6} {'درجة':<8} {'تفاصيل'}")
    print(f"   {'-' * 60}")
    for d in metrics["_answer_correctness_details"]:
        note = d.get("note", "")
        if not note:
            note = f"أرقام: {d.get('numbers_matched')}/{d.get('numbers_total')} | كلمات: {d.get('keywords_matched')}/{d.get('keywords_total')}"
        print(f"   {d['case_id']:<6} {d['score']:<8.2f} {note}")


if __name__ == "__main__":
    input_path = Path("adaptive_test_results.json")
    if not input_path.exists():
        print(f"❌ ملف {input_path} غير موجود. شغّل test_adaptive_rag.py أولاً.")
        raise SystemExit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    metrics = compute_metrics(results)
    print_report(metrics)

    output_path = "metrics_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n✅ تم حفظ التقرير بـ {output_path}")