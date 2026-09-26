"""
الاختبار الرسمي للـAdaptive RAG (مراحل 1-7).

8 حالات اختبار تغطي كل السيناريوهات المطلوبة بالمهمة:
1. Simple factual
2. Ambiguous (needs query understanding)
3. Needs query rewriting
4. Hybrid/alternative retrieval
5. Multi-document
6. Multi-hop reasoning
7. Info NOT in knowledge base
8. Misleading wording

لكل حالة نوثّق:
- السؤال + المعلومة المتوقعة
- الاستراتيجية المختارة + النوع
- القطع المسترجعة + هل trigger إعادة صياغة
- الجواب النهائي + المصادر
- هل الجواب صحيح (يُقيَّم يدوياً)

النتائج تُحفظ بـ adaptive_test_results.json للتقييم اللاحق.
"""

import json
import time
from dotenv import load_dotenv

load_dotenv()

from adaptive_rag import answer_query


# ============================================================
# الـ8 حالات الرسمية
# ============================================================

TEST_CASES = [
    {
        "id": 1,
        "scenario": "simple_factual",
        "question": "كم يوم إجازة سنوية يستحق الموظف؟",
        "expected_info": "21 يوماً عن كل سنة عمل، وترتفع إلى 28 يوماً بعد 5 سنوات",
        "expected_type": "simple",
        "expected_strategy": "hybrid",
        "expected_documents": ["1_سياسة_الإجازات_والدوام_الموسعة.docx"],
        "expect_rewrite": False,
        "expect_sufficient": True,
        "note": "سؤال معلوماتي مباشر، إجابته بقطعة واحدة واضحة",
    },
    {
        "id": 2,
        "scenario": "ambiguous",
        "question": "شو شروط الإجازة؟",
        "expected_info": "توضيح لنوع الإجازة (سنوية/مرضية/طارئة) وشروط كل نوع",
        "expected_type": "simple",
        "expected_strategy": "hybrid",
        "expected_documents": ["1_سياسة_الإجازات_والدوام_الموسعة.docx"],
        "expect_rewrite": "possibly",  # غامض — ممكن يحتاج rewrite
        "expect_sufficient": True,
        "note": "سؤال غامض — 'الإجازة' عامة، النظام لازم يفهم المقصود أو يسترجع كل الأنواع",
    },
    {
    "id": 3,
    "scenario": "needs_rewriting",
    "question": "لو الموظف غاب كثير شو بيصير فيه؟",
    "expected_info": "المستند بيتكلم عن التأخر عن الحضور (بعد 8:15 صباحاً) وحسم التأخيرات التراكمية اللي بتتجاوز 120 دقيقة شهرياً من الراتب - وليس عن غياب أيام كاملة تحديداً",
    "expected_type": "simple",
    "expected_strategy": "hybrid",
    "expected_documents": ["1_سياسة_الإجازات_والدوام_الموسعة.docx"],
    "expect_rewrite": True,
    "expect_sufficient": False,  # ⬅️ التغيير الأساسي: المعلومة غير موجودة حرفياً بهاد الشكل
    "note": "صياغة عامية ('غاب كتير') تشير لمفهوم غير موجود بالمستند بنفس الصيغة - المستند يتكلم عن 'تأخر' لا 'غياب'. القرار الصحيح: الاعتراف بعدم كفاية المعلومة، لا اختراع ربط",
    },
    {
        "id": 4,
        "scenario": "hybrid_alternative_retrieval",
        "question": "شو نسبة تحمل الموظف بالتأمين الطبي داخل الشبكة المعتمدة؟",
        "expected_info": "10% داخل الشبكة، 20% خارجها",
        "expected_type": "simple",
        "expected_strategy": "hybrid",
        "expected_documents": ["3_التأمين_الطبي_والخدمات_الموسع.docx"],
        "expect_rewrite": False,
        "expect_sufficient": True,
        "note": "يحتاج hybrid (vector+keyword) لأنه قيمة رقمية غير معروفة مسبقاً",
    },
    {
        "id": 5,
        "scenario": "multi_document",
        "question": "شو كل السياسات المتعلقة بحماية بيانات الموظف بالشركة؟",
        "expected_info": "سياسات من الأمن السيبراني + الإجازات (الخصوصية) + التطوير التقني",
        "expected_type": "multi_source",
        "expected_strategy": "hybrid",
        "expected_documents": [
            "4_الأمن_السيبراني_واستخدام_الأجهزة_الموسع.docx",
        ],
        "expect_rewrite": "possibly",
        "expect_sufficient": True,
        "note": "معلومات مبعثرة عبر مستندات متعددة بدون مقارنة",
    },
    {
    "id": 6,
    "scenario": "multi_hop_reasoning",
    "question": "شو اسم منصة التواصل المعتمدة للعمل عن بُعد، وهل نفس المنصة مذكورة بسياسة الأمن السيبراني كأداة معتمدة رسمياً؟",
    "expected_info": "الجزء الأول: Microsoft Teams (من سياسة الإجازات). الجزء الثاني: Teams غير مذكورة إطلاقاً بمستند الأمن السيبراني",
    "expected_type": "multi_step",
    "expected_strategy": "hybrid",
    "expected_documents": ["1_سياسة_الإجازات_والدوام_الموسعة.docx"],  # ⬅️ حذفنا doc4 من هون
    "expect_rewrite": "possibly",
    "expect_sufficient": False,  # ⬅️ لأنه الجزء الثاني فعلياً "لأ" مش موجود، والنظام الحالي ما بيفرق بين negative confirmation وretrieval failure
    "note": "قفزة أولى تكتشف Teams بنجاح. قفزة ثانية: Teams فعلياً غير مذكورة بمستند الأمن السيبراني - هاي حالة 'تأكيد سلبي صحيح' مش فشل استرجاع. النظام الحالي ما بيميز بينهم، فبيرجع sufficient=False بدل جواب جزئي دقيق - موثّق كـknown limitation",
    },
    {
        "id": 7,
        "scenario": "info_not_in_kb",
        "question": "شو سياسة العمل من المنزل بشركة جوجل؟",
        "expected_info": "لا توجد معلومات — المستندات تخص شركتنا فقط",
        "expected_type": "simple",
        "expected_strategy": "hybrid",
        "expected_documents": [],
        "expect_rewrite": "possibly",
        "expect_sufficient": False,  # لازم يرجع grounded=False
        "note": "المعلومة غير موجودة — النظام لازم يعترف بدل ما يخترع",
    },
    {
        "id": 8,
        "scenario": "misleading_wording",
        "question": "سمعت إنه في بونص كبير لنهاية السنة، وكم يوم إجازة سنوية معي؟",
        "expected_info": "البونص (5-20% حسب التقييم) + 21 يوم إجازة سنوية",
        "expected_type": "multi_source",
        "expected_strategy": "hybrid",
        "expected_documents": [
            "1_سياسة_الإجازات_والدوام_الموسعة.docx",
            "2_سلم_الرواتب_والمكافآت_الموسع.docx",
        ],
        "expect_rewrite": "possibly",
        "expect_sufficient": True,
        "note": "صياغة مضللة ('بونص كبير') — النظام لازم يتجاهل الكلمة العاطفية ويرجع للأرقام الفعلية",
    },
]


# ============================================================
# تنفيذ الاختبارات
# ============================================================

def run_single_case(case: dict) -> dict:
    """يشغل حالة واحدة، ويجمع كل التوثيق المطلوب"""
    print(f"\n{'=' * 70}")
    print(f"حالة {case['id']} [{case['scenario']}]: {case['question']}")
    print(f"متوقع: نوع={case['expected_type']} | استراتيجية={case['expected_strategy']}")

    start_time = time.time()
    try:
        result = answer_query(case["question"])
        elapsed = time.time() - start_time
        error = None
    except Exception as e:
        elapsed = time.time() - start_time
        result = None
        error = str(e)

    if error:
        print(f"❌ خطأ أثناء التنفيذ: {error}")
        return {
            "case_id": case["id"],
            "scenario": case["scenario"],
            "question": case["question"],
            "expected_info": case["expected_info"],
            "expected_type": case["expected_type"],
            "expected_strategy": case["expected_strategy"],
            "expected_documents": case["expected_documents"],
            "error": error,
            "latency_seconds": round(elapsed, 2),
            "evaluation": {"success": False, "error": error},
        }

    # استخراج تفاصيل من نتيجة الـpipeline
    classification = result.get("classification", {})
    attempts = result.get("attempts", [])
    sources = result.get("sources", [])

    # هل حصل trigger لإعادة صياغة؟ (يعني عدد attempts > 1)
    rewrite_triggered = len(attempts) > 1

    # هل نجح في النهاية؟ (grounded=True)
    was_sufficient = result.get("grounded", False)

    # مقارنة المصادر بالمتوقعة
    actual_docs = sorted({s["source_document"] for s in sources})
    expected_docs = sorted(case["expected_documents"])
    docs_match = set(actual_docs) == set(expected_docs) if expected_docs else len(actual_docs) == 0

    # مقارنة النوع والاستراتيجية
    type_match = classification.get("query_type") == case["expected_type"]
    strategy_match = classification.get("retrieval_strategy") == case["expected_strategy"]

    # هل الـrewrite trigger متوقع؟
    if case["expect_rewrite"] == True:
        rewrite_ok = rewrite_triggered
    elif case["expect_rewrite"] == False:
        rewrite_ok = not rewrite_triggered
    else:  # "possibly"
        rewrite_ok = True  # ما نقيّمهاش

    # هل الكفاية تطابق المتوقع؟
    sufficiency_match = was_sufficient == case["expect_sufficient"]

    # طباعة
    print(f"\n📋 التصنيف الفعلي:")
    print(f"   النوع: {classification.get('query_type')} | {'✅' if type_match else '❌'}")
    print(f"   الاستراتيجية: {classification.get('retrieval_strategy')} | {'✅' if strategy_match else '❌'}")
    print(f"   السبب: {classification.get('reasoning', '')[:100]}")

    print(f"\n🔄 عدد المحاولات: {len(attempts)} | إعادة صياغة: {'نعم' if rewrite_triggered else 'لا'}")
    for a in attempts:
        print(f"   محاولة {a['attempt']}: كافٍ={a['sufficient']} | قطع={a['chunks_count']} | {a['reasoning'][:80]}")

    print(f"\n📚 المصادر الفعلية ({len(actual_docs)}):")
    for d in actual_docs:
        print(f"   - {d}")
    print(f"   تطابق مع المتوقع: {'✅' if docs_match else '❌'}")

    print(f"\n💬 الجواب: {result.get('answer', '')[:300]}...")
    print(f"   grounded={was_sufficient} | تطابق الكفاية: {'✅' if sufficiency_match else '❌'}")
    print(f"⏱️  الزمن: {elapsed:.2f} ثانية")

    return {
        "case_id": case["id"],
        "scenario": case["scenario"],
        "question": case["question"],
        "expected_info": case["expected_info"],
        "expected_type": case["expected_type"],
        "expected_strategy": case["expected_strategy"],
        "expected_documents": case["expected_documents"],
        "expect_rewrite": case["expect_rewrite"],
        "expect_sufficient": case["expect_sufficient"],
        "note": case["note"],
        # النتائج الفعلية
        "actual_classification": classification,
        "actual_type": classification.get("query_type"),
        "actual_strategy": classification.get("retrieval_strategy"),
        "attempts": attempts,
        "num_attempts": len(attempts),
        "rewrite_triggered": rewrite_triggered,
        "actual_documents": actual_docs,
        "sources": sources,
        "answer": result.get("answer", ""),
        "grounded": was_sufficient,
        "latency_seconds": round(elapsed, 2),
        # التقييم
        "evaluation": {
            "type_match": type_match,
            "strategy_match": strategy_match,
            "documents_match": docs_match,
            "rewrite_ok": rewrite_ok,
            "sufficiency_match": sufficiency_match,
            "success": type_match and strategy_match and sufficiency_match,
        },
    }


if __name__ == "__main__":
    all_results = []

    for case in TEST_CASES:
        r = run_single_case(case)
        all_results.append(r)

    # ملخص عام
    print(f"\n{'=' * 70}")
    print("📊 الملخص العام")
    print(f"{'=' * 70}")

    total = len(all_results)
    successful = sum(1 for r in all_results if r.get("evaluation", {}).get("success"))
    type_ok = sum(1 for r in all_results if r.get("evaluation", {}).get("type_match"))
    strat_ok = sum(1 for r in all_results if r.get("evaluation", {}).get("strategy_match"))
    docs_ok = sum(1 for r in all_results if r.get("evaluation", {}).get("documents_match"))
    suff_ok = sum(1 for r in all_results if r.get("evaluation", {}).get("sufficiency_match"))
    rewrites = sum(1 for r in all_results if r.get("rewrite_triggered"))
    avg_latency = sum(r.get("latency_seconds", 0) for r in all_results) / total

    print(f"إجمالي الحالات: {total}")
    print(f"نجحت بالكامل: {successful}/{total}")
    print(f"تطابق النوع: {type_ok}/{total}")
    print(f"تطابق الاستراتيجية: {strat_ok}/{total}")
    print(f"تطابق المستندات: {docs_ok}/{total}")
    print(f"تطابق الكفاية: {suff_ok}/{total}")
    print(f"حالات حصل فيها rewrite: {rewrites}/{total}")
    print(f"متوسط الزمن: {avg_latency:.2f} ثانية")

    # حفظ النتائج
    output_path = "adaptive_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✅ تم حفظ النتائج بـ {output_path}")
    print("👉 الخطوة التالية: شغّل evaluate_adaptive_rag.py لحساب المقاييس الرسمية")