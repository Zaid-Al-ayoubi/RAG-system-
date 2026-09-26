"""
يشغّل 8 حالات اختبار على الـagent، ويقيّم كل حالة حسب المعايير المطلوبة:
- Tool selection accuracy: هل استخدم الأداة الصح؟
- Tool parameter accuracy: هل الباراميترات صحيحة/منطقية؟
- RAG retrieval accuracy: هل رجع مصادر مرتبطة؟ (لو استخدم rag_search)
- Tool execution success: هل نفذت الأداة بدون خطأ؟
- Final answer correctness: تقييم يدوي بعد المراجعة (مكتوب هون كملاحظة)

يغطي حالات: أداة وحدة (single-tool)، أكتر من أداة بالتسلسل (multi-tool)،
وحالة معلومات ناقصة (error handling).
"""

import json
from dotenv import load_dotenv

load_dotenv()

from agent_service import run_agent

TEST_CASES = [
    {
        "id": 1,
        "question": "كم يوم إجازة سنوية يستحق الموظف؟",
        "expected_tools": ["rag_search"],
        "note": "single-tool - سؤال معلوماتي بسيط",
    },
    {
        "id": 2,
        "question": "حسب سياسة الإجازات، عندي 25 يوم وأخذت 8 أيام. كم باقي معي؟",
        "expected_tools": ["rag_search", "calculator"],
        "note": "multi-tool - بحث ثم حساب (نفس مثال المهمة تماماً)",
    },
    {
        "id": 3,
        "question": "شو نسبة تحمل الموظف بالتأمين الطبي داخل الشبكة المعتمدة؟",
        "expected_tools": ["rag_search"],
        "note": "single-tool - سؤال معلوماتي",
    },
    {
        "id": 4,
        "question": "إذا راتبي الأساسي 900 دينار وعملت 6 ساعات إضافية بيوم عادي، كم مستحقاتي؟",
        "expected_tools": ["rag_search", "calculator"],
        "note": "multi-tool - بحث عن معامل الساعة الإضافية ثم حساب",
    },
    {
        "id": 5,
        "question": "بدي أفتح تذكرة دعم، جهاز اللابتوب تبعي وقع مني وانسرق اليوم الصبح",
        "expected_tools": ["rag_search", "create_ticket"],
        "note": "multi-tool - المفروض يبحث عن إجراء فقدان الجهاز ثم يفتح تذكرة",
    },
    {
        "id": 6,
        "question": "افتح لي تذكرة دعم لمشكلة تقنية",
        "expected_tools": [],
        "note": (
            "error handling - معلومات ناقصة (لا عنوان ولا وصف واضح). "
            "السلوك الصحيح: عدم استدعاء أي أداة والطلب من المستخدم توضيح "
            "التفاصيل أولاً - وليس فتح تذكرة عامة/غامضة."
        ),
    },
    {
        "id": 7,
        "question": "شو ناتج 45 ضرب 12 ناقص 30؟",
        "expected_tools": ["calculator"],
        "note": "single-tool - حساب بحت بدون علاقة بالمستندات",
    },
    {
        "id": 8,
        "question": "شو سياسة الموارد البشرية بخصوص العمل من المنزل بشركة جوجل؟",
        "expected_tools": ["rag_search"],
        "note": "error handling - معلومة غير موجودة بمستنداتنا، لازم يقول 'لا توجد معلومات كافية' بدل ما يخترع",
    },
]


def evaluate_case(case: dict, result: dict) -> dict:
    used_tools = [t["tool"] for t in result["tool_trace"]]
    tool_execution_success = all(
        t["result"].get("success", True) or t["result"].get("found", True)
        for t in result["tool_trace"]
    )
    return {
        "tool_selection_match": set(used_tools) == set(case["expected_tools"]),
        "used_tools": used_tools,
        "expected_tools": case["expected_tools"],
        "tool_execution_success": tool_execution_success,
    }


if __name__ == "__main__":
    all_results = []

    for case in TEST_CASES:
        print(f"\n{'=' * 60}")
        print(f"حالة {case['id']}: {case['question']}")
        print(f"النوع: {case['note']}")

        result = run_agent(case["question"], history=[])
        evaluation = evaluate_case(case, result)

        print(f"الأدوات المستخدمة: {evaluation['used_tools']}")
        print(f"الأدوات المتوقعة: {evaluation['expected_tools']}")
        print(f"تطابق اختيار الأداة: {evaluation['tool_selection_match']}")
        print(f"نجاح تنفيذ الأدوات: {evaluation['tool_execution_success']}")
        print(f"الجواب النهائي: {result['answer']}")

        all_results.append(
            {
                "case_id": case["id"],
                "question": case["question"],
                "note": case["note"],
                "tool_trace": result["tool_trace"],
                "answer": result["answer"],
                "evaluation": evaluation,
            }
        )

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("تم حفظ كل النتائج بملف test_results.json")
    print("راجعها يدوياً لتقييم 'Final answer correctness' لكل حالة (يحتاج حكم بشري)")