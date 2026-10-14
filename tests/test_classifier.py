"""
يجرب classify_query() مع توقع صريح للنوع والاستراتيجية، ويقارن تلقائياً
- نفس نمط test_cases.py المستخدم لتقييم الـagent الأسبوع الماضي.
"""

from dotenv import load_dotenv

load_dotenv()

from query_classifier import classify_query

TEST_QUERIES = [
    {
        "query": "كم يوم إجازة سنوية يستحق الموظف؟",
        "expected_type": "simple",
        "expected_strategy": "hybrid",
    },
    {
        "query": "قارن نسبة تحمل الموظف بالتأمين الطبي داخل الشبكة مع نسبة التحمل خارجها",
        "expected_type": "comparison",
        "expected_strategy": "hybrid",
    },
    {
        # مثال multi-hop حقيقي: الجزء الثاني مستحيل صياغته بدون معرفة
        # اسم المنصة (Microsoft Teams) من جواب الجزء الأول أولاً
        "query": "شو اسم منصة التواصل المعتمدة للعمل عن بُعد، وهل نفس المنصة مذكورة بسياسة الأمن السيبراني كأداة معتمدة رسمياً؟",
        "expected_type": "multi_step",
        "expected_strategy": "hybrid",
    },
    {
        "query": "حسب سياسة الأمن السيبراني تحديداً، شو شروط كلمة المرور؟",
        "expected_type": "simple",
        "expected_strategy": "metadata_filtered",
    },
    {
        "query": "شو كل السياسات المتعلقة بحماية بيانات الموظف بالشركة؟",
        "expected_type": "multi_source",
        "expected_strategy": "hybrid",
    },
    {
        "query": "شو تفاصيل التذكرة رقم TCK-1B0759CD؟",
        "expected_type": "simple",
        "expected_strategy": "keyword",
    },
]

if __name__ == "__main__":
    correct_type = 0
    correct_strategy = 0

    for case in TEST_QUERIES:
        result = classify_query(case["query"])

        type_match = result["query_type"] == case["expected_type"]
        strategy_match = result["retrieval_strategy"] == case["expected_strategy"]
        correct_type += type_match
        correct_strategy += strategy_match

        print(f"\n{'=' * 60}")
        print(f"السؤال: {case['query']}")
        print(f"النوع: متوقع={case['expected_type']} | فعلي={result['query_type']} | {'✅' if type_match else '❌'}")
        print(f"الاستراتيجية: متوقع={case['expected_strategy']} | فعلي={result['retrieval_strategy']} | {'✅' if strategy_match else '❌'}")
        print(f"فلتر الميتاداتا: {result['metadata_filter']}")
        print(f"أسئلة فرعية: {result['sub_queries']}")
        print(f"السبب: {result['reasoning']}")

    total = len(TEST_QUERIES)
    print(f"\n{'=' * 60}")
    print(f"دقة النوع: {correct_type}/{total}")
    print(f"دقة الاستراتيجية: {correct_strategy}/{total}")