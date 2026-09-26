"""
يختبر مرحلة 1 (التصنيف) ومرحلة 2 (الاسترجاع التكيّفي) سوا كسلسلة كاملة:
يصنف السؤال، وبعدين يسترجع فعلياً حسب القرار، ويطبع النتائج التفصيلية.

هاد أول مرة نشوف فيها استرجاع حقيقي من Azure بأكتر من استراتيجية
واحدة بنفس الجلسة - قارن شكل النتائج بين الأسئلة المختلفة.
"""

from dotenv import load_dotenv

load_dotenv()

from query_classifier import classify_query
from adaptive_retriever import adaptive_retrieve

TEST_QUERIES = [
    "كم يوم إجازة سنوية يستحق الموظف؟",
    "قارن نسبة تحمل الموظف بالتأمين الطبي داخل الشبكة مع نسبة التحمل خارجها",
    "شو اسم منصة التواصل المعتمدة للعمل عن بُعد، وهل نفس المنصة مذكورة بسياسة الأمن السيبراني كأداة معتمدة رسمياً؟",
    "حسب سياسة الأمن السيبراني تحديداً، شو شروط كلمة المرور؟",
    "شو كل السياسات المتعلقة بحماية بيانات الموظف بالشركة؟",
    "شو تفاصيل التذكرة رقم TCK-1B0759CD؟",
]

if __name__ == "__main__":
    for query in TEST_QUERIES:
        print(f"\n{'=' * 60}")
        print(f"السؤال: {query}")

        classification = classify_query(query)
        print(f"التصنيف: {classification['query_type']} | الاستراتيجية: {classification['retrieval_strategy']}")
        if classification["metadata_filter"]:
            print(f"فلتر الميتاداتا (خام من الموديل): {classification['metadata_filter']}")
        if classification["sub_queries"]:
            print(f"أسئلة فرعية: {classification['sub_queries']}")

        results = adaptive_retrieve(query, classification, top_k=3)
        print(f"\nعدد القطع المسترجعة: {len(results)}")
        for i, r in enumerate(results, 1):
            sub_note = f"  [من فرعي: {r['matched_sub_query']}]" if r["matched_sub_query"] != query else ""
            print(f"  {i}. {r['source_document']} — {r['section']}{sub_note}")
            print(f"     {r['content'][:100]}...")
