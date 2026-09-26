"""
تقييم جودة الـRAG Agent باستخدام TruLens: RAG Triad
(Context Relevance / Groundedness / Answer Relevance)

⚠️ يحتاج فعلياً اتصال شغال بـAzure OpenAI (لأنه بيستخدمه كـ"حكم" judge
يقيّم الجودة)، فما رح يشتغل لحد ما يرجعلك الوصول. الهيكل جاهز بالكامل،
بس شغّله بعد رجوع الوصول وتفعيل USE_MOCK_SERVICES=false.

⚠️ تنويه صريح: تأكدت من صحة كل أسماء الـimports والدوال هون فعلياً
بتشغيل الكود عندي (مش نظرياً من الذاكرة)، بس TruLens مكتبة بتتطور
بسرعة، فلو طلع عندك خطأ استيراد بنسخة مختلفة، راجع trulens.org.
"""

import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # لازم ينفذ هون، قبل استيراد search_service و llm_service بالأسفل
# عشان USE_MOCK_SERVICES يكون موجود فعلياً بالـenvironment وقت ما بيتحسب

from trulens.core import TruSession, Metric, Selector
from trulens.apps.app import TruApp
from trulens.apps.app import instrument as default_instrument  # للدوال العادية
from trulens.core.otel.instrument import instrument  # الكلاس نفسه، عشان نقدر نمرر span_type
from trulens.otel.semconv.trace import SpanAttributes
from trulens.providers.openai import AzureOpenAI as TruAzureOpenAI

from search_service import search_documents
from llm_service import generate_answer


def _retrieval_attributes(ret, exception, *args, **kwargs):
    # بتنعرض هاي الدالة على كل استدعاء لـretrieve()، وبترجع dict
    # يوصف الـspan: شو كان السؤال (query) وشو رجع (نصوص القطع بس، لأنه
    # هاد اللي TruLens بده لتقييم الـcontext - بغض النظر عن شكل رجعة الدالة الفعلية)
    query = args[1] if len(args) > 1 else kwargs.get("query")
    contents = [c["content"] for c in ret] if ret else []
    return {
        SpanAttributes.RETRIEVAL.QUERY_TEXT: query,
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: contents,
    }


class RAGPipeline:
    """نلف نفس الـpipeline المبني بـsearch_service و llm_service بكلاس واحد،
    ونعلّم الدوال يلي بدنا TruLens يراقبها بـinstrument."""

    # span_type=RETRIEVAL يعلّم TruLens إنه هاي الدالة تحديداً هي مصدر
    # الـ"context" - وهيك Selector.select_context() تحت بيعرف يلاقيها
    # تلقائياً بدون ما نأشر عليها بالاسم يدوياً
    @instrument(span_type=SpanAttributes.SpanType.RETRIEVAL, attributes=_retrieval_attributes)
    def retrieve(self, query: str) -> list[dict]:
        # منرجع القواميس الكاملة (مش نصوص بس) عشان generate_answer يضل
        # قادر يستخدم source_document وsection زي ما هو متوقع منه
        return search_documents(query, top_k=3)

    @default_instrument
    def answer(self, query: str) -> str:
        chunks = self.retrieve(query)
        return generate_answer(query, chunks, history=[])


rag = RAGPipeline()

# الـprovider هون هو "الحكم" (judge) اللي بيقيّم الجودة - لازم يكون LLM حقيقي.
# بنمررله المفاتيح صراحة بدل ما نعتمد على قراءته التلقائية من الـenvironment،
# لأنه بيدور عن أسماء متغيرات قياسية (AZURE_OPENAI_API_KEY, OPENAI_API_VERSION)
# مختلفة عن الأسماء اللي اخترناها نحن بباقي المشروع
judge = TruAzureOpenAI(
    deployment_name=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)


def _groundedness_fn(source, statement):
    # groundedness_measure_with_cot_reasons بده source كنص واحد (str)،
    # بس Selector.select_context(collect_list=True) بيرجعلنا القطع كـlist،
    # فبندمجهم بسطر واحد قبل ما نمررهم للحكم
    joined = "\n".join(source) if isinstance(source, list) else source
    return judge.groundedness_measure_with_cot_reasons(joined, statement)


# مهم: تحت وضع OTEL، الـ.on() بياخد dict واحد بس (مش kwargs منفصلة)
f_context_relevance = (
    Metric(implementation=judge.context_relevance_with_cot_reasons, name="Context Relevance")
    .on({
        "question": Selector.select_record_input(),
        "context": Selector.select_context(collect_list=False),  # يقيّم كل قطعة لحالها
    })
    .aggregate(np.mean)  # ثم ناخد متوسط الدقة عبر كل القطع
)

f_groundedness = (
    Metric(implementation=_groundedness_fn, name="Groundedness")
    .on({
        "source": Selector.select_context(collect_list=True),  # كل القطع دفعة وحدة
        "statement": Selector.select_record_output(),
    })
)

f_answer_relevance = (
    Metric(implementation=judge.relevance_with_cot_reasons, name="Answer Relevance")
    .on({
        "prompt": Selector.select_record_input(),
        "response": Selector.select_record_output(),
    })
)

session = TruSession()

tru_rag = TruApp(
    rag,
    app_name="hr-rag-agent",
    app_version="v1",
    feedbacks=[f_context_relevance, f_groundedness, f_answer_relevance],
)

# أسئلة اختبار تغطي الملفات الخمسة - وسّعها حسب مستنداتك الفعلية
test_questions = [
    "كم يوم إجازة سنوية يستحق الموظف؟",
    "شو نسبة تحمل الموظف بالتأمين الطبي داخل الشبكة المعتمدة؟",
    "كيف بتنحسب قيمة ساعة العمل الإضافي؟",
    "ما هي سياسة كلمات المرور؟",
]

if __name__ == "__main__":
    with tru_rag as recording:
        for q in test_questions:
            result = rag.answer(q)
            # نطبع الجواب الفعلي مباشرة (بمعزل عن TruLens) عشان نتأكد
            # إنه الـpipeline نفسه (retrieve + generate) شغال صح على Azure الحقيقي
            print(f"\n[سؤال] {q}")
            print(f"[جواب] {result}")

    # مهم: TruLens بيحسب معايير التقييم الثلاث بالخلفية (async) بعد ما ترجع
    # الإجابة مباشرة، مش أثناءها. force_flush() بتوقف السكريبت وتستنى
    # لحد ما كل التقييمات المعلقة تخلص قبل ما نطبع النتيجة
    session.force_flush()

    # حتى بعد force_flush، لاحظنا أحياناً تأخير بسيط (حالة سباق/race condition)
    # قبل ما النتائج تكون قابلة للاستعلام فعلياً - فنجرب كم مرة مع فاصل بسيط
    import time
    for attempt in range(5):
        records_df, _ = session.get_records_and_feedback(app_ids=[tru_rag.app_id])
        if any("Groundedness" in c or "Answer Relevance" in c for c in records_df.columns):
            break
        time.sleep(3)
        print(f"...لسا التقييمات ما خلصت، بحاول كمان مرة ({attempt + 1}/5)")
    print("\n=== تفاصيل كل سجل ===")
    print(records_df.to_string())

    print("\n=== الملخص ===")
    print(session.get_leaderboard())
    # session.run_dashboard()  # يفتح لوحة تفاعلية بالمتصفح لمراجعة كل سؤال بالتفصيل