"""
مرحلة 1: تحليل وتصنيف السؤال (Query Understanding).

نداء واحد للموديل يرجع:
- نوع السؤال (simple / multi_source / comparison / multi_step)
- استراتيجية الاسترجاع المقترحة (vector / keyword / hybrid / metadata_filtered)
- فلتر ميتاداتا لو السؤال حدد مستند معين صراحة
- أسئلة فرعية لو السؤال مقارنة أو متعدد مصادر (لاسترجاع متوازي)

هاي أول مرحلة بكل الـpipeline - قرارها بيحدد شكل كل المراحل الجاية.
"""

import os
import json
from openai import AzureOpenAI

VALID_TYPES = {"simple", "multi_source", "comparison", "multi_step"}
VALID_STRATEGIES = {"vector", "keyword", "hybrid", "metadata_filtered"}

CLASSIFIER_SYSTEM_PROMPT = """أنت محلل أسئلة لنظام RAG داخلي لسياسات شركة. مهمتك تصنيف سؤال المستخدم فقط - لا تجاوب على السؤال نفسه.

صنّف السؤال لأحد 4 أنواع بالضبط:
- "simple": سؤال معلوماتي بسيط، إجابته موجودة كمعلومة واحدة مباشرة بمستند واحد.
- "multi_source": السؤال يحتاج معلومات مبعثرة عبر أكتر من قسم أو مستند، بدون حاجة لمقارنة أو ترابط منطقي بينها.
- "comparison": السؤال يطلب مقارنة صريحة بين عنصرين أو أكتر **مسمّيين بالاسم صراحة بنص السؤال نفسه** (مثال: "قارن بين X وY"، "شو الفرق بين A وB").
- "multi_step": السؤال يحتاج استدلال متسلسل - جزء منه يعتمد على نتيجة جزء آخر.

قاعدة فاصلة مهمة جداً بين comparison وmulti_step:
اختبر كل سؤال فرعي محتمل بهاد السؤال: "هل أقدر أصوغ هاد السؤال الفرعي بمصطلح محدد الآن، من كلام المستخدم فقط، بدون أي نتيجة استرجاع سابقة؟"
- لو الجواب "نعم" لكل الأسئلة الفرعية (كل عنصر مسمّى بالاسم صراحة) → "comparison".
- لو أي سؤال فرعي يحتوي مرجعاً غير محلول يعتمد على إجابة سؤال فرعي آخر (مثل: "نفس الشيء"، "هذا"، "تلك المنصة/القيمة/الجهة المذكورة") ولا يمكن صياغته بمصطلح محدد إلا بعد معرفة إجابة أولى → "multi_step"، حتى لو ظاهرياً يبدو السؤال وكأنه مقارنة أو تحقق.

اقترح استراتيجية استرجاع واحدة من:
- "vector": لصياغة بعيدة عن لغة المستند (معنى واضح لكن كلمات مختلفة).
- "keyword": لتطابق حرفي دقيق، بس فقط لو السؤال نفسه يحتوي **معرّفاً محدداً معروفاً مسبقاً** (رقم تذكرة، كود، اسم علم) والمطلوب إيجاده بالنص كما هو بالضبط.
- "hybrid": الافتراضي العام لمعظم الأسئلة، بما فيها أي سؤال يطلب اكتشاف قيمة أو رقم أو نسبة **غير معروفة مسبقاً** من فهم المستند (مثال: "شو نسبة X؟"، "كم قيمة Y؟") - هون المستخدم ما عنده رقم يدور عنه حرفياً، هو بده يكتشف الرقم من فهم السؤال دلالياً، فـ"hybrid" هو الصحيح وليس "keyword".
- "metadata_filtered": لو السؤال حدد مستند أو قسم معين صراحة.

تحذير مهم: مجرد وجود مفهوم رقمي أو نسبة مئوية بالسؤال لا يعني تلقائياً "keyword". اسأل نفسك: هل عند المستخدم قيمة معروفة بالفعل يريد مطابقتها حرفياً، أم هو يطلب اكتشاف قيمة غير معروفة له؟ الحالة الثانية = hybrid.

لو نوع السؤال "comparison" أو "multi_source"، قسّمه لأسئلة فرعية مستقلة (sub_queries) - كل واحد يغطي جزء واحد بوضوح.
لو الاستراتيجية "metadata_filtered"، حدد اسم المستند المستهدف بالضبط بحقل metadata_filter.

أرجع النتيجة بصيغة JSON فقط، بدون أي نص إضافي، بهاد الشكل بالضبط:
{
  "query_type": "simple أو multi_source أو comparison أو multi_step",
  "retrieval_strategy": "vector أو keyword أو hybrid أو metadata_filtered",
  "metadata_filter": null أو اسم المستند كنص,
  "sub_queries": null أو قائمة نصوص,
  "reasoning": "سبب مختصر للتصنيف (جملة أو جملتين)"
}"""


def _client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def classify_query(query: str) -> dict:
    client = _client()
    chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

    response = client.chat.completions.create(
        model=chat_deployment,
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_object"},  # يجبر الموديل يرجع JSON صالح تركيبياً دايماً
        temperature=0.1,  # منخفضة عمداً - بدنا تصنيف ثابت ومتوقع، مش إبداع
    )

    raw = response.choices[0].message.content
    return _parse_and_validate(raw, query)


def _parse_and_validate(raw: str, original_query: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # فشل تحليل الـJSON نفسه (نادر جداً مع response_format، بس نتحسب له)
        return _fallback_classification(original_query, reason="فشل تحليل رد الموديل كـJSON")

    query_type = data.get("query_type")
    strategy = data.get("retrieval_strategy")

    if query_type not in VALID_TYPES or strategy not in VALID_STRATEGIES:
        # الموديل رجع قيمة غير متوقعة (هلوسة بالتصنيف نفسه، مش بالجواب) -
        # نرجع تصنيف افتراضي آمن بدل ما نوقف النظام بالكامل
        return _fallback_classification(
            original_query,
            reason=f"قيم غير صالحة من الموديل: query_type={query_type}, strategy={strategy}",
        )

    return {
        "query_type": query_type,
        "retrieval_strategy": strategy,
        "metadata_filter": data.get("metadata_filter"),
        "sub_queries": data.get("sub_queries"),
        "reasoning": data.get("reasoning", ""),
    }


def _fallback_classification(query: str, reason: str) -> dict:
    # خطة أمان: أعم وأبسط مسار ممكن (hybrid على السؤال الأصلي كامل).
    # فلسفة "أفضل جواب عادي من ولا جواب أبداً" - نفس مبدأ MAX_AGENT_STEPS
    # بالـagent الأسبوع الماضي (حد أقصى بدل حلقة لا نهائية أو انهيار)
    return {
        "query_type": "simple",
        "retrieval_strategy": "hybrid",
        "metadata_filter": None,
        "sub_queries": None,
        "reasoning": f"[fallback] {reason}",
    }