"""
مرحلة 7: توليد الجواب النهائي المدعوم بالمصادر (Grounded Answer).

يستخدم نتيجة retrieve_with_quality_loop() مباشرة:
- لو السياق كافٍ (sufficient=True): يولّد جواباً حقيقياً من الموديل مع المصادر.
- لو غير كافٍ رغم كل محاولات إعادة الصياغة: يرجع رسالة "لا توجد معلومات
  كافية" بشكل حتمي (deterministic) - بدون ما نطلب من الموديل يحاول
  يجاوب من سياق نعرف أصلاً إنه ضعيف. هاي طبقة حماية إضافية ضد الهلوسة،
  أقوى من الاعتماد على تعليمة الـprompt وحدها (نفس درس مرحلة 4 بالضبط).
"""

import os
from openai import AzureOpenAI

ANSWER_SYSTEM_PROMPT = """أنت مساعد داخلي للموظفين، تجاوب فقط بناءً على
المعلومات المرفقة لك من مستندات الشركة.

قواعد صارمة:
- جاوب فقط من المعلومات الموجودة في السياق المرفق، ولا تستخدم أي معرفة خارجية.
- اذكر اسم المستند والقسم المصدر لكل معلومة تذكرها.
- كن مختصراً ومباشراً، ولا تخترع أي تفاصيل أو أرقام غير موجودة حرفياً بالسياق.

قاعدة إلزامية للأسئلة متعددة الأجزاء:
- لو السؤال يحتوي أكتر من جزء (مثلاً: "شو X، وهل Y؟")، عالج كل جزء لحاله.
- لو جزء معين من السؤال غير مدعوم بأي نص صريح بالسياق المرفق، لا تستنتج
  ولا تخمّن إجابته، ولا تربطه بمصدر غير داعم له. صرّح بوضوح:
  "لا توجد معلومات كافية بالمستندات المتاحة للإجابة على هذا الجزء تحديداً"
  لهاد الجزء بالذات، حتى لو باقي الأجزاء مُجابة بثقة."
"""


def _client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def _build_context_string(chunks: list[dict]) -> str:
    parts = [f"[من: {c['source_document']} - {c['section']}]\n{c['content']}" for c in chunks]
    return "\n\n---\n\n".join(parts)


def _deduplicate_sources(chunks: list[dict]) -> list[dict]:
    # ممكن يصير تكرار لو أكتر من sub_query أو محاولة رجعوا نفس القطعة بالضبط
    seen = set()
    unique = []
    for c in chunks:
        key = (c["source_document"], c["section"])
        if key not in seen:
            seen.add(key)
            unique.append({"source_document": c["source_document"], "section": c["section"]})
    return unique


def generate_grounded_answer(query: str, pipeline_result: dict, history: list[dict] | None = None) -> dict:
    """
    ياخد نتيجة retrieve_with_quality_loop() + تاريخ المحادثة (اختياري)，
    ويرجع dict فيه: answer, sources, grounded
    """
    history = history or []
    chunks = pipeline_result["chunks"]

    if not pipeline_result["sufficient"]:
        attempts_count = len(pipeline_result["attempts"])
        return {
            "answer": (
                "لا توجد معلومات كافية بقاعدة المعرفة للإجابة على هذا السؤال بدقة، "
                f"رغم إعادة محاولة الاسترجاع {attempts_count} مرة/مرات بصياغات "
                "واستراتيجيات مختلفة."
            ),
            "sources": [],
            "grounded": False,
        }

    client = _client()
    chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    context_str = _build_context_string(chunks)

    messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]
    messages.extend(history)  # <-- الإضافة الأساسية: نفس فلسفة llm_service.py القديمة
    messages.append(
        {"role": "user", "content": f"السياق المتاح:\n{context_str}\n\nسؤال المستخدم: {query}"}
    )

    response = client.chat.completions.create(
        model=chat_deployment,
        messages=messages,
        temperature=0.2,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": _deduplicate_sources(chunks),
        "grounded": True,
    }