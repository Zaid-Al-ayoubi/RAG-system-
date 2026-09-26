"""
مرحلة 3: إعادة صياغة السؤال (Query Rewriting).
تُستدعى فقط لما فحص الكفاية (مرحلة 4) يقرر إنه السياق المسترجع غير كافٍ.
"""

import os
from openai import AzureOpenAI

REWRITE_SYSTEM_PROMPT = """أنت خبير بإعادة صياغة أسئلة البحث لتحسين نتائج الاسترجاع من مستندات سياسات شركة.

النتائج المسترجعة لأول محاولة لم تكن كافية. أعد صياغة السؤال بكلمات ومرادفات
مختلفة، أقرب للغة الرسمية المستخدمة بمستندات الشركة (سياسات، أنظمة، لوائح)،
بدون ما تغيّر معنى السؤال الأصلي أو تضيف تفاصيل غير موجودة فيه.

استخدم محتوى النتائج غير الكافية (المرفقة) كدليل على لغة المستندات الفعلية،
حتى لو لم تكن مرتبطة مباشرة بالسؤال.

أرجع فقط السؤال المعاد صياغته كنص عادي، بدون أي شرح أو علامات اقتباس."""


def _client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def rewrite_query(original_query: str, previous_chunks: list[dict]) -> str:
    client = _client()
    chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

    context_hint = (
        "\n".join(c["content"][:200] for c in previous_chunks[:3])
        if previous_chunks
        else "(لا توجد نتائج سابقة إطلاقاً)"
    )

    response = client.chat.completions.create(
        model=chat_deployment,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"السؤال الأصلي: {original_query}\n\nنتائج المحاولة السابقة (للاستئناس بلغة المستندات):\n{context_hint}",
            },
        ],
        temperature=0.4,  # أعلى شوي من باقي المراحل - بدنا تنويع فعلي بالصياغة، مش تكرار حرفي
    )

    rewritten = (response.choices[0].message.content or "").strip()
    # حماية: لو الموديل رجع نص فاضي لأي سبب، نرجع السؤال الأصلي بدل ما
    # نمرر سؤال فاضي للمحاولة الجاية ونكسر الحلقة
    return rewritten if rewritten else original_query
