import os
import json
from openai import AzureOpenAI

# ⬇️ هاد لازم يضل موجود — هو الأصلي يلي كان بالملف من البداية
QUALITY_CHECK_SYSTEM_PROMPT = """أنت مقيّم صارم لجودة نتائج بحث. مهمتك فقط تحديد
هل النصوص المسترجعة تحتوي فعلاً على إجابة كافية لسؤال المستخدم - لا تجاوب
على السؤال بنفسك، ولا تخترع أي معلومة غير موجودة حرفياً بالنصوص.

اعتبر السياق "غير كافٍ" (sufficient: false) لو:
- النصوص لا تتطرق للموضوع المطلوب إطلاقاً
- النصوص عامة أو بعيدة عن التفصيل المحدد المطلوب بالسؤال
- ناقص جزء أساسي من السؤال (مثلاً السؤال يطلب رقمين واحد بس موجود)

أرجع النتيجة بصيغة JSON فقط، بدون أي نص إضافي:
{
  "sufficient": true أو false,
  "reasoning": "سبب مختصر (جملة أو جملتين)"
}"""

QUALITY_CHECK_MULTI_SOURCE_PROMPT = """أنت مقيّم صارم لجودة نتائج بحث لسؤال
استكشافي عريض يطلب تجميع معلومات مبعثرة (مش قيمة واحدة محددة).

اعتبر السياق "كافٍ" (sufficient: true) لو النصوص تغطي عدة سياسات/معلومات
حقيقية ذات صلة مباشرة بموضوع السؤال - حتى لو مش شاملة لكل التفاصيل
الممكنة نظرياً. لا ترفض السياق فقط لأنه "غير شامل بالكامل" - هاد النوع
من الأسئلة أصلاً ما إله إجابة نهائية واحدة.

اعتبره "غير كافٍ" فقط لو النصوص لا تتطرق لموضوع السؤال إطلاقاً، أو
كلها بعيدة تماماً عن المطلوب.

أرجع JSON فقط:
{"sufficient": true أو false, "reasoning": "..."}"""


# ⬇️ هاد هو الجديد يلي ضفناه
QUALITY_CHECK_MULTI_PART_PROMPT = """أنت مقيّم صارم لجودة نتائج بحث لسؤال
مكوّن من عدة أجزاء. مهمتك تحديد لكل جزء من السؤال هل النصوص المسترجعة
تجيب عليه فعلياً أم لا - بشكل مستقل لكل جزء.

أرجع النتيجة بصيغة JSON فقط:
{
  "parts": [
    {"part": "وصف الجزء", "sufficient": true أو false, "reasoning": "..."}
  ],
  "sufficient": true أو false  // true فقط لو كل الأجزاء كافية
}"""


def _client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def check_sufficiency(query: str, chunks: list[dict], query_type: str = "simple") -> dict:
    if not chunks:
        return {"sufficient": False, "reasoning": "لا توجد أي نتائج مسترجعة إطلاقاً.", "parts": None}

    if query_type in ("multi_step", "comparison"):
        system_prompt = QUALITY_CHECK_MULTI_PART_PROMPT
    elif query_type == "multi_source":
        system_prompt = QUALITY_CHECK_MULTI_SOURCE_PROMPT
    else:
        system_prompt = QUALITY_CHECK_SYSTEM_PROMPT
   
    system_prompt = (
        QUALITY_CHECK_MULTI_PART_PROMPT
        if query_type in ("multi_step", "comparison")
        else QUALITY_CHECK_SYSTEM_PROMPT
    )

    client = _client()
    chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

    context_str = "\n\n---\n\n".join(c["content"] for c in chunks)

    response = client.chat.completions.create(
        model=chat_deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"السؤال: {query}\n\nالنصوص المسترجعة:\n{context_str}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    try:
        data = json.loads(response.choices[0].message.content)
        return {
            "sufficient": bool(data.get("sufficient", False)),
            "reasoning": data.get("reasoning", ""),
            "parts": data.get("parts"),  # None لو كان simple/multi_source (ما رجعش parts أصلاً)
        }
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {
            "sufficient": False,
            "reasoning": "[fallback] فشل تحليل رد فحص الكفاية",
            "parts": None,
        }