"""
أدوات فهم الصور والـOCR، لنفس Agent الموجود (agent_service.py).
نفس نمط tools.py: schema + تنفيذ فعلي لكل أداة.

قرار تصميمي مهم: أداتان منفصلتان (مش أداة واحدة "ذكية" تسوي الاثنين
دايماً) — عشان الموديل نفسه هو يلي بيقرر يحتاج فهم بصري بس، أو OCR بس،
أو الاثنين سوا، حسب السؤال. هيك نحقق متطلب "لا OCR عشوائي" بنفس آلية
tool-calling الموجودة أصلاً، بدون أي منطق إضافي صلب بالكود.
"""

import os
import base64
from openai import AzureOpenAI


def _client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _mime_type(image_path: str) -> str:
    ext = image_path.lower().rsplit(".", 1)[-1]
    return {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "jpeg")


def _vision_call(image_path: str, system_prompt: str, user_text: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"الصورة غير موجودة: {image_path}")

    client = _client()
    b64 = _encode_image(image_path)
    mime = _mime_type(image_path)

    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                    },
                ],
            },
        ],
        max_completion_tokens=500,
    )
    return response.choices[0].message.content


# ============================================================
# 1) Image Understanding — فهم بصري عام
# ============================================================

ANALYZE_SYSTEM_PROMPT = """أنت نظام فهم صور. صف أو جاوب فقط بناءً على ما
تراه فعلياً بالصورة المرفقة - أشخاص، أجسام، مشهد، ألوان، تفاصيل مرئية.

لا تحاول قراءة أو استخراج أي نص موجود بالصورة بدقة حرفية - إذا كان هناك
نص وسُئلت عن محتواه الدقيق، اذكر أن هناك نصاً ظاهراً لكن التفاصيل الدقيقة
تحتاج أداة قراءة نص مخصصة.

إذا كانت المعلومة المطلوبة غير ظاهرة فعلياً بالصورة، قل ذلك صراحة بدل
التخمين."""


def tool_analyze_image(image_path: str, question: str) -> dict:
    try:
        answer = _vision_call(image_path, ANALYZE_SYSTEM_PROMPT, question)
        return {"success": True, "answer": answer}
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"خطأ أثناء تحليل الصورة: {e}"}


# ============================================================
# 2) OCR — استخراج نص (عربي/إنجليزي)
# ============================================================

OCR_SYSTEM_PROMPT = """أنت نظام OCR. مهمتك الوحيدة استخراج كل نص ظاهر
فعلياً بالصورة المرفقة، حرفياً وبدقة - عربي أو إنجليزي أو الاثنين معاً.

قواعد صارمة:
- انسخ النص كما هو بالضبط، بدون تصحيح إملائي أو تخمين كلمات غير واضحة.
- إذا كانت الصورة رديئة الجودة أو النص غير واضح جزئياً، اذكر ذلك صراحة
  ولا تخترع نصاً غير موجود فعلياً.
- إذا لم يوجد أي نص بالصورة إطلاقاً، قل ذلك بوضوح.
- حافظ على ترتيب النص كما يظهر بالصورة (عناوين، حقول، أرقام).
- مهم جداً: لا تدع صياغة السؤال تؤثر على إجابتك. إذا سُئلت "هل يوجد
  اسم X؟" ولم يكن X موجوداً حرفياً بالصورة، أجب بالنص الفعلي الموجود
  فقط - لا تؤكد أو تخترع اسماً لمجرد أن السؤال افترض وجوده."""

def tool_extract_text_ocr(image_path: str) -> dict:
    try:
        extracted = _vision_call(
            image_path, OCR_SYSTEM_PROMPT, "استخرج كل نص موجود بهذه الصورة."
        )
        return {"success": True, "extracted_text": extracted}
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"خطأ أثناء استخراج النص: {e}"}


# ============================================================
# Schemas بصيغة Azure OpenAI function/tool calling
# ============================================================

IMAGE_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": (
                "يحلل صورة مرفقة ويجاوب على أسئلة عن محتواها البصري: أشخاص، "
                "أجسام، ألوان، مشهد عام. استخدمها عندما يسأل المستخدم عن "
                "مظهر الصورة أو تفاصيلها المرئية - وليس لقراءة نص داخلها "
                "بدقة حرفية."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "مسار ملف الصورة المرفقة",
                    },
                    "question": {
                        "type": "string",
                        "description": "سؤال المستخدم عن الصورة",
                    },
                },
                "required": ["image_path", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_text_ocr",
            "description": (
                "يستخرج أي نص مكتوب موجود فعلياً داخل صورة مرفقة (عربي أو "
                "إنجليزي)، بدقة حرفية. استخدمها فقط عندما يسأل المستخدم عن "
                "محتوى نصي محدد بالصورة (رقم، تاريخ، عنوان، فقرة نص) - ولا "
                "تستخدمها لأسئلة عن المظهر البصري العام."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "مسار ملف الصورة المرفقة",
                    }
                },
                "required": ["image_path"],
            },
        },
    },
]

IMAGE_TOOL_IMPLEMENTATIONS = {
    "analyze_image": tool_analyze_image,
    "extract_text_ocr": tool_extract_text_ocr,
}