"""
مرحلة 5: إعادة الترتيب والتصفية (Re-ranking).

تُطبّق على كل استراتيجيات الاسترجاع بدون استثناء - حتى hybrid رغم
الـsemantic ranker المدمج فيها من Azure. السبب: الـsemantic ranker بيرتب
الأفضل بالأول، لكن ما "يصفي" القطع الضعيفة - وهاد بالضبط ما شفناه
تجريبياً (قطعة "هيكلية الراتب" ظهرت كمصدر بسؤال عن الإجازات رغم
استخدام hybrid). إعادة الترتيب هون وظيفتها أوسع من "ترتيب": تقييم
صلة صريح + استبعاد أي قطعة تحت حد أدنى معقول.
"""

import os
import json
from openai import AzureOpenAI

RERANK_SYSTEM_PROMPT = """أنت نظام تقييم صلة نصوص مسترجعة بسؤال معين.

لكل نص مرقم أدناه، قيّم درجة صلته الفعلية بالإجابة على السؤال، من 0
(غير مرتبط إطلاقاً) إلى 1 (إجابة مباشرة ودقيقة للسؤال بالتحديد).

كن صارماً: نص يذكر نفس الموضوع العام لكن لا يجيب التفصيل المطلوب
بالسؤال يستحق درجة منخفضة (0.2-0.4)، وليس درجة عالية لمجرد التشابه
الموضوعي العام.

أرجع JSON فقط بهاد الشكل، بدون أي نص إضافي:
{"scores": [{"index": 0, "score": 0.9}, {"index": 1, "score": 0.1}]}"""


def _client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def rerank_and_filter(query: str, candidates: list[dict], top_k: int = 3, min_score: float = 0.5) -> list[dict]:
    if not candidates:
        return []

    client = _client()
    chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

    numbered = "\n\n".join(f"[{i}] {c['content']}" for i, c in enumerate(candidates))

    response = client.chat.completions.create(
        model=chat_deployment,
        messages=[
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": f"السؤال: {query}\n\nالنصوص:\n{numbered}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,  # صفر - تقييم متسق ومتكرر، مش إبداعي
    )

    try:
        data = json.loads(response.choices[0].message.content)
        scores = {item["index"]: float(item["score"]) for item in data.get("scores", [])}
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        # فشل تحليل نتيجة إعادة الترتيب - أفضل نرجع المرشحين الأصليين
        # بترتيبهم الخام (بدون تصفية) بدل ما نفقد كل النتائج بالغلط
        return candidates[:top_k]

    scored = [(scores.get(i, 0.0), c) for i, c in enumerate(candidates)]
    scored.sort(key=lambda x: x[0], reverse=True)

    filtered = [c for score, c in scored if score >= min_score]

    if not filtered and scored:
        # لو التصفية استبعدت الكل، نرجع أفضل مرشح وحد على الأقل (حتى لو
        # تحت الحد)، ونخلي فحص الكفاية (مرحلة 4 - حكم LLM أذكى وأدق
        # بالسياق) يقرر نهائياً، بدل ما رقم صارم (min_score) يقفل الباب
        # بدري ويمنع حتى فرصة تقييم أدق. اكتشفنا هاد فعلياً بحالة اختبار
        # حقيقية (سؤال عن "غياب" رجع صفر نتائج بثلاث محاولات متتالية)
        return [scored[0][1]]

    return filtered[:top_k]