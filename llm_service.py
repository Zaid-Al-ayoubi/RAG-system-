"""
طبقة التوليد (generation). نفس فكرة search_service.py: mock و real.
هون كمان مكان الـsystem prompt المسؤول عن الـgrounding (منع الهلوسة).
"""

import os

USE_MOCK = os.environ.get("USE_MOCK_SERVICES", "true").lower() == "true"

# النص هون هو خط الدفاع الأساسي ضد الهلوسة (hallucination):
# بنقول للموديل صراحة يجاوب من السياق المرفق بس، وشو يعمل لو ما لقى الجواب
SYSTEM_PROMPT = """أنت مساعد داخلي للموظفين، تجاوب فقط بناءً على المعلومات المرفقة لك من مستندات الشركة.

قواعد صارمة يجب الالتزام بها دائماً:
- جاوب فقط من المعلومات الموجودة في السياق (Context) المرفق أدناه، ولا تستخدم أي معرفة خارجية عنك.
- إذا لم تجد إجابة كافية في السياق المرفق، قل بوضوح: "لا توجد معلومات كافية في المستندات المتاحة للإجابة على هذا السؤال."
- عند الإجابة، اذكر اسم المستند أو القسم المصدر إذا كان ذلك مفيداً للمستخدم.
- كن مختصراً ومباشراً، ولا تخترع أي تفاصيل أو أرقام غير موجودة حرفياً في السياق."""


def _build_context_string(context_chunks: list[dict]) -> str:
    # بنحول قائمة الـchunks لنص واحد منسق، كل قطعة معلّمة بمصدرها
    # عشان الموديل يقدر يذكر المصدر إذا احتاج
    parts = [
        f"[من: {c['source_document']} - {c['section']}]\n{c['content']}"
        for c in context_chunks
    ]
    return "\n\n---\n\n".join(parts)


def _mock_generate(query: str, context_chunks: list[dict], history: list[dict]) -> str:
    if not context_chunks:
        return "لا توجد معلومات كافية في المستندات المتاحة للإجابة على هذا السؤال. (mock)"
    top = context_chunks[0]
    return (
        f"(mock) بناءً على '{top['source_document']} - {top['section']}':\n"
        f"{top['content'][:250]}..."
    )


def _real_generate(query: str, context_chunks: list[dict], history: list[dict]) -> str:
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )

    context_str = _build_context_string(context_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)  # تاريخ المحادثة السابق = "الذاكرة"
    messages.append(
        {
            "role": "user",
            "content": f"السياق المتاح:\n{context_str}\n\nسؤال المستخدم: {query}",
        }
    )

    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        messages=messages,
        temperature=0.2,  # منخفضة عمداً: بدنا دقة والتزام بالمصدر، مش إبداع
    )
    return response.choices[0].message.content


def generate_answer(query: str, context_chunks: list[dict], history: list[dict]) -> str:
    if USE_MOCK:
        return _mock_generate(query, context_chunks, history)
    return _real_generate(query, context_chunks, history)