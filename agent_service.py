"""
حلقة الـAgentic RAG.

الفرق الجوهري عن llm_service.py القديم: هناك إحنا (الكود) كنا نستدعي
search_documents دايماً بشكل صريح (hard-coded). هون الموديل نفسه بيقرر
شو يستخدم ومتى، بناءً على فهمه للسؤال - وممكن يستخدم أكتر من أداة
بالتسلسل (مثال المهمة: rag_search ثم calculator).
"""

import os
import json
from openai import AzureOpenAI

from tools import TOOLS_SCHEMA, TOOL_IMPLEMENTATIONS

MAX_AGENT_STEPS = 5  # حد أقصى لعدد "الجولات" مع الموديل، يمنع حلقة لا نهائية

AGENT_SYSTEM_PROMPT = """أنت مساعد داخلي للموظفين، عندك أدوات تقدر تستخدمها:
- rag_search: للبحث في مستندات سياسات الشركة
- calculator: لأي عملية حسابية
- create_ticket: لإنشاء تذكرة دعم لو الطلب يحتاج متابعة رسمية

قواعد صارمة وإلزامية:
1. نطاق العمل والكيانات الخارجية (CRITICAL):
   - المستندات تخص **شركتنا فقط**.
   - إذا كان سؤال الموظف يتعلق بكيان خارجي أو شركة أخرى (مثل: Google, Microsoft...)، أو موضوع خارج نطاق عملنا، **يُمنع منعاً باتاً** عرض أو تطبيق سياسات شركتنا الداخلية عليه.
   - في هذه الحالة، أجب فوراً وبشكل مباشر دون إطالة: 
     "لا توجد معلومات كافية في مستندات الشركة الإدارية حول [اسم الموضوع/الشركة المطلوب]".

2. استخدام الأدوات واسترجاع المعلومات:
   - استخدم rag_search دائماً قبل ما تجاوب على أي سؤال يخص سياسات الشركة، ولا تعتمد على معرفتك الخاصة.
   - لو السؤال يحتاج حساب رقمي (مثل: كم إجازة متبقية)، استخدم أولاً rag_search لجلب الأرقام الأساسية، ثم استخدم calculator لحساب النتيجة الفعلية - لا تحسب يدوياً بنفسك.
   - لو طلب المستخدم إجراءً يتعلق بحادثة لها إجراء موثق (مثل فقدان جهاز، إصابة عمل)، استخدم rag_search أولاً للتحقق من الإجراء الرسمي واذكره للمستخدم قبل فتح أي تذكرة.

3. إنشاء التذاكر والتعامل مع أخطاء المدخلات:
   - لا تنشئ تذكرة دعم (create_ticket) بعنوان أو وصف عام/غامض. لو طلب الموظف فتح تذكرة بدون تفاصيل كافية، اسأله عن التفاصيل الناقصة ولا تستدعِ الأداة.
   - لو ما لقيت معلومات كافية بعد البحث، وضح ذلك للمستخدم صراحة بدل اختلاق إجابة.

4. التوثيق والمصادر:
   - اذكر مصدر المعلومة (اسم المستند والقسم) في إجابتك النهائية دائماً عند الاعتماد على المستندات.
"""

def _client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def run_agent(user_message: str, history: list[dict]) -> dict:
    """
    بترجع dict فيه:
    - answer: الجواب النهائي النصي
    - tool_trace: قائمة بكل أداة اتنفذت بالترتيب (اسمها، باراميتراتها، نتيجتها)
      - هاد بالضبط اللي بدنا نعرضه لاحقاً كـ"أمثلة single-tool وmulti-tool"
    """
    client = _client()
    chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    tool_trace = []

    for _ in range(MAX_AGENT_STEPS):
        response = client.chat.completions.create(
            model=chat_deployment,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",  # الموديل حر يقرر يستخدم أداة أو لأ
            temperature=0.1,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            # ما طلب أي أداة هاي الجولة -> هاد الجواب النهائي، نوقف الحلقة
            return {"answer": message.content, "tool_trace": tool_trace}

        # الموديل طلب استخدام أداة وحدة أو أكتر بنفس الجولة
        messages.append(message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            implementation = TOOL_IMPLEMENTATIONS.get(tool_name)
            if implementation is None:
                result = {"success": False, "error": f"أداة غير معروفة: {tool_name}"}
            else:
                try:
                    result = implementation(**tool_args)
                except TypeError as e:
                    # باراميترات ناقصة أو غلط - منرجع الخطأ للموديل نفسه (مش
                    # نفشل السكريبت كامل) عشان يقدر يتعامل معه: يوضح
                    # للمستخدم، أو يجرب يستخرج المعلومة الناقصة بطريقة تانية
                    result = {"success": False, "error": f"باراميترات ناقصة أو غير صحيحة: {e}"}

            tool_trace.append({"tool": tool_name, "arguments": tool_args, "result": result})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    return {
        "answer": "تجاوزت العملية الحد الأقصى من خطوات الأدوات المسموحة، حاول تبسيط سؤالك.",
        "tool_trace": tool_trace,
    }