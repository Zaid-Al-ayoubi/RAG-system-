"""
Flask API: يستقبل سؤال المستخدم، يسترجع السياق، يولّد الجواب،
ويحافظ على ذاكرة المحادثة لكل جلسة مستخدم على حدة.
"""

import uuid
from dotenv import load_dotenv

load_dotenv()  # لازم قبل استيراد search_service, llm_service, agent_service بالأسفل

from flask import Flask, request, jsonify, render_template

from agent_service import run_agent

app = Flask(__name__)

# ذاكرة المحادثة: قاموس بالذاكرة {session_id: [{"role":.., "content":..}, ...]}
# ملاحظة مهمة: هاد تخزين مؤقت (in-memory) بيضيع لو أعدت تشغيل السيرفر
# أو لو شغّلت أكتر من instance بنفس الوقت. لمشروع إنتاجي حقيقي
# لازم يكون بـ Redis أو قاعدة بيانات، بس لغرض هاد التسليم كافي تماماً.
conversation_store: dict[str, list[dict]] = {}

# كم "دور محادثة" (سؤال + جواب) نحتفظ فيهم كحد أقصى بالذاكرة.
# ليش هاد الحد مهم: كل ما كبرت الذاكرة، كل ما زاد حجم الـprompt المرسل
# للموديل مع كل رسالة جديدة -> تكلفة أعلى + خطر "lost in the middle"
# (الموديل يهمل معلومات بنص محادثة طويلة كتير)
MAX_HISTORY_TURNS = 6


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not user_message:
        return jsonify({"error": "الرسالة فاضية"}), 400

    # نجيب تاريخ المحادثة السابق لهاي الجلسة تحديداً (فاضي لو جلسة جديدة)
    history = conversation_store.get(session_id, [])

    # الـagent بيقرر لحاله شو يستخدم من الأدوات (بحث، حاسبة، تذكرة)
    # وممكن يستخدم أكتر من وحدة بالتسلسل قبل ما يرجع الجواب النهائي
    result = run_agent(user_message, history)
    answer = result["answer"]
    tool_trace = result["tool_trace"]

    # نحدّث الذاكرة بعد ما حصلنا على الجواب
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})

    # نقص الذاكرة لآخر MAX_HISTORY_TURNS دور بس (كل دور = رسالتين)
    conversation_store[session_id] = history[-MAX_HISTORY_TURNS * 2:]

    # نستخرج مصادر RAG (لو استُخدمت) من tool_trace لعرضها بالواجهة زي القديم
    sources = []
    for call in tool_trace:
        if call["tool"] == "rag_search" and call["result"].get("found"):
            for r in call["result"]["results"]:
                sources.append({"source_document": r["source_document"], "section": r["section"]})

    return jsonify(
        {
            "session_id": session_id,
            "answer": answer,
            "sources": sources,
            "tool_trace": tool_trace,  # مفيدة لعرض التسلسل الكامل (بحث ثم حساب مثلاً)
        }
    )


if __name__ == "__main__":
    # use_reloader=False: بيمنع Flask من مراقبة ملفات venv وإعادة تشغيل
    # نفسه تلقائياً. لاحظنا إنه تنفيذ بعض ملفات مكتبة azure.search (مثل
    # paging.py) كان بيخلي المراقبة تظن إنه الكود تغيّر، فيعيد التشغيل
    # بمنتصف معالجة الطلب ويقطع الاتصال بالمتصفح فجأة.
    # debug=True يضل مفعّل عشان لسا نستفيد من رسائل الخطأ التفصيلية.
    app.run(debug=True, port=5000, use_reloader=False)