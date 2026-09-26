"""
تعريف الأدوات الثلاث المطلوبة للـAgentic RAG.
كل أداة إلها جزئين:
1. الـSchema: وصف JSON نديه للموديل عشان يفهم شو الأداة وشو باراميتراتها
2. التنفيذ: الدالة الفعلية اللي بتشتغل لما الموديل يقرر يستخدم الأداة
"""

import re
import uuid

from search_service import search_documents

# ============================================================
# 1) RAG Search
# ============================================================

def tool_rag_search(query: str) -> dict:
    chunks = search_documents(query, top_k=3)
    if not chunks:
        return {"found": False, "results": []}
    return {
        "found": True,
        "results": [
            {
                "content": c["content"],
                "source_document": c["source_document"],
                "section": c["section"],
            }
            for c in chunks
        ],
    }


# ============================================================
# 2) Calculator
# ============================================================

# نتأكد إنه التعبير يحتوي بس أرقام وعلامات حسابية أساسية، عشان نمنع
# تنفيذ أي كود خبيث لو الموديل "اخترع" تعبير غريب أو حد حاول يستغل الأداة
_SAFE_EXPRESSION = re.compile(r"^[0-9\.\+\-\*/\(\)\s%]+$")


def tool_calculator(expression: str) -> dict:
    if not expression or not _SAFE_EXPRESSION.match(expression):
        return {
            "success": False,
            "error": "تعبير رياضي غير صالح أو يحتوي على رموز غير مسموحة",
        }
    try:
        # namespace فاضي تماماً (لا builtins) عشان eval ما يقدر ينفذ أي شي
        # غير العمليات الحسابية البحتة، حتى لو التعبير نفسه اجتاز الفحص فوق
        result = eval(expression, {"__builtins__": {}}, {})
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": f"خطأ أثناء الحساب: {e}"}


# ============================================================
# 3) Ticket Creation (mock)
# ============================================================

_TICKETS: dict[str, dict] = {}  # تخزين مؤقت بالذاكرة، يكفي لأغراض الـmock


def tool_create_ticket(subject: str, description: str, priority: str = "medium") -> dict:
    if not subject or not description:
        return {
            "success": False,
            "error": "لازم عنوان (subject) ووصف (description) لإنشاء التذكرة",
        }
    ticket_id = f"TCK-{uuid.uuid4().hex[:8].upper()}"
    _TICKETS[ticket_id] = {
        "subject": subject,
        "description": description,
        "priority": priority,
        "status": "open",
    }
    return {"success": True, "ticket_id": ticket_id, "status": "open"}


# ============================================================
# Schemas بصيغة Azure OpenAI function/tool calling
# ============================================================

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "يبحث في قاعدة معرفة مستندات الشركة (سياسات الإجازات، الرواتب، "
                "التأمين، الأمن السيبراني، التطوير التقني) ويرجع أقرب المعلومات "
                "المرتبطة بالسؤال مع مصدرها. استخدمها دائماً قبل الإجابة على أي "
                "سؤال يخص سياسات الشركة."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "سؤال أو استعلام البحث بلغة طبيعية",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "ينفذ عملية حسابية رياضية (جمع، طرح، ضرب، قسمة). استخدمها لأي "
                "سؤال يحتاج حساب أرقام، مثل حساب أيام إجازة متبقية أو مبالغ مالية."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "تعبير رياضي، مثال: '25 - 8' أو '(1500 * 1.5)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": (
                "ينشئ تذكرة دعم فني/إداري وهمية عندما يطلب المستخدم متابعة "
                "مشكلة أو طلب رسمي لا تكفي المعلومات المتاحة للإجابة عليه مباشرة."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "عنوان مختصر للمشكلة أو الطلب",
                    },
                    "description": {
                        "type": "string",
                        "description": "وصف تفصيلي للمشكلة أو الطلب",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "أولوية التذكرة",
                    },
                },
                "required": ["subject", "description"],
            },
        },
    },
]

TOOL_IMPLEMENTATIONS = {
    "rag_search": tool_rag_search,
    "calculator": tool_calculator,
    "create_ticket": tool_create_ticket,
}
