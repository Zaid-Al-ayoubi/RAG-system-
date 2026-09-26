"""
يقرأ ملفات Word، يقسمها لقطع نصية (chunks) مع الحفاظ على السياق،
ويخرج ملف JSON جاهز للرفع على Azure AI Search.
"""

import os                  # للتعامل مع مسارات الملفات (join, basename..)
import json                 # عشان نكتب النتيجة النهائية بصيغة JSON
import glob                 # عشان نلاقي كل ملفات .docx بمجلد معين تلقائياً
from docx import Document   # من مكتبة python-docx، عشان نفتح ونقرأ ملفات Word
import tiktoken              # مكتبة OpenAI لحساب عدد التوكنز الفعلي لأي نص

# بننشئ الـ encoder مرة وحدة بس بأول الملف، مش جوا كل دالة،
# عشان ما نعيد تحميله من جديد كل مرة (تحميله فيه تكلفة وقت بسيطة)
encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    # بنحول النص لتوكنز (أرقام) وبنرجع طولهم
    # هاد أدق بكتير من len(text) لأنه توكن الواحد مش نفس حرف الواحد
    return len(encoding.encode(text))


def extract_paragraphs_with_sections(docx_path: str) -> list[dict]:
    doc = Document(docx_path)          # يفتح ملف الوورد ويحمله بالذاكرة
    paragraphs = []                     # هون رح نجمع كل فقرة صالحة
    current_section = "مقدمة"           # قيمة افتراضية لحد ما نلاقي أول عنوان

    for para in doc.paragraphs:         # بيمشي على كل فقرة بالملف بالترتيب
        text = para.text.strip()        # ياخد نص الفقرة ويشيل المسافات الزايدة
        if not text:                    # لو الفقرة فاضية (سطر فاضي مثلاً)
            continue                    # تجاهلها وكمل للي بعدها

        if para.style.name.startswith("Heading"):
            # لو نمط الفقرة "Heading" (يعني عنوان)، حدّث القسم الحالي
            current_section = text
            continue                    # العنوان نفسه ما بنضيفه كفقرة محتوى

        # فقرة محتوى عادية: نضيفها مع اسم القسم اللي هي تحته
        paragraphs.append({"text": text, "section": current_section})

    return paragraphs


def chunk_paragraphs(
    paragraphs: list[dict],
    source_document: str,
    max_tokens: int = 600,      # الحد الأقصى لحجم كل chunk بالتوكنز
    overlap_tokens: int = 80,   # قد إيش نرجع للخلف كـ"تداخل" بين القطع
) -> list[dict]:
    chunks = []                        # القائمة النهائية للقطع الجاهزة
    current_chunk_paras: list[dict] = []  # الفقرات المتجمعة بالـchunk الحالي
    current_tokens = 0                  # مجموع توكنز الـchunk الحالي لحد هلق
    chunk_index = 0                     # رقم تسلسلي لكل chunk بنفس المستند

    def flush_chunk(paras: list[dict]):
        # هاي دالة داخلية بتاخد مجموعة فقرات وتحولها لـchunk نهائي وتضيفه للقائمة
        nonlocal chunk_index            # عشان نقدر نعدل chunk_index من برا نطاقها
        if not paras:                   # لو ما في فقرات، ما في شي نعمله
            return
        content = "\n".join(p["text"] for p in paras)  # يدمج نصوص الفقرات بسطر لكل وحدة
        # نجمع كل الأقسام الفريدة يلي هاد الـchunk بيغطيها (مش بس أول وحدة)
        # عشان لو chunk واحد ضم أكتر من قسم، ما نخفي هاي المعلومة عن المستخدم
        unique_sections = list(dict.fromkeys(p["section"] for p in paras))
        chunks.append(
            {
                "chunk_id": f"{source_document}_{chunk_index}",  # معرف فريد لكل قطعة
                "content": content,
                "source_document": source_document,   # لنعرف من أي ملف جاي هاد الchunk
                "section": unique_sections[0],         # القسم الرئيسي (للعرض المختصر)
                "sections_covered": unique_sections,    # كل الأقسام الفعلية جوا هاد الـchunk
            }
        )
        chunk_index += 1                # نزيد الرقم التسلسلي للـchunk الجاي

    for para in paragraphs:             # بيمشي على كل فقرة بالترتيب اللي طلعت فيه
        para_tokens = count_tokens(para["text"])  # كم توكن بهاي الفقرة تحديداً

        # الشرط الأساسي: لو إضافة الفقرة الحالية رح تتجاوز الحد الأقصى
        # وعندنا أصلاً فقرات متجمعة (مش chunk فاضي)، لازم نسكر الـchunk الحالي
        if current_tokens + para_tokens > max_tokens and current_chunk_paras:
            flush_chunk(current_chunk_paras)   # نحفظ الـchunk اللي خلصناه

            overlap_paras = []          # هون رح نبني تداخل بسيط مع القطعة الجاية
            overlap_count = 0
            for p in reversed(current_chunk_paras):
                # بنمشي من آخر فقرة للأول (معكوس) عشان ناخد "آخر شوية كلام"
                t = count_tokens(p["text"])
                if overlap_count + t > overlap_tokens:
                    break                # لو زدنا عن الحد المسموح للتداخل، وقف
                overlap_paras.insert(0, p)  # نرجعها لترتيبها الطبيعي (مش معكوس)
                overlap_count += t

            current_chunk_paras = overlap_paras  # الـchunk الجديد بيبلش بهاي الفقرات
            current_tokens = overlap_count        # وتوكنزه الحالية = توكنز التداخل

        current_chunk_paras.append(para)  # نضيف الفقرة الحالية للـchunk الشغال عليه
        current_tokens += para_tokens     # ونحدث عداد التوكنز

    flush_chunk(current_chunk_paras)  # مهم جداً: نحفظ آخر chunk متبقي بعد ما تخلص الحلقة
    return chunks


def process_all_documents(input_dir: str, output_path: str):
    all_chunks = []                     # هون رح نجمع قطع كل الملفات الخمسة سوا
    docx_files = glob.glob(os.path.join(input_dir, "*.docx"))
    # glob بيرجع لستة بكل مسارات الملفات يلي بتنتهي بـ.docx جوا input_dir

    if not docx_files:                  # لو ما لقى ولا ملف
        print(f"ما لقيت ملفات .docx بمجلد: {input_dir}")
        return                          # نوقف الدالة هون، ما في داعي نكمل

    for docx_path in docx_files:        # لكل ملف وورد لقيناه
        source_document = os.path.basename(docx_path)  # ياخد بس اسم الملف بدون المسار الكامل
        paragraphs = extract_paragraphs_with_sections(docx_path)  # يستخرج فقراته
        chunks = chunk_paragraphs(paragraphs, source_document)     # يقسمها لقطع
        all_chunks.extend(chunks)       # يضيفها على القائمة الكلية
        print(f"{source_document}: {len(chunks)} chunk")  # تقرير سريع بالتيرمنال

    with open(output_path, "w", encoding="utf-8") as f:
        # encoding="utf-8" ضروري هون عشان النص العربي ينكتب صح بالملف
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        # ensure_ascii=False يمنع تحويل الحروف العربية لأكواد \uXXXX غير مقروءة
        # indent=2 بس عشان الملف يطلع مقروء ومنسق لو فتحته يدوياً

    print(f"\nمجموع القطع: {len(all_chunks)} -> اتحفظت بـ {output_path}")


if __name__ == "__main__":
    # هاد الشرط بيضمن إنه هاد الكود يشتغل بس لما تشغل الملف مباشرة
    # مش لو استوردته كموديول بملف تاني
    process_all_documents(input_dir="./sample_docs_large", output_path="chunks.json")