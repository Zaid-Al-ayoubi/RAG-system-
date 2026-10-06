"""
image_agent.py
وكيل فهم صور + OCR. كل صورة ترفعها تاخد رقم ثابت (ID)، وتقدر ترجعلها
بالاسم بأي وقت بالمحادثة - مش لازم تكون "آخر صورة" المرفوعة.
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()

from agent_service import run_agent

MAX_HISTORY_TURNS = 6
VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")


def clean_answer(text: str) -> str:
    text = re.sub(r"[*_`#>]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def looks_like_path(text: str) -> bool:
    t = text.strip().strip('"')
    return t.lower().endswith(VALID_EXT)


def load_image(path: str, images: dict, current_id: list) -> int | None:
    path = path.strip().strip('"')
    if not os.path.isfile(path):
        print(f"   ⚠️ الملف غير موجود: {path}")
        return None
    if not path.lower().endswith(VALID_EXT):
        print(f"   ⚠️ الامتداد غير مدعوم ({', '.join(VALID_EXT)})")
        return None

    new_id = len(images) + 1
    images[new_id] = path
    current_id[0] = new_id
    print(f"✅ صورة رقم {new_id}: {os.path.basename(path)}")
    return new_id


def main():
    print("وكيل فهم الصور + OCR.")
    print("ارفع صورة بكتابة مسارها. للرجوع لصورة سابقة: 'صورة <رقم>'. Ctrl+C للخروج.\n")

    images: dict[int, str] = {}       # {1: "path1.jpg", 2: "path2.png", ...}
    current_id = [None]               # list عشان نقدر نعدلها جوا load_image
    history: list[dict] = []

    # أول صورة إلزامية قبل ما نبلش
    while current_id[0] is None:
        path = input("مسار أول صورة: ")
        load_image(path, images, current_id)

    try:
        while True:
            question = input("\nسؤالك: ").strip()
            if not question:
                continue

            # رفع صورة جديدة مباشرة بلصق مسار
            if looks_like_path(question):
                load_image(question, images, current_id)
                continue

            # الرجوع الصريح لصورة سابقة: "صورة 2" أو "image 2"
            match = re.match(r"^(?:صورة|image)\s+(\d+)$", question.strip(), re.IGNORECASE)
            if match:
                target_id = int(match.group(1))
                if target_id in images:
                    current_id[0] = target_id
                    print(f"↩️  رجعنا لصورة رقم {target_id}: {os.path.basename(images[target_id])}")
                else:
                    print(f"   ⚠️ ما في صورة بهالرقم. الصور المتاحة: {list(images.keys())}")
                continue

            # سؤال عادي عن الصورة الحالية (current_id)
            active_path = images[current_id[0]]
            hinted = (
                f"{question}\n\n"
                f"(مسار الصورة رقم {current_id[0]} المطلوب تحليلها الآن: "
                f"{active_path}. استخدم هذا المسار بالتحديد مع الأداة "
                f"المناسبة - لا تعتمد على أي نص أو وصف سابق بالمحادثة.)"
            )

            out = run_agent(hinted, history)
            answer = clean_answer(out["answer"])
            print(f"🤖 [صورة {current_id[0]}] {answer}")
            print(f"   🔧 أدوات مستخدمة: {[t['tool'] for t in out['tool_trace']]}")

            history.append({"role": "user", "content": f"[صورة {current_id[0]}] {question}"})
            history.append({"role": "assistant", "content": answer})
            history = history[-MAX_HISTORY_TURNS * 2:]

    except KeyboardInterrupt:
        print("\nخلص.")


if __name__ == "__main__":
    main()