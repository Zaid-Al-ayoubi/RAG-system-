"""
خطوة 3: صوت -> Agent (RAG + أدوات) -> صوت، مع ذاكرة محادثة.
"""
import os
import re
from dotenv import load_dotenv

load_dotenv()  # لازم قبل استيراد agent_service (search_service بيقرأ USE_MOCK وقت الاستيراد)

import azure.cognitiveservices.speech as speechsdk
from agent_service import run_agent

KEY = os.environ["AZURE_SPEECH_KEY"]
REGION = os.environ["AZURE_SPEECH_REGION"]
MAX_HISTORY_TURNS = 6

VOICES = {"ar": "ar-JO-TaimNeural", "en": "en-US-JennyNeural"}

LANG_HINT = {
    "ar": "أجب باللغة العربية.",
    "en": "Answer in English.",
}
VOICE_STYLE = (
    " هذه محادثة صوتية: أجب بجملتين إلى أربع جمل كحد أقصى، بنص عادي بدون "
    "رموز أو نقاط أو تنسيق، ولا تقرأ أسماء الملفات. "
)

speech_config = speechsdk.SpeechConfig(subscription=KEY, region=REGION)
auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
    languages=["ar-JO", "en-US"]
)
recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config,
    auto_detect_source_language_config=auto_detect,
    audio_config=speechsdk.audio.AudioConfig(use_default_microphone=True),
)


def clean_for_speech(text: str) -> str:
    # شبكة أمان: لو الموديل رجع markdown رغم التعليمات
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S+\.docx", "", text)
    return re.sub(r"\s+", " ", text).strip()


def speak(text: str, lang: str):
    cfg = speechsdk.SpeechConfig(subscription=KEY, region=REGION)
    cfg.speech_synthesis_voice_name = VOICES[lang]
    synth = speechsdk.SpeechSynthesizer(
        speech_config=cfg,
        audio_config=speechsdk.audio.AudioOutputConfig(use_default_speaker=True),
    )
    synth.speak_text_async(text).get()


history: list[dict] = []
print("جاهز. احكي (Ctrl+C للخروج)")

try:
    while True:
        print("\n🎤 استمع...")
        result = recognizer.recognize_once_async().get()

        if result.reason != speechsdk.ResultReason.RecognizedSpeech:
            continue

        detected = speechsdk.AutoDetectSourceLanguageResult(result).language or "en-US"
        lang = "ar" if detected.startswith("ar") else "en"
        question = result.text
        print(f"[{detected}] أنت: {question}")

        # التعليمة بتنحط بس على النسخة المرسلة للموديل
        hinted = f"{question}\n\n({LANG_HINT[lang]}{VOICE_STYLE})"
        out = run_agent(hinted, history)
        answer = clean_for_speech(out["answer"])
        print(f"🤖 الوكيل: {answer}")
        print(f"   أدوات: {[t['tool'] for t in out['tool_trace']]}")

        # نخزن السؤال الأصلي (بدون التعليمة) عشان ما تتراكم التعليمات بالتاريخ
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        history = history[-MAX_HISTORY_TURNS * 2:]

        speak(answer, lang)
except KeyboardInterrupt:
    print("\nخلص.")