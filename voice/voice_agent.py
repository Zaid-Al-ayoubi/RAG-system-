"""
voice_barge_in.py
صوت -> Agent -> صوت مع دعم مقاطعة (barge-in).
"""

import os
import re
import time
import queue
from dotenv import load_dotenv

load_dotenv()

import azure.cognitiveservices.speech as speechsdk
from agent_service import run_agent

KEY = os.environ["AZURE_SPEECH_KEY"]
REGION = os.environ["AZURE_SPEECH_REGION"]

VOICES = {"ar": "ar-JO-TaimNeural", "en": "en-US-JennyNeural"}
LANG_HINT = {"ar": "أجب باللغة العربية.", "en": "Answer in English."}
VOICE_STYLE = (
    " هذه محادثة صوتية: أجب بجملتين إلى أربع جمل كحد أقصى، بنص طبيعي متصل "
    "بدون رموز أو تنسيق. لا تذكر اسم الملف، ولا تستخدم كلمة 'المصدر' أو "
    "'Source' كعبارة استشهاد منفصلة في نهاية الجواب — إذا احتجت تشير لمصدر "
    "المعلومة، اذكر اسم القسم بشكل طبيعي ضمن الجملة نفسها."
)


class VoiceSession:
    def __init__(self):
        self.speech_config = speechsdk.SpeechConfig(subscription=KEY, region=REGION)
        self.auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=["ar-JO", "en-US"]
        )
        self.history: list[dict] = []
        self.max_history_turns = 6

        self.event_queue: queue.Queue = queue.Queue()

        self.current_synthesizer: speechsdk.SpeechSynthesizer | None = None
        self.is_speaking = False
        self._speak_started_at = 0.0
        self._speak_ended_at = 0.0
        self._barge_in_grace_seconds = 1.0

        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous"
        )

        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "2000"
        )

        self.recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            auto_detect_source_language_config=self.auto_detect,
            audio_config=speechsdk.audio.AudioConfig(use_default_microphone=True),
        )
        self.recognizer.recognizing.connect(self._on_recognizing)
        self.recognizer.recognized.connect(self._on_recognized)

    # ---------- STT callbacks ----------
    def _on_recognizing(self, evt):
        """partial result — أثناء ما المستخدم عم يحكي."""
        text = evt.result.text.strip()
        if not self.is_speaking or not text:
            return

        # فلتر طول أدنى: كلمة وحدة ممكن تكون ضجيج/بداية صدى
        if len(text.split()) < 2:
            return

        # فلتر صدى: قارن مع الجواب الحالي المُشغَّل
        if self._looks_like_echo(text):
            return

        print(f"\n🛑 [barge-in] كشفت كلام جزئي: '{text}'")
        self._trigger_barge_in()

    def _looks_like_echo(self, text: str) -> bool:
        if not self.history or self.history[-1]["role"] != "assistant":
            return False
        clean_new = re.sub(r"[^\w\s]", "", text.lower()).split()
        clean_last = re.sub(r"[^\w\s]", "", self.history[-1]["content"].lower()).split()
        if not clean_new:
            return False
        overlap = sum(1 for w in clean_new if w in clean_last)
        return (overlap / len(clean_new)) > 0.6

    def _on_recognized(self, evt):
        """final result — المستخدم خلّص جملة."""
        if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return

        text = evt.result.text.strip()

        # فلتر 1: نص فاضي أو قصير جداً
        if len(text.split()) < 2:
            print(f"   (تجاهلنا: '{text}' — سؤال قصير جداً)")
            return

        # فلتر 2: cooldown — تجاهل أي كلام بعد أقل من 1.0s من نهاية TTS
        if self._speak_ended_at > 0:
            elapsed = time.time() - self._speak_ended_at
            if elapsed < 1.0:
                print(f"   (تجاهلنا: '{text[:50]}' — cooldown)")
                return

        # فلتر 3: مقارنة مع آخر جواب للوكيل (نفس دالة الصدى الموحّدة)
        if self._looks_like_echo(text):
            print(f"   (تجاهلنا: '{text[:50]}' — يشبه كلام الوكيل)")
            return

        detected = (
            speechsdk.AutoDetectSourceLanguageResult(evt.result).language or "en-US"
        )
        lang = "ar" if detected.startswith("ar") else "en"
        self.event_queue.put({"type": "question", "text": text, "lang": lang})

    # ---------- Barge-in ----------
    def _trigger_barge_in(self):
        if self.current_synthesizer is None or not self.is_speaking:
            return

        elapsed = time.time() - self._speak_started_at
        if elapsed < self._barge_in_grace_seconds:
            return

        self.current_synthesizer.stop_speaking_async()
        self.is_speaking = False
        print(f"\n🛑 [barge-in] المستخدم قاطع — وقفنا الصوت (بعد {elapsed:.1f}s)")

    # ---------- TTS ----------
    def speak(self, text: str, lang: str):
        cfg = speechsdk.SpeechConfig(subscription=KEY, region=REGION)
        cfg.speech_synthesis_voice_name = VOICES[lang]
        synth = speechsdk.SpeechSynthesizer(
            speech_config=cfg,
            audio_config=speechsdk.audio.AudioOutputConfig(use_default_speaker=True),
        )
        self.current_synthesizer = synth
        self.is_speaking = True
        self._speak_started_at = time.time()
        try:
            synth.speak_text_async(text).get()
        finally:
            self.is_speaking = False
            self.current_synthesizer = None
            self._speak_ended_at = time.time()

    # ---------- Main loop ----------
    def run(self):
        self.recognizer.start_continuous_recognition_async()
        print("جاهز. احكي (Ctrl+C للخروج)")

        try:
            while True:
                try:
                    event = self.event_queue.get(timeout=0.5)
                except queue.Empty:
                    continue  # ما في حدث جديد لسا — رجّع افحص Ctrl+C وكمل

                if event["type"] != "question":
                    continue

                question = event["text"]
                lang = event["lang"]
                print(f"\n[{lang}] أنت: {question}")

                hinted = f"{question}\n\n({LANG_HINT[lang]}{VOICE_STYLE})"
                out = run_agent(hinted, self.history)
                answer = self._clean_for_speech(out["answer"])
                print(f"🤖 الوكيل: {answer}")
                print(f"   🔧 أدوات مستخدمة: {[t['tool'] for t in out['tool_trace']]}")

                self.history.append({"role": "user", "content": question})
                self.history.append({"role": "assistant", "content": answer})
                self.history = self.history[-self.max_history_turns * 2 :]

                self.speak(answer, lang)
        except KeyboardInterrupt:
            print("\nخلص.")
        finally:
            self.recognizer.stop_continuous_recognition_async()

    @staticmethod
    def _clean_for_speech(text: str) -> str:
        text = re.sub(r"[*_`#>]+", "", text)
        text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\S+\.docx,?\s*", "", text)
        return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    VoiceSession().run()
