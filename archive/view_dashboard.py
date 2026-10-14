"""
يفتح لوحة TruLens التفاعلية بالمتصفح، بالاعتماد على البيانات المحفوظة
مسبقاً بـ default.sqlite (من تشغيل evaluate_rag_triad.py قبل هيك).
ما في داعي تعيد تشغيل الـpipeline كامل من جديد.
"""

from trulens.core import TruSession
from trulens.dashboard.run import run_dashboard

session = TruSession()
run_dashboard(session)

# بعد ما تشغل هاد الملف، السيرفر بيضل شغال بالتيرمنال (ما بيرجعلك الموجه).
# افتح المتصفح يدوياً على الرابط يلي بيطبعه بالتيرمنال، عادة:
# http://localhost:8501
# لو ما انفتح تلقائياً، انسخ الرابط والصقه بالمتصفح.
# لإيقاف السيرفر: Ctrl+C بالتيرمنال.