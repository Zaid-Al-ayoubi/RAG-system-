"""
تحقق سريع: هل deployment الحالي (gpt-5.4-mini) بيدعم صور؟
"""
import os
import base64
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)

# أي صورة صغيرة عندك محلياً للتجربة
with open("Screenshot 2026-09-20 182213.png", "rb") as f:
    b64_image = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "شو تشوف بهالصورة؟ صفها بجملتين."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                },
            ],
        }
    ],
    max_completion_tokens=200,
)

print(response.choices[0].message.content)