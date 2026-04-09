import base64, cv2
from openai import OpenAI

client = OpenAI(
    api_key="sk-e653c57fb2de45ed973d565640b08a92",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

img = cv2.imread("/home/ypf/ReKep/test_output/rgb.png")
img_b64 = base64.b64encode(cv2.imencode('.png', img)[1]).decode()

response = client.chat.completions.create(
    model="qwen3-vl-flash-2026-01-22",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe what you see in one sentence."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]
    }],
    temperature=0.0,
    max_tokens=100,
)
print(response.choices[0].message.content)