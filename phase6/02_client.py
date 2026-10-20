"""
เฟส 6 / ขั้น 2 - Client เรียก API (ต้องเปิด server จาก 01 ไว้ก่อน)
เตรียม:  pip install requests
รัน:     python phase6/02_client.py

สาธิตการรับคำตอบแบบ streaming ทีละ token
"""
import requests

URL = "http://127.0.0.1:8000/chat"

question = "อธิบาย machine learning สั้น ๆ"
print(f"ถาม: {question}\nตอบ: ", end="", flush=True)

# stream=True เพื่อรับ token ทยอยมา
with requests.post(URL, json={"message": question}, stream=True) as r:
    for chunk in r.iter_content(chunk_size=None):
        if chunk:
            print(chunk.decode("utf-8"), end="", flush=True)
print()
