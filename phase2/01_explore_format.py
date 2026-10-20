"""
เฟส 2 / ขั้น 1 - เข้าใจ format ของข้อมูล fine-tune
รัน:  python phase2/01_explore_format.py

แนวคิด: เราเทรนแบบ SFT (Supervised Fine-Tuning) ด้วยรูปแบบ "บทสนทนา"
แต่ละตัวอย่างมี messages = [user, assistant]
tokenizer.apply_chat_template จะแปลงเป็นข้อความจริงที่โมเดลเห็นตอนเทรน
"""
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# ตัวอย่างข้อมูล 1 แถว (ไทย + อังกฤษ)
samples = [
    {
        "messages": [
            {"role": "user", "content": "เมืองหลวงของไทยคือที่ไหน"},
            {"role": "assistant", "content": "เมืองหลวงของประเทศไทยคือกรุงเทพมหานครครับ"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Translate 'good morning' to Thai"},
            {"role": "assistant", "content": "'good morning' แปลว่า 'สวัสดีตอนเช้า'"},
        ]
    },
]

for i, s in enumerate(samples, 1):
    print(f"\n{'='*60}\nตัวอย่างที่ {i}")
    print("messages (สิ่งที่เราเขียน):")
    for m in s["messages"]:
        print(f"   [{m['role']}] {m['content']}")

    # แปลงเป็นข้อความจริงที่โมเดลเห็นตอนเทรน (มี special token <|im_start|> ฯลฯ)
    text = tokenizer.apply_chat_template(s["messages"], tokenize=False)
    print("\nหลัง apply_chat_template (สิ่งที่โมเดลเห็นจริง):")
    print(text)

    n_tokens = len(tokenizer.apply_chat_template(s["messages"]))
    print(f"จำนวน token ทั้งตัวอย่าง: {n_tokens}")
