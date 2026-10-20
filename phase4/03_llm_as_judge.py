"""
เฟส 4 / ขั้น 3 - LLM-as-Judge: ใช้โมเดลเก่ง ๆ ให้คะแนนคำตอบของเรา
รัน:  python phase4/03_llm_as_judge.py

แนวคิด: การประเมินด้วยคนแพงและช้า เลยใช้ LLM ที่เก่งกว่า (เช่น Claude)
มาเป็น "กรรมการ" ให้คะแนนคำตอบตาม rubric ที่เรากำหนด
เป็นวิธีมาตรฐานในการ eval คุณภาพคำตอบแบบ scale ได้

เตรียมก่อนรัน:
    pip install anthropic
    ตั้ง environment variable:  ANTHROPIC_API_KEY=sk-ant-...
"""
import os
import json
import anthropic

JUDGE_MODEL = "claude-haiku-4-5-20251001"   # ถูก+เร็ว เหมาะเป็นกรรมการ

# คู่ (คำถาม, คำตอบจากโมเดลเรา) - ปกติดึงมาจากผลรัน 02_inference.py
samples = [
    {"question": "AI คืออะไร",
     "answer": "AI คือปัญญาประดิษฐ์ ระบบที่ทำให้คอมพิวเตอร์เรียนรู้และตัดสินใจได้"},
    {"question": "RAG ย่อมาจากอะไร",
     "answer": "RAG ย่อมาจาก Retrieval-Augmented Generation"},
]

JUDGE_PROMPT = """คุณเป็นกรรมการประเมินคุณภาพคำตอบ AI
ให้คะแนน 1-5 ตาม rubric นี้:
- ความถูกต้อง (ข้อมูลผิดหรือไม่)
- ความครบถ้วน (ตอบตรงคำถามไหม)
- ความชัดเจน (อ่านเข้าใจง่ายไหม)

คำถาม: {question}
คำตอบที่ต้องประเมิน: {answer}

ตอบกลับเป็น JSON เท่านั้น รูปแบบ:
{{"score": <1-5>, "reason": "<เหตุผลสั้น ๆ>"}}"""

client = anthropic.Anthropic()   # อ่าน key จาก ANTHROPIC_API_KEY

total = 0
for s in samples:
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(**s)}],
    )
    raw = msg.content[0].text.strip()
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        verdict = {"score": "?", "reason": raw}
    total += verdict["score"] if isinstance(verdict.get("score"), int) else 0
    print(f"\nคำถาม: {s['question']}")
    print(f"คำตอบ: {s['answer']}")
    print(f"คะแนน: {verdict['score']}/5 - {verdict['reason']}")

print(f"\n{'='*50}\nคะแนนรวม: {total}/{len(samples)*5}")
print("-> ใช้ตัวเลขนี้เทียบระหว่างโมเดลเวอร์ชันต่าง ๆ ได้")
