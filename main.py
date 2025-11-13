from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Placeholder functions - replace these with your real MCQ logic
def generate_question(text):
    return "Sample question?"

def extract_answers(text):
    return ["Sample answer"]

def generate_distractors(answer):
    return ["OptionA", "OptionB", "OptionC"]

class MCQ(BaseModel):
    question: str
    correct_answer: str
    distractors: List[str]

class MCQRequest(BaseModel):
    text: str
    num_questions: int = 1

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development; change on deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate_mcqs")
def generate_mcqs(request: MCQRequest):
    text = request.text
    num = request.num_questions

    mcqs = []
    for _ in range(num):
        q = generate_question(text)
        answers = extract_answers(text)
        ans = answers[0] if answers else "N/A"
        dists = generate_distractors(ans)
        mcqs.append(MCQ(question=q, correct_answer=ans, distractors=dists))
    return {"mcqs": mcqs}
