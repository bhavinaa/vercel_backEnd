from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class ChatMessages(BaseModel):
    messages: list[str]

@app.post("/summarize")
async def summarize_chat(chat_messages: ChatMessages):
    try:
        text = " ".join(chat_messages.messages)
        if not text.strip():
            raise ValueError("No valid text provided for summarization.")
        summary = summarizer(text, max_length=50, min_length=5, do_sample=False)
        return {"summary": summary[0]["summary_text"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
