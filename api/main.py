from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI()

class UserRequest(BaseModel):
    message: str

@app.post("/recommend")
async def recommend_facility(user_request: UserRequest):
    try:
        completion = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal::ANHKfbn9",
            messages=[
                {"role": "user", "content": user_request.message}
            ]
        )
        response_message = completion.choices[0].message.content
        return {"response": response_message}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Vercel requires this callable
def handler(event, context):
    return app(event, context)
