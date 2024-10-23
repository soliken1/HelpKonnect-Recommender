# main.py
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
            model="",
            messages=[
                {"role": "user", "content": user_request.message}
            ]
        )
        response_message = completion.choices[0].message['content']
        return {"response": response_message}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
