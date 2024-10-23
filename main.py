from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="",
  messages=[
    {"role": "user", "content": "Recommend me a facility that may assist my mental problems"}
  ]
)
print(completion.choices[0].message)

