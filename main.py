from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import numpy as np

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return np.array(response['data'][0]['embedding'])

facilities = [
    {"id": "Kalinaw MindCenter", "description": "A Mental Health Clinic in Cebu that provides Psychiatric Consultations, Psychotherapy & Counselling.", "embedding": get_embedding("A Mental Health Clinic in Cebu that provides Psychiatric Consultations, Psychotherapy & Counselling.")},
    {"id": "RBE Psychological Services", "description": "RBEPS offers holistic services to individuals, couples, and families to overcome mental health issues.", "embedding": get_embedding("RBEPS offers holistic services to individuals, couples, and families to overcome mental health issues.")}
]

@app.route("/recommend", methods=["POST"])
def recommend_with_interaction():
    data = request.get_json()
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    try:
        completion = openai.ChatCompletion.create(
            model="ft:gpt-4o-mini-2024-07-18:personal::ANJY20nd",
            messages=[{"role": "user", "content": message}]
        )
        user_response = completion.choices[0].message["content"]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    query_embedding = get_embedding(message)

    similarities = [
        (facility, np.dot(query_embedding, facility["embedding"].flatten()))
        for facility in facilities
    ]

    best_match = max(similarities, key=lambda x: x[1])[0]

    response_message = f"{user_response}\n\nRecommended Facility: {best_match['id']}\nDescription: {best_match['description']}"

    return jsonify({"response": response_message})

if __name__ == "__main__":
    app.run(debug=True)
