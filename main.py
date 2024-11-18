from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import numpy as np
import requests

url = "https://helpkonnect.vercel.app/api/fetchFacilities"
user_preference = ""
response = requests.get(url)
data = response.json()

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_embedding(text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return np.array(response['data'][0]['embedding'])


# Fetching facilities data and formatting it to match the required structure
facilities = [
    {
        "id": facility["facilityName"],
        "description": facility["facilityDescription"],
        "expertise": facility["facilityExpertise"],
        "embedding": get_embedding(facility["facilityDescription"] + " " + facility["facilityExpertise"])
    }
    for facility in data['fetchFacility']
]


def is_recommendation_request(message):
    keywords = ["recommend", "suggest", "facility", "help", "support"]
    return any(keyword in message.lower() for keyword in keywords)


@app.route("/preference", methods=["POST"])
def analyze_user_preference():
    return jsonify({"response": "in progress"})


@app.route("/recommend", methods=["POST"])
def recommend_with_interaction():
    data = request.get_json()
    user_id = data.get("userId", "")
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

    if is_recommendation_request(message):
        query_embedding = get_embedding(message)

        similarities = [
            (facility, np.dot(query_embedding, facility["embedding"].flatten()))
            for facility in facilities
        ]

        best_match = max(similarities, key=lambda x: x[1])[0]

        response_message = f"{user_response}\n\nRecommended Facility: {best_match['id']}\nDescription: {best_match['description']}\nExpertise: {best_match['expertise']}"
    else:
        response_message = user_response

    return jsonify({"response": response_message})


if __name__ == "__main__":
    app.run(debug=True)
