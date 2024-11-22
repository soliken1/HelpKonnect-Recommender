from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import numpy as np
import requests

url = "https://helpkonnect.vercel.app/api/fetchFacilities"
user_preference_url = "https://helpkonnect.vercel.app/api/fetchUser"
user_answers_url = "https://helpkonnect.vercel.app/api/fetchAnswers"
analyze_answer_url = "https://helpkonnect.vercel.app/api/analyzePref"
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
        "embedding": get_embedding(facility["facilityExpertise"])
    }
    for facility in data['fetchFacility']
]


def is_recommendation_request(message):
    keywords = ["recommend", "suggest", "facility", "help", "support"]
    return any(keyword in message.lower() for keyword in keywords)


@app.route("/preference", methods=["POST"])
def analyze_user_preference():
    data = request.get_json()
    user_id = data.get("userId", "")

    if not user_id:
        return jsonify({"error": "No userId provided"}), 400

    try:
        # Step 1: Fetch User Answers
        answers_response = requests.post(user_answers_url, json={"userId": user_id})
        answers_response.raise_for_status()
        user_answers = answers_response.json().get("userAnswers", [])

        if not user_answers:
            return jsonify({"error": "No answers found for the user"}), 404

        # Prepare answers for analysis
        formatted_answers = "\n".join([
            f"Question: {qa['question']}\nAnswer: {qa['answer']}"
            for qa in user_answers
        ])

        # Step 2: Analyze Answers using OpenAI
        try:
            analysis_response = openai.ChatCompletion.create(
                model="ft:gpt-4o-mini-2024-07-18:personal::ANJY20nd",
                messages=[
                    {"role": "system", "content": "You are a system designed to analyze user preferences from a "
                                                  "series of questions and answers. Your goal is to generate a "
                                                  "concise list of tags summarizing the user's preferences. Each tag "
                                                  "should be a maximum of three words and should end with a comma. "
                                                  "Avoid using complete sentences or numbered lists. Focus on key "
                                                  "aspects of the preferences, capturing the user's input succinctly. "
                                                  "Here is the user's input:"},
                    {"role": "user", "content": formatted_answers}
                ]
            )
            analyzed_preference = analysis_response.choices[0].message["content"].strip()
        except Exception as e:
            return jsonify({"error": f"OpenAI analysis failed: {str(e)}"}), 500

        # Step 3: Update User Preference
        update_response = requests.post(analyze_answer_url, json={
            "userId": user_id,
            "userPref": analyzed_preference
        })
        update_response.raise_for_status()

        # Step 4: Return Success Response
        return jsonify({"response": "User preference updated successfully", "preference": analyzed_preference})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/recommend", methods=["POST"])
def recommend_with_interaction():
    data = request.get_json()
    user_id = data.get("userId", "")
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Fetch user preference
    try:
        preference_response = requests.post(user_preference_url, json={"userId": user_id})
        preference_response.raise_for_status()
        user_preference = preference_response.json().get("preference", "")
    except Exception as e:
        return jsonify({"error": f"Failed to fetch user preference: {str(e)}"}), 500

    # Combine user preference with the query message
    combined_query = f"{message} {user_preference}"

    try:
        # Generate embedding for the combined query
        query_embedding = get_embedding(combined_query)

        completion = openai.ChatCompletion.create(
            model="ft:gpt-4o-mini-2024-07-18:personal::ANJY20nd",
            messages=[{"role": "user", "content": message}]
        )
        user_response = completion.choices[0].message["content"]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if is_recommendation_request(message):
        # Compute similarities
        similarities = [
            (facility, np.dot(query_embedding, facility["embedding"].flatten()))
            for facility in facilities
        ]

        # Find the best match
        best_match = max(similarities, key=lambda x: x[1])[0]

        response_message = (
            f"{user_response}\n\n"
            f"Recommended Facility: {best_match['id']}\n"
            f"Description: {best_match['description']}\n"
            f"Expertise: {best_match['expertise']}"
        )
    else:
        response_message = user_response

    return jsonify({"response": response_message})


if __name__ == "__main__":
    app.run(debug=True)
