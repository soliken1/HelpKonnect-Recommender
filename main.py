# main.py
from flask import Flask, request, jsonify
import openai
import os

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/recommend", methods=["POST"])
def recommend_facility():
    data = request.get_json()
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    try:
        completion = openai.ChatCompletion.create(
            model="ft:gpt-4o-mini-2024-07-18:personal::ANJY20nd",
            messages=[{"role": "user", "content": message}]
        )
        response_message = completion.choices[0].message["content"]
        return jsonify({"response": response_message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
