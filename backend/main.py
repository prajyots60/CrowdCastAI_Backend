from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

model_paths = {
    "success": "models/success_prediction_model.pkl",
    "funding": "models/funding_estimation_model.pkl"
}

@app.route("/predict/<model_type>", methods=["POST"])
def predict(model_type):
    if model_type not in model_paths:
        return jsonify({"error": "Invalid model type"}), 400

    try:
        input_data = request.json
        model = joblib.load(model_paths[model_type])

        if model_type == "success":
            features = [
                input_data["category"],
                input_data["product_stage"],
                input_data["project_type"],
                input_data["duration_days"],
                input_data["launch_month"],
                input_data["launch_year"],
                input_data["goal"],
                int(input_data["is_promoted"]),
                int(input_data["is_indemand"])
            ]
        else:
            features = list(input_data.values())

        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0].tolist()

        return jsonify({
            "prediction": int(prediction),
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
