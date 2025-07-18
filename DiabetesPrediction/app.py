# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from model import NeuralNetwork   # <--- IMPORTANT

# Load trained model
with open('nn_model.pkl', 'rb') as f:
    nn = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print("Received data:", data)

    # Extract the values in the correct order your model expects
    features = [
        float(data.get("pregnancies", 0)),
        float(data["glucose"]),
        float(data["bloodPressure"]),
        float(data["skinThickness"]),
        float(data["insulin"]),
        float(data["bmi"]),
        float(data.get("diabetesPedigreeFunction", 0)),  # if used
        float(data["age"]),
    ]

    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    print("features:",features)

    pred,prob = nn.predict(features_scaled)
    print("Prediction and probability:",pred,prob)

    return jsonify({
        "prediction": pred[0][0],        # 0 or 1
        "probability": prob[0][0]
    })



if __name__ == '__main__':
    app.run(port=5000, debug=True)
