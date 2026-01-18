from flask import Flask,request,jsonify
import joblib
import os
from pathlib import Path

# Define the Flask app
app = Flask(__name__)
# Define the path to the model
MODEL_PATH = Path("artifacts/model.pkl")  

if not MODEL_PATH.exists():
    import train_model as _train
    # Convenience code to train the model if not present, 
    # Train and Save the model if it doesn't exist
    _train.main()

# Load the model
model = joblib.load(MODEL_PATH)  

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data.get('features')
    if features is None:
        return jsonify({"error": "No features provided"}), 400

    pred = model.predict(features)
    return jsonify({"prediction": int[pred[0]]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)