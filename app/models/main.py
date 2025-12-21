from flask import Flask, request, jsonify
from prediction_model import SoccerPredictionModel

app = Flask(__name__)
model = SoccerPredictionModel('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.json
    result = model.predict_from_payload(payload)   # implement this
    return jsonify(result)

@app.route('/health')
def health():
    return 'ok', 200

    web: gunicorn main:app --bind 0.0.0.0:$PORT
