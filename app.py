from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("battery_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extract and reshape the input features
    features = np.array([
        data['temperature'],
        data['voltage'],
        data['current'],
        data['charge_cycles']
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)[0]
    
    # Convert prediction to readable status
    status = "Unsafe" if prediction == 1 else "Safe"

    # Return both numeric and readable response
    return jsonify({
        'burn_risk': int(prediction),
        'status': status
    })

if __name__ == "__main__":
    app.run(debug=True)
