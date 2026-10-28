from flask import Flask, render_template, request, jsonify
import pickle
import random
import os
import numpy as np
import requests

app = Flask(__name__)

# Load your model and scaler
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError as e:
    raise FileNotFoundError("Model or scaler file not found. Please run train.py to generate them.") from e

# Define outcome mapping
outcome_mapping = {
    0: "Low Risk",
    1: "Medium Risk: High Cholesterol",
    2: "Medium Risk: High Blood Pressure",
    3: "High Risk: High Cholesterol & High Blood Pressure"
}

@app.route('/')
def main():
    return render_template('main.html')  # Serve the main page first

@app.route('/index')
def index():
    return render_template('index.html')  # This page contains the form for user input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        gender = request.form['gender']
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        cholesterol = float(request.form['cholesterol'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])

        # Convert gender to numeric
        gender_value = 0 if gender == 'Male' else 1
        
        # Create feature array
        features = np.array([[gender_value, glucose, blood_pressure, cholesterol, insulin, bmi, dpf, age]])

        # Scale features
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)[0]

        # Map prediction to readable outcome
        result = outcome_mapping.get(prediction, "Unknown Condition")

        return render_template('result.html', prediction=result)

    except ValueError as e:
        return f"Please ensure all inputs are valid numbers. Error: {e}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/advice', methods=['POST'])
def advice():
    prediction = request.form['prediction']
    
    # Customize advice based on the prediction result
    if prediction == "Low Risk":
        advice_message = (
            "Great job! Your health risk level is low. "
            "Maintain a balanced diet, stay active, and keep up regular checkups to continue in good health!"
        )
    elif prediction == "Medium Risk: High Cholesterol":
        advice_message = (
            "Your cholesterol levels are above normal. Consider reducing saturated fat intake, "
            "increasing fiber-rich foods, and consulting your doctor to monitor your cholesterol regularly."
        )
    elif prediction == "Medium Risk: High Blood Pressure":
        advice_message = (
            "High blood pressure can increase your risk of health complications. "
            "Incorporate more fruits, vegetables, and whole grains into your diet, and manage stress effectively."
        )
    elif prediction == "High Risk: High Cholesterol & High Blood Pressure":
        advice_message = (
            "Your health risk is high. It is strongly recommended that you consult a healthcare provider to address "
            "both cholesterol and blood pressure. Follow a heart-healthy diet, avoid smoking, and stay active."
        )
    else:
        advice_message = "Consult a healthcare professional for personalized advice based on your unique health profile."

    return render_template('advice.html', prediction=prediction, advice=advice_message)

def get_ai_response(user_message):
    api_key = os.environ.get('OPENAI_API_KEY')  # Replace with your actual API key
    if not api_key:
        return "Chatbot is not configured. Set the OPENAI_API_KEY environment variable."
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    data = {
        'model': 'gpt-3.5-turbo',  # Choose the model you want to use
        'messages': [{'role': 'user', 'content': user_message}],
        'max_tokens': 150,
        'temperature': 0.7,  # Adjust the creativity of the response
    }
    
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    ai_response = get_ai_response(user_message)
    return jsonify({'response': ai_response})

# Define the ask route for the chatbot feature
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')

    # Generate a response - you could replace this with a call to get_ai_response() if using the OpenAI API
    responses = [
        "Regular checkups are important for your health!",
        "Stay hydrated and eat plenty of fruits and vegetables.",
        "Don't forget to exercise regularly!",
        "Make sure to get enough sleep!",
        "It's always good to consult with a healthcare professional."
    ]
    answer = random.choice(responses)

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
