from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your model (make sure the model was trained with 4 features)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Map predictions to human-readable outcomes
outcome_mapping = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Cholesterol",
    3: "High Blood Pressure"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form (4 features) and validate them
        age = float(request.form['age'])
        blood_pressure = float(request.form['blood_pressure'])
        cholesterol = float(request.form['cholesterol'])
        bmi = float(request.form['bmi'])

        # Validate ranges (example ranges; adjust as necessary)
        if not (1 <= age <= 100):
            return "Age must be between 1 and 100."
        if not (50 <= blood_pressure <= 200):
            return "Blood Pressure must be between 50 and 200."
        if not (100 <= cholesterol <= 300):
            return "Cholesterol must be between 100 and 300."
        if not (10 <= bmi <= 50):
            return "BMI must be between 10 and 50."

        # Create feature array for prediction
        features = np.array([[age, blood_pressure, cholesterol, bmi]])

        # Use the model to predict
        prediction = model.predict(features)

        # Return the mapped prediction result
        result = outcome_mapping.get(prediction[0], "Unknown Condition")
        return f'The predicted medical outcome is: {result}'

    except ValueError:
        return "Please ensure all inputs are valid numbers."

if __name__ == '__main__':
    app.run(debug=True)
