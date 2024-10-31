import pandas as pd
import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load your model and scaler
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("Model file 'model.pkl' not found.")

try:
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("Scaler file 'scaler.pkl' not found.")

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
        # Get input values
        gender = request.form['gender']
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        cholesterol = float(request.form['cholesterol'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])

        # Convert gender to numerical value: 0 for Male, 1 for Female
        gender_value = 0 if gender == 'Male' else 1
        
        # Create the feature array with all 8 features
        features = np.array([[gender_value, glucose, blood_pressure, cholesterol, insulin, bmi, dpf, age]])

        # Scale features
        scaled_features = scaler.transform(features)

        # Use the model to predict
        prediction = model.predict(scaled_features)
        result = outcome_mapping.get(prediction[0], "Unknown Condition")

        # Save the input data to a CSV file
        output_data = {
            'Gender': gender,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'Cholesterol': cholesterol,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age,
            'Prediction': result
        }

        # Check if the CSV file exists
        csv_file = 'predictions.csv'
        if not os.path.exists(csv_file):
            df = pd.DataFrame(columns=output_data.keys())
        else:
            df = pd.read_csv(csv_file)

        # Append the new data and save it back to CSV
        df = df.append(output_data, ignore_index=True)
        df.to_csv(csv_file, index=False)

        # Render the result template
        return render_template('result.html', prediction=result)

    except ValueError as e:
        return f"Please ensure all inputs are valid numbers. Error: {e}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
