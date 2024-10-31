# data.py
import pandas as pd
import os

# Define the filename
filename = 'medical_data.csv'

# Check if the file exists
if not os.path.exists(filename):
    # Create sample data if the file does not exist
    data = {
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Glucose': [148, 85, 183, 89, 137, 116, 78, 115, 100, 196],
        'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 70, 80, 90],
        'Cholesterol': [200, 150, 230, 180, 250, 190, 210, 160, 220, 240],
        'Insulin': [0, 0, 0, 94, 168, 0, 88, 0, 0, 0],
        'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 34.6, 30.1, 39.4],
        'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.349, 0.393, 0.725],
        'Age': [50, 31, 32, 21, 33, 30, 26, 29, 23, 38],
        'Outcome': [1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
    }
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Save DataFrame to CSV
    df.to_csv(filename, index=False)
    print(f"{filename} created with sample data.")

# Load the data
data = pd.read_csv(filename)

# Convert Gender to numerical values
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Define the risk classification based on cholesterol and blood pressure levels
def classify_risk(row):
    if row['Cholesterol'] > 200 and row['BloodPressure'] > 130:
        return "High Risk: High Cholesterol & High Blood Pressure"
    elif row['Cholesterol'] > 200:
        return "Medium Risk: High Cholesterol"
    elif row['BloodPressure'] > 130:
        return "Medium Risk: High Blood Pressure"
    else:
        return "Low Risk"

# Apply the risk classification to each row
data['RiskLevel'] = data.apply(classify_risk, axis=1)

# Save the modified data
data.to_csv('medical_data.csv', index=False)
print("Data with risk levels saved as 'medical_data.csv'.")
