import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

def classify_risk(row):
    if row['Cholesterol'] > 200 and row['BloodPressure'] > 130:
        return 'High Risk: High Cholesterol & High Blood Pressure'
    elif row['Cholesterol'] > 200:
        return 'Medium Risk: High Cholesterol'
    elif row['BloodPressure'] > 130:
        return 'Medium Risk: High Blood Pressure'
    else:
        return 'Low Risk'

def train_model():
    # Load the data
    filename = 'medical_data.csv'
    data = pd.read_csv(filename)

    # Convert Gender to numerical values
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

    # Define features and target
    features = data[['Gender', 'Glucose', 'BloodPressure', 'Cholesterol', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    target = data['Outcome']

    # Apply risk classification
    data['RiskLevel'] = data.apply(classify_risk, axis=1)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save the model and scaler
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    train_model()
