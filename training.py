from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

# Example: Load your dataset (ensure the correct path and file)
data = pd.read_csv('medical_data.csv')

# Example: Splitting into features and target variable
X = data.drop('target', axis=1)  # Replace 'target' with your actual target column
y = data['target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model to a file using pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as model.pkl")
