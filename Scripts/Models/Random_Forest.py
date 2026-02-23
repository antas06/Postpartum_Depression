import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # For saving the Random Forest model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the split datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Separate Features (X) and Target (y)
X_train = train_data.drop('Labelling', axis=1)
y_train = train_data['Labelling']

X_test = test_data.drop('Labelling', axis=1)
y_test = test_data['Labelling']

print(f"Training Features Shape: {X_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")

# Dictionary to store performance for both models
model_performance = {}

#Random Forest Model
from sklearn.ensemble import RandomForestClassifier

print("--- Training Random Forest ---")
# Initialize and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Make Predictions
rf_preds = rf_model.predict(X_test)

# Calculate Metrics
rf_acc = accuracy_score(y_test, rf_preds)
rf_prec = precision_score(y_test, rf_preds)
rf_rec = recall_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds)

# Store Performance
model_performance['Random Forest'] = {'Accuracy': rf_acc, 'Precision': rf_prec, 'Recall': rf_rec, 'F1-Score': rf_f1}

# Save the Model
joblib.dump(rf_model, 'random_forest_model.pkl')

print("Random Forest Training Complete!")
print(f"Accuracy: {rf_acc:.4f} | F1-Score: {rf_f1:.4f}")
print("Model saved as 'random_forest_model.pkl'")