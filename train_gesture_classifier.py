import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os

def train_gesture_classifier():
    # Check if data file exists
    if not os.path.exists('gesture_data.csv'):
        print("Error: gesture_data.csv not found. Please collect data first.")
        return

    # Load collected data
    data = pd.read_csv('gesture_data.csv')
    
    # Validate data
    if len(data) < 10:
        print("Error: Not enough training samples. Need at least 10.")
        return
    
    unique_classes = data['label'].unique()
    if len(unique_classes) < 2:
        print(f"Error: Need at least 2 gesture classes. Found only: {unique_classes}")
        return
    
    print(f"Training on {len(data)} samples with classes: {unique_classes}")
    print("Class distribution:")
    print(data['label'].value_counts())

    # Prepare features and labels
    X = data.drop('label', axis=1)  # Features (landmarks)
    y = data['label']               # Labels

    # Handle NaN values if any
    if X.isna().any().any():
        print("Warning: NaN values found in features. Filling with 0.")
        X = X.fillna(0)

    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        stratify=y,  # Maintain class balance
        random_state=42
    )

    # Train SVM
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    model.fit(X_train, y_train)

    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Training accuracy: {train_acc:.2f}")
    print(f"Test accuracy: {test_acc:.2f}")

    # Save model and scaler
    joblib.dump(model, 'gesture_classifier.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model saved to gesture_classifier.joblib and scaler.joblib")

    # Save class names for reference
    np.save('class_names.npy', unique_classes)
    print("Class names saved to class_names.npy")

if __name__ == "__main__":
    train_gesture_classifier()