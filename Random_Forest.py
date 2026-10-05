import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def main():
    # 1. Load Preprocessed Data
    print("Loading preprocessed dataset artifacts from 'data/processed/'...")
    try:
        X_train = np.load('data/processed/X_train.npy')
        X_test = np.load('data/processed/X_test.npy')
        
        # The preprocessing script exports the target variable under the column name 'label'[cite: 3]
        y_train = pd.read_csv('data/processed/y_train.csv')['label'].values
        y_test = pd.read_csv('data/processed/y_test.csv')['label'].values
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure preprocess.py has been executed first.")
        return

    # 2. Initialize Random Forest Classifier
    print("Initializing RandomForestClassifier...")
    rf_classifier = RandomForestClassifier(
        n_estimators=100, 
        max_depth=None,
        random_state=42, 
        n_jobs=-1,
        verbose=1
    )

    # 3. Model Training
    print("Training Random Forest baseline model...")
    rf_classifier.fit(X_train, y_train)

    # 4. Inference
    print("Generating predictions on the test set...")
    predictions = rf_classifier.predict(X_test)
    
    # Extract probabilities for the positive class (Anomaly = 1) for a valid ROC-AUC calculation
    probabilities = rf_classifier.predict_proba(X_test)[:, 1]

    # 5. Evaluation Metrics
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    class_report = classification_report(y_test, predictions, target_names=['Normal (0)', 'Anomaly (1)'])
    conf_matrix = confusion_matrix(y_test, predictions)

    print("\n--- Random Forest Evaluation Results ---")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score:     {roc_auc:.4f}")
    print("\nClassification Report:\n", class_report)
    print("Confusion Matrix:\n", conf_matrix)

    # 6. Serialize and Export the Model
    model_path = 'rf_baseline_model.pkl'
    joblib.dump(rf_classifier, model_path)
    print(f"\nModel successfully saved to '{model_path}'.")

if __name__ == "__main__":
    main()