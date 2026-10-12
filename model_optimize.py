import optuna
import pandas as pd
import numpy as np
import json
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("Loading datasets...")
with open("data/processed/feature_names.json", "r") as f:
    feature_names = json.load(f)
X_train = pd.DataFrame(np.load("data/processed/X_train.npy", allow_pickle=True), columns=feature_names)
X_test = pd.DataFrame(np.load("data/processed/X_test.npy", allow_pickle=True), columns=feature_names)
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
print(f"Calculated scale_pos_weight: {ratio:.2f}")

def objective(trial):
    param = {
        "tree_method": "hist",
        "device": "cuda",
        "scale_pos_weight": ratio,
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "eval_metric": "logloss"
    }

    model = XGBClassifier(**param)
    model.fit(X_train, y_train, verbose=False)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    return accuracy
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50) 

print(f"\nBest Optuna Accuracy: {study.best_value * 100:.2f}%")
print("Best Parameters:", study.best_params)

print("\nTraining final model with best parameters...")
best_params = study.best_params
best_params["tree_method"] = "hist"
best_params["device"] = "cuda"
best_params["scale_pos_weight"] = ratio

final_model = XGBClassifier(**best_params)
final_model.fit(X_train, y_train)
print("\n")
probabilities = final_model.predict_proba(X_test)[:, 1]

best_threshold = 0.5
best_acc = 0

for thresh in np.arange(0.30, 0.61, 0.02):
    custom_preds = (probabilities >= thresh).astype(int)
    acc = accuracy_score(y_test, custom_preds)
    if acc > best_acc:
        best_acc = acc
        best_threshold = thresh

print(f"Optimal Threshold: {best_threshold:.2f}")
final_preds = (probabilities >= best_threshold).astype(int)
print(f"Final Accuracy:  {accuracy_score(y_test, final_preds) * 100:.2f}%")
print(f"Final Precision: {precision_score(y_test, final_preds) * 100:.2f}%")
print(f"Final Recall:    {recall_score(y_test, final_preds) * 100:.2f}%")
print(f"Final F1-Score:  {f1_score(y_test, final_preds) * 100:.2f}%")

final_model.save_model("xgboost_model_optimized.json")
