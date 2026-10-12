import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

THRESHOLD = 0.02


def clean_service_columns(df):
    keep_services = {
        "service_http",
        "service_private",
        "service_domain_u",
        "service_smtp",
        "service_ftp_data",
        "service_eco_i",
        "service_other",
        "service_ftp",
        "service_telnet",
        "service_finger",
        "service_ecr_i",
        "service_auth",
        "service_pop_3",
        "service_time",
        "service_domain",
    }

    all_service_cols = [c for c in df.columns if c.startswith("service_")]
    rare_service_cols = [c for c in all_service_cols if c not in keep_services]

    if "service_other" not in df.columns:
        df["service_other"] = 0

    if rare_service_cols:
        df["service_other"] = (
            df["service_other"].fillna(0).astype(bool)
            | df[rare_service_cols].fillna(0).astype(bool).any(axis=1)
        ).astype(int)
        df.drop(columns=rare_service_cols, inplace=True)

    return df


def preprocess(df):
    df = df.copy()
    df["src_bytes"] = np.log1p(df["src_bytes"])
    df["dst_bytes"] = np.log1p(df["dst_bytes"])
    df = clean_service_columns(df)
    return df


train_df = preprocess(pd.read_csv("data.csv"))
test_df = preprocess(pd.read_csv("data2.csv"))

X = train_df.drop(columns=["target"])
y = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]
X_test = X_test.reindex(columns=X.columns, fill_value=0)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    objective="binary:logistic",
    tree_method="hist",
    device="cuda",
)

print("Starting cross-validation...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

model.fit(X, y)
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)

print(f"\nThreshold used: {THRESHOLD}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"Classification report:\n{classification_report(y_test, predictions, zero_division=0)}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, predictions, labels=[0, 1])}")

model.save_model("xgboost_model.json")

with open("Report.txt", "w") as f:
    f.write("XGBoost Model Report\n")
    f.write(f"Threshold used: {THRESHOLD}\n")
    f.write(f"Accuracy: {accuracy_score(y_test, predictions):.3f}\n")
    f.write(f"Classification Report:\n{classification_report(y_test, predictions, zero_division=0)}\n")
    f.write(f"Confusion Matrix:\n{confusion_matrix(y_test, predictions, labels=[0, 1])}\n")
