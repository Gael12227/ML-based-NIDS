"""
Module A: SHAP explanation + Snort rule generation for one flagged flow.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from rule_generator import generate_snort_rule


def to_raw_values(pipeline, packet_df: pd.DataFrame) -> dict:
    """Undo StandardScaler on the numeric columns so rules use real units
    (e.g. src_bytes in bytes). One-hot columns are already 0/1."""
    num_cols = list(pipeline.transformers_[0][2])
    scaler = pipeline.named_transformers_["num"]
    row = packet_df.iloc[0]
    raw = row.to_dict()
    scaled = row[num_cols].to_numpy(dtype=float).reshape(1, -1)
    raw.update(dict(zip(num_cols, scaler.inverse_transform(scaled)[0])))
    return raw


def analyze_and_mitigate(model, packet_df, feature_names, raw_values=None,
                         explainer=None, top_k=3, sid=1000001):
    """
    model:         trained XGBoost classifier
    packet_df:     1-row DataFrame, preprocessed (same columns as X_train)
    feature_names: list of column names from feature_names.json
    raw_values:    dict of UNSCALED feature values (use to_raw_values). If
                   omitted, packet_df values are used, which are scaled and
                   will give wrong rule thresholds.
    explainer:     optional pre-built shap.TreeExplainer (build once, reuse)

    Returns (explanation, snort_rule). `explanation` is a single-row
    shap.Explanation for the anomaly class, usable with shap.plots.waterfall
    or shap.plots.force in Streamlit.
    """
    packet_df = pd.DataFrame(packet_df.to_numpy(), columns=feature_names)
    explainer = explainer or shap.TreeExplainer(model)

    expl = explainer(packet_df)[0]
    if expl.values.ndim == 2:
        expl = expl[:, 1]

    impacts = sorted(zip(feature_names, expl.values), key=lambda x: x[1], reverse=True)
    top_features = [f for f, v in impacts[:top_k] if v > 0]

    values = raw_values if raw_values is not None else packet_df.iloc[0].to_dict()
    snort_rule = generate_snort_rule(top_features, values, sid=sid)
    return expl, snort_rule


if __name__ == "__main__":
    from xgboost import XGBClassifier

    data = Path("data/processed")
    X_test_array = np.load(data / "X_test.npy", allow_pickle=True)
    X_test = pd.DataFrame(X_test_array, columns=json.load(open(data / "feature_names.json")))
    pipeline = joblib.load(data / "pipeline.pkl")
    feature_names = json.load(open(data / "feature_names.json"))

    model = XGBClassifier()
    model.load_model("xgboost_model_optimized.json")

    flagged_idx = np.where(model.predict(X_test) == 1)[0]
    print(f"{len(flagged_idx)} flows flagged as anomaly in the test set")

    found = 0
    for i in flagged_idx[:300]:
        packet_df = X_test.iloc[[i]]
        expl, rule = analyze_and_mitigate(
            model, packet_df, feature_names,
            raw_values=to_raw_values(pipeline, packet_df),
        )
        if not rule.startswith("#"):
            print(f"Flow {i} -> Snort rule:\n{rule}\n")
            found += 1
            if found == 3:
                break

    if found == 0:
        print("None of the first 300 flagged flows produced a header-derivable rule.")