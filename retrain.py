import os
import json
import joblib
import oracledb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from xgboost import XGBClassifier

print("Initializing MLOps Continuous Learning Pipeline...")
load_dotenv()
try:
    db_conn = oracledb.connect(
        user=os.getenv("DB_USER"), 
        password=os.getenv("DB_PASS"), 
        dsn=os.getenv("DB_DSN")
    )
    cursor = db_conn.cursor()
except Exception as e:
    print(f"[!] Database connection failed: {e}")
    exit()
cursor.execute("""
    SELECT alert_id, raw_features 
    FROM Threat_Logs 
    WHERE threat_severity = 'False Positive'
""")
rows = cursor.fetchall()

if not rows:
    print("No false positives reported. Model is up to date.")
    exit()

print(f"Found {len(rows)} False Positives. Commencing data extraction...")
with open("data/processed/feature_names.json", "r") as f:
    feature_names = json.load(f)
pipeline = joblib.load("data/processed/pipeline.pkl")

new_flows = []
alert_ids = []

for row in rows:
    alert_ids.append(row[0])
    flow_dict = json.loads(row[1])
    new_flows.append(flow_dict)
raw_df = pd.DataFrame(new_flows)[feature_names]
num_cols = list(pipeline.transformers_[0][2])
scaler = pipeline.named_transformers_["num"]
scaled_df = raw_df.copy()
scaled_df[num_cols] = scaler.transform(raw_df[num_cols])
X_new = scaled_df
y_new = np.zeros(len(X_new))
print("Hot-loading existing model onto RTX 3050...")
model = XGBClassifier(device="cuda", tree_method="hist")
model.load_model("xgboost_model_optimized.json")

print("Incrementally training new decision trees...")
model.fit(X_new, y_new, xgb_model="xgboost_model_optimized.json")
model.save_model("xgboost_model_optimized.json")
print("Model successfully updated and saved.")
for aid in alert_ids:
    cursor.execute("""
        UPDATE Threat_Logs 
        SET threat_severity = 'False Positive (Retrained)' 
        WHERE alert_id = :1
    """, (aid,))
db_conn.commit()

print("MLOps Pipeline Complete. CI/CD loop closed.")