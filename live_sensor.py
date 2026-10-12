import oracledb
import pandas as pd
import json
import joblib
import os
from dotenv import load_dotenv
import oracledb
from scapy.all import sniff, IP, TCP, UDP, ICMP
from xgboost import XGBClassifier
from explainability_engine import analyze_and_mitigate
load_dotenv()
model = XGBClassifier()
model.load_model("xgboost_model_optimized.json")
pipeline = joblib.load("data/processed/pipeline.pkl")
with open("data/processed/feature_names.json", "r") as f:
    feature_names = json.load(f)

db_conn = oracledb.connect(
    user=os.getenv("DB_USER"), 
    password=os.getenv("DB_PASS"), 
    dsn=os.getenv("DB_DSN")
)
cursor = db_conn.cursor()

def extract_live_features(pkt):
    flow_data = {feat: 0.0 for feat in feature_names}
    if IP in pkt:
        if 'src_bytes' in flow_data:
            flow_data['src_bytes'] = float(len(pkt[IP].payload))
        if TCP in pkt:
            if 'protocol_type_tcp' in flow_data: flow_data['protocol_type_tcp'] = 1.0
            dport = pkt[TCP].dport
            if dport == 80 and 'service_http' in flow_data: flow_data['service_http'] = 1.0
            elif dport == 443 and 'service_http_443' in flow_data: flow_data['service_http_443'] = 1.0
            elif dport == 22 and 'service_ssh' in flow_data: flow_data['service_ssh'] = 1.0
            flags = pkt[TCP].flags
            if "S" in flags and 'flag_S0' in flow_data: flow_data['flag_S0'] = 1.0
            elif "R" in flags and 'flag_REJ' in flow_data: flow_data['flag_REJ'] = 1.0
        elif UDP in pkt:
            if 'protocol_type_udp' in flow_data: flow_data['protocol_type_udp'] = 1.0
            if pkt[UDP].dport == 53 and 'service_domain_u' in flow_data: flow_data['service_domain_u'] = 1.0
        elif ICMP in pkt:
            if 'protocol_type_icmp' in flow_data: flow_data['protocol_type_icmp'] = 1.0
    return flow_data

def process_live_packet(pkt):
    if IP not in pkt:
        return
        
    flow_data = extract_live_features(pkt)
    raw_packet_df = pd.DataFrame([flow_data])
    
    num_cols = list(pipeline.transformers_[0][2])
    scaler = pipeline.named_transformers_["num"]
    scaled_df = raw_packet_df.copy()
    scaled_df[num_cols] = scaler.transform(raw_packet_df[num_cols])
    
    probabilities = model.predict_proba(scaled_df)
    anomaly_probability = float(probabilities[0][1])
    
    if anomaly_probability >= 0.30:
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        
        expl, snort_rule = analyze_and_mitigate(
            model, scaled_df, feature_names, raw_values=raw_packet_df.iloc[0].to_dict()
        )

        if anomaly_probability >= 0.85:
            anomaly_origin = "Known Dataset Anomaly"
        else:
            anomaly_origin = "Zero-Day Anomaly"
            
        if snort_rule.startswith("#"):
            threat_severity = "Not Harmful (Benign Spike)"
        else:
            threat_severity = "Harmful (Action Required)"

        cursor.execute("""
            INSERT INTO Threat_Logs (source_ip, destination_ip, confidence_score, threat_severity, anomaly_origin) 
            VALUES (:1, :2, :3, :4, :5) RETURNING alert_id INTO :alert_id
        """, (src_ip, dst_ip, anomaly_probability, threat_severity, anomaly_origin))
        
        alert_id = cursor.var(oracledb.NUMBER)
        
        cursor.execute("""
            INSERT INTO Generated_Rules (alert_id, snort_syntax) 
            VALUES (:1, :2)
        """, (alert_id.getvalue()[0], snort_rule))
        
        db_conn.commit()

sniff(iface="Wi-Fi", filter="ip", prn=process_live_packet, store=0)