import pandas as pd
import json
from scapy.all import rdpcap, IP, TCP, UDP, ICMP

def extract_flow_features(pcap_filepath, feature_names_path="data/processed/feature_names.json"):
    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)
    flow_stats = {col: 0.0 for col in feature_names}
    
    print(f"Reading {pcap_filepath}...")
    packets = rdpcap(pcap_filepath)
    for pkt in packets:
        if IP in pkt:
            if 'src_bytes' in flow_stats:
                flow_stats['src_bytes'] = float(len(pkt[IP].payload))
            if TCP in pkt:
                if 'protocol_type_tcp' in flow_stats: flow_stats['protocol_type_tcp'] = 1.0
                flags = pkt[TCP].flags
                if flags == 'S' and 'flag_S0' in flow_stats:
                    flow_stats['flag_S0'] = 1.0
                elif flags == 'R' and 'flag_REJ' in flow_stats:
                    flow_stats['flag_REJ'] = 1.0
                dport = pkt[TCP].dport
                if dport == 80 and 'service_http' in flow_stats: flow_stats['service_http'] = 1.0
                elif dport == 22 and 'service_ssh' in flow_stats: flow_stats['service_ssh'] = 1.0
                elif dport == 443 and 'service_http_443' in flow_stats: flow_stats['service_http_443'] = 1.0
                    
            elif UDP in pkt:
                if 'protocol_type_udp' in flow_stats: flow_stats['protocol_type_udp'] = 1.0
                dport = pkt[UDP].dport
                if dport == 53 and 'service_domain_u' in flow_stats: flow_stats['service_domain_u'] = 1.0
                
            elif ICMP in pkt:
                if 'protocol_type_icmp' in flow_stats: flow_stats['protocol_type_icmp'] = 1.0
            
            break
    packet_df = pd.DataFrame([flow_stats])
    packet_df = packet_df[feature_names]
    
    return packet_df

if __name__ == "__main__":
    try:
        df = extract_flow_features('test.pcap')
        print(f"PCAP parsed successfully. Shape: {df.shape}")
        print(df[['src_bytes', 'protocol_type_tcp', 'service_http']].head())
    except Exception as e:
        print(f"Error: {e}")