"""
Minimal Snort rule generator used by explainability_engine.py.

Translates SHAP-flagged features into Snort rule fields through a fixed
dictionary. Only attributes that exist in a packet/flow header can appear in
a rule; all other NSL-KDD features (content features, most rate features) are
ignored on purpose.
"""

# Preprocessed one-hot column -> Snort `flags:` value
FLAG_MAP = {
    "flag_S0": "S",     # SYN seen, no reply
    "flag_REJ": "R",    # connection rejected
    "flag_RSTO": "R",   # reset by originator
    "flag_RSTR": "R",   # reset by responder
}

# NSL-KDD `service` one-hot column -> destination port
SERVICE_PORTS = {
    "service_http": 80, "service_http_443": 443, "service_ftp": 21,
    "service_ftp_data": 20, "service_ssh": 22, "service_telnet": 23,
    "service_smtp": 25, "service_domain": 53, "service_domain_u": 53,
    "service_pop_3": 110, "service_imap4": 143,
}


def _protocol(packet_data: dict) -> str:
    for proto in ("tcp", "udp", "icmp"):
        if packet_data.get(f"protocol_type_{proto}", 0) == 1:
            return proto
    return "tcp"


def generate_snort_rule(top_features, packet_data, sid=1000001):
    """
    top_features: feature names ranked by SHAP (most influential first).
    packet_data:  dict of RAW (unscaled) feature values for the flagged flow.
    Returns a Snort rule string, or a comment line if nothing is translatable.
    """
    proto = _protocol(packet_data)
    dst_port = "any"
    options = []

    for feat in top_features:
        val = packet_data.get(feat, 0)
        if feat == "src_bytes" and val >= 1:
            options.append(f"dsize:>{int(val)}")
        elif feat in FLAG_MAP and val == 1 and proto == "tcp":
            options.append(f"flags:{FLAG_MAP[feat]}")
        elif feat in SERVICE_PORTS and val == 1 and proto in ("tcp", "udp"):
            dst_port = SERVICE_PORTS[feat]
        elif feat == "count" and val >= 1:
            options.append(f"detection_filter:track by_src, count {int(val)}, seconds 2")

    if not options and dst_port == "any":
        return ("# No header-derivable rule: top SHAP features "
                f"({', '.join(top_features)}) cannot be expressed in Snort syntax.")

    msg = "Auto-Mitigated Anomaly: " + ", ".join(top_features)
    body = "; ".join([f'msg:"{msg}"'] + options + [f"sid:{sid}", "rev:1"])
    return f"alert {proto} any any -> any {dst_port} ({body};)"