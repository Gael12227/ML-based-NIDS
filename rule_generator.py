FLAG_MAP = {
    "flag_S0": "S",     # SYN seen, no reply
    "flag_REJ": "R",    # connection rejected
    "flag_RSTO": "R",   # reset by originator
    "flag_RSTR": "R",   # reset by responder
}

SERVICE_PORTS = {
    "service_http": 80, "service_http_443": 443, "service_ftp": 21,
    "service_ftp_data": 20, "service_ssh": 22, "service_telnet": 23,
    "service_smtp": 25, "service_domain": 53, "service_domain_u": 53,
    "service_pop_3": 110, "service_imap4": 143,
}

DSIZE_MIN = 20
DSIZE_MAX = 1460


def _protocol(packet_data: dict) -> str:
    for proto in ("tcp", "udp", "icmp"):
        if packet_data.get(f"protocol_type_{proto}", 0) == 1:
            return proto
    return "tcp"


def generate_snort_rule(top_features, packet_data, sid=1000001):
    """
    Returns a Snort rule string, or a comment line if nothing is translatable.
    """
    proto = _protocol(packet_data)
    dst_port = "any"
    options = []

    for feat in top_features:
        val = packet_data.get(feat, 0)
        if feat == "src_bytes" and DSIZE_MIN <= val <= DSIZE_MAX:
            options.append(f"dsize:>{int(val)}")
        elif feat in FLAG_MAP and val == 1 and proto == "tcp":
            options.append(f"flags:{FLAG_MAP[feat]}")
        elif feat == "service_ecr_i" and val == 1 and proto == "icmp":
            options.append("itype:0") 
        elif feat in SERVICE_PORTS and val == 1 and proto in ("tcp", "udp"):
            dst_port = SERVICE_PORTS[feat]
        elif feat == "count" and val >= 1:
            options.append(f"detection_filter:track by_src, count {int(val)}, seconds 2")
    if not options:
        return ("# No header-derivable rule: top SHAP features "
                f"({', '.join(top_features)}) cannot be expressed in Snort syntax.")

    msg = "Auto-Mitigated Anomaly: " + ", ".join(top_features)
    body = "; ".join([f'msg:"{msg}"'] + options + [f"sid:{sid}", "rev:1"])
    return f"alert {proto} any any -> any {dst_port} ({body};)"

def generate_iptables_rule(top_features, packet_data):
    """
    Translates SHAP-flagged features into a Linux iptables DROP command.
    """
    proto = _protocol(packet_data)
    dst_port = None
    for feat in top_features:
        val = packet_data.get(feat, 0)
        if feat in SERVICE_PORTS and val == 1 and proto in ("tcp", "udp"):
            dst_port = SERVICE_PORTS[feat]
            break

    if dst_port:
        return f"iptables -A INPUT -p {proto} --dport {dst_port} -j DROP"
    else:
        return f"iptables -A INPUT -p {proto} -m state --state NEW -j DROP"