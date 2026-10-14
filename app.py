import html as html_lib
import io
import json
import tempfile
from datetime import datetime
import os
from dotenv import load_dotenv
import oracledb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from xgboost import XGBClassifier
from explainability_engine import analyze_and_mitigate, to_raw_values

try:
    from pcap_parser import extract_flow_features
    PCAP_MODULE_READY = True
except ImportError:
    PCAP_MODULE_READY = False

load_dotenv()
try:
    db_conn = oracledb.connect(
        user=os.getenv("DB_USER"), 
        password=os.getenv("DB_PASS"), 
        dsn=os.getenv("DB_DSN")
    )
    cursor = db_conn.cursor()
except Exception as e:
    st.error(f"Database connection failed: {e}")

st.set_page_config(
    page_title="Closed-Loop NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --blush: #F6E2E9;
  --cream: #FEFAF3;
  --sage: #B8CFB3;
  --sage-deep: #84A282;
  --ink: #000000;
  --alert: #B0506D;
  --watch: #D4A0B3;
  --clear: #84A282;
}

html, body, [class*="css"], .stApp {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
.stApp { background: var(--cream); }
.block-container { padding-top: 2rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }

section[data-testid="stSidebar"] {
  background: var(--blush);
  border-right: 1px solid var(--sage-deep);
}
section[data-testid="stSidebar"] h2 { font-size: 0.95rem; font-weight: 600; }
h1, h2, h3, h4 { font-weight: 600; letter-spacing: -0.01em; }
hr { border-color: var(--sage-deep) !important; opacity: 0.5; }

/* ---------- All text black (every surface is light) ---------- */
.stApp, .stApp *, div[data-baseweb="popover"], div[data-baseweb="popover"] * {
  color: #000000 !important;
}

/* ---------- App header ---------- */
.app-head { display: flex; justify-content: space-between; align-items: flex-end;
  gap: 1rem; flex-wrap: wrap; padding-bottom: 1rem; margin-bottom: 1.25rem;
  border-bottom: 2px solid var(--sage); }
.app-head h1 { margin: 0; font-size: 1.75rem; padding: 0; }
.app-head p { margin: 0.25rem 0 0; font-size: 0.95rem; }
.status { display: flex; gap: 1.5rem; font-size: 0.85rem; }
.status b { display: block; font-size: 1.05rem; font-weight: 600; }

/* ---------- Verdict panel ---------- */
.verdict { background: var(--blush); border: 1px solid var(--sage-deep);
  border-left: 8px solid var(--tone); border-radius: 6px; padding: 1.4rem 1.6rem 1.2rem; }
.verdict.alert { --tone: var(--alert); }
.verdict.watch { --tone: var(--watch); }
.verdict.clear { --tone: var(--clear); }
.verdict h2 { margin: 0 0 0.3rem; font-size: 1.5rem; padding: 0; }
.verdict .lede { margin: 0 0 1.25rem; max-width: 70ch; }

.facts { display: grid; grid-template-columns: repeat(3, 1fr);
  border-top: 1px solid var(--sage-deep); margin-top: 1.3rem; }
.facts div { padding: 0.9rem 1rem 0 0; }
.facts div + div { padding-left: 1rem; border-left: 1px solid var(--sage-deep); }
.facts span { display: block; font-size: 0.8rem; }
.facts strong { font-size: 1.05rem; font-weight: 600; }

/* ---------- Confidence gauge ---------- */
.gauge { position: relative; height: 14px; display: flex; }
.gauge .z { height: 100%; }
.gauge .z:first-child { border-radius: 7px 0 0 7px; }
.gauge .z:last-child { border-radius: 0 7px 7px 0; }
.gauge .z.c { background: var(--sage); }
.gauge .z.w { background: var(--watch); }
.gauge .z.a { background: var(--alert); }
.gauge .pin { position: absolute; top: -7px; width: 4px; height: 28px; margin-left: -2px;
  background: #000; border-radius: 2px; }
.gauge-labels { display: flex; font-size: 0.78rem; margin-top: 0.55rem; }
.gauge-labels span { overflow: hidden; white-space: nowrap; }

/* ---------- Notices (replace st.info / st.warning) ---------- */
.notice { border-left: 5px solid var(--sage-deep); border-radius: 4px;
  padding: 0.8rem 1rem; margin: 0.6rem 0; background: #FFFFFF; }
.notice.success { background: var(--sage); }
.notice.warn { background: var(--blush); border-left-color: var(--alert); }

/* ---------- Empty state ---------- */
.empty { background: var(--blush); border: 1px dashed var(--sage-deep); border-radius: 6px;
  padding: 2.5rem 2rem; }
.empty h3 { margin: 0 0 0.4rem; font-size: 1.15rem; }
.empty p { margin: 0; max-width: 62ch; }

/* ---------- Centered analysis section ---------- */
.center { text-align: center; }
.center-h { text-align: center; margin: 1.2rem 0 0.2rem; font-size: 1.1rem; }

/* ---------- HTML tables ---------- */
.tbl-wrap { overflow: auto; max-height: 420px; border: 1px solid var(--sage-deep);
  border-radius: 6px; background: var(--cream); }
.tbl-wrap table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.tbl-wrap thead tr, .tbl-wrap th { text-align: left !important; }
.tbl-wrap th { background: var(--sage); padding: 0.55rem 0.8rem; font-weight: 600;
  position: sticky; top: 0; }
.tbl-wrap td { padding: 0.5rem 0.8rem; border-top: 1px solid rgba(132,162,130,0.45); }
.tbl-wrap tbody tr:nth-child(even) td { background: rgba(246,226,233,0.6); }

/* ---------- Tabs ---------- */
button[data-baseweb="tab"] { font-weight: 500; background: transparent !important; }
div[data-baseweb="tab-highlight"] { background-color: var(--sage-deep) !important; }
div[data-baseweb="tab-border"] { background-color: var(--sage) !important; }

/* ---------- Buttons ---------- */
button[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"] {
  background: var(--sage-deep) !important; border: 1px solid var(--sage-deep) !important;
  border-radius: 4px; font-weight: 500;
}
button[data-testid="stBaseButton-secondary"], .stButton > button[kind="secondary"],
.stDownloadButton > button, [data-testid="stFileUploaderDropzone"] button {
  background: var(--cream) !important; border: 1px solid var(--sage-deep) !important;
  border-radius: 4px; font-weight: 500;
}
button[data-testid="stBaseButton-primary"]:hover, button[data-testid="stBaseButton-secondary"]:hover,
.stDownloadButton > button:hover, [data-testid="stFileUploaderDropzone"] button:hover {
  background: var(--sage) !important; border-color: var(--sage-deep) !important;
}
button:disabled { opacity: 0.45 !important; }

/* ---------- Dropdown (select box) ---------- */
div[data-baseweb="select"] > div {
  background: var(--cream) !important; border: 1px solid var(--sage-deep) !important;
}
div[data-baseweb="select"] input { -webkit-text-fill-color: #000 !important; }
div[data-baseweb="select"] svg { fill: #000 !important; }
div[data-baseweb="popover"], div[data-baseweb="popover"] ul, div[data-baseweb="menu"] {
  background: var(--cream) !important;
}
div[data-baseweb="popover"] li { background: var(--cream) !important; }
div[data-baseweb="popover"] li:hover { background: var(--sage) !important; }
div[data-baseweb="popover"] li[aria-selected="true"] { background: var(--blush) !important; }

/* ---------- File uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
  background: var(--cream) !important; border: 1px dashed var(--sage-deep) !important;
}
[data-testid="stFileUploaderDropzone"] svg { fill: #000 !important; }

/* ---------- Radio ---------- */
[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
  background-color: var(--cream) !important; border-color: var(--sage-deep) !important;
}

/* ---------- Code block (Snort rule) ---------- */
[data-testid="stCode"] pre, [data-testid="stCode"] {
  background: #FFFFFF !important; border: 1px solid var(--sage-deep); border-radius: 6px;
}
[data-testid="stCode"] *, [data-testid="stCode"] code span {
  background: transparent !important; color: #000000 !important;
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
}
code, pre { font-family: 'IBM Plex Mono', ui-monospace, monospace !important; }

/* ---------- Images ---------- */
[data-testid="stImage"] img { border: 1px solid var(--sage-deep); border-radius: 6px; }

@media (max-width: 760px) {
  .facts { grid-template-columns: 1fr; }
  .facts div + div { padding-left: 0; border-left: 0; border-top: 1px solid var(--sage-deep); margin-top: 0.8rem; }
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

NORMAL_THR = 0.30
KNOWN_THR = 0.85

@st.cache_resource
def load_assets():
    model = XGBClassifier()
    model.load_model("xgboost_model_optimized.json")
    pipeline = joblib.load("data/processed/pipeline.pkl")
    with open("data/processed/feature_names.json", "r") as f:
        feature_names = json.load(f)
    explainer = shap.TreeExplainer(model)
    X_test_array = np.load("data/processed/X_test.npy", allow_pickle=True)
    X_test = pd.DataFrame(X_test_array, columns=feature_names)
    flagged_idx = np.where(model.predict(X_test) == 1)[0]
    proba = model.predict_proba(X_test)[:, 1]
    return model, pipeline, feature_names, explainer, X_test, flagged_idx, proba


model, pipeline, feature_names, explainer, X_test, flagged_idx, test_proba = load_assets()

if "result" not in st.session_state:
    st.session_state.result = None
    
if "ledger" not in st.session_state:
    st.session_state.ledger = []
    
if "allowlist_ips" not in st.session_state:
    st.session_state.allowlist_ips = set()

NORMAL_THR = 0.30
KNOWN_THR = 0.85

def notice(text, kind="info"):
    st.markdown(f'<div class="notice {kind}">{html_lib.escape(text)}</div>', unsafe_allow_html=True)


def html_table(df):
    """Render a DataFrame as a light, palette-matched HTML table."""
    table = df.to_html(
        index=False, escape=True, border=0,
        float_format=lambda x: f"{x:.3f}",
    ).replace("\n", "")
    st.markdown(f'<div class="tbl-wrap">{table}</div>', unsafe_allow_html=True)


def as_dict(raw):
    """Coerce whatever to_raw_values returns into a plain dict."""
    if raw is None:
        return {}
    if isinstance(raw, pd.DataFrame):
        return raw.iloc[0].to_dict()
    if isinstance(raw, pd.Series):
        return raw.to_dict()
    if isinstance(raw, dict):
        return raw
    return {}


def build_drivers(expl, raw_values):
    """Top SHAP contributors as a table. Returns an empty frame on any shape surprise."""
    try:
        vals = np.asarray(expl.values)
        if vals.ndim > 1:
            vals = vals.reshape(-1, vals.shape[-1])[0] if vals.shape[-1] == len(feature_names) else vals.reshape(-1)
        names = list(expl.feature_names) if expl.feature_names is not None else feature_names
        df = pd.DataFrame({"Feature": names, "SHAP contribution": vals})
        df["Observed value"] = [str(raw_values.get(n, "-")) for n in names]
        df["Pushes toward"] = np.where(df["SHAP contribution"] >= 0, "Attack", "Normal")
        df = df.reindex(df["SHAP contribution"].abs().sort_values(ascending=False).index)
        return df[["Feature", "Observed value", "SHAP contribution", "Pushes toward"]].head(12).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def render_waterfall(expl):
    shap.plots.waterfall(expl, show=False)
    fig = plt.gcf()
    fig.set_size_inches(7, 5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def analyze(flow_df, raw_values, source, flow_label, normal_thr, known_thr, check_normal):
    """Run the full detection + explanation + mitigation chain for one flow."""
    raw_values_dict = as_dict(raw_values)
    if source in st.session_state.allowlist_ips:
        prob = 0.0
        result = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "source": source,
            "flow": flow_label,
            "prob": prob,
            "normal_thr": normal_thr,
            "known_thr": known_thr,
            "rule": "Bypassed: Source IP is on the SOC Allowlist.",
            "drivers": pd.DataFrame(),
            "shap_png": None,
            "level": "clear",
            "origin": "Allowlist Override",
            "severity": "None",
            "raw": pd.DataFrame({"Feature": list(raw_values_dict.keys()), "Value": [str(v) for v in raw_values_dict.values()]}),
            "revoked": False,
            "alert_id": None
        }
        st.session_state.result = result
        st.success(f"Traffic from {source} automatically permitted (SOC Allowlist).")
        return
    prob = float(model.predict_proba(flow_df)[0][1])

    result = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": source,
        "flow": flow_label,
        "prob": prob,
        "normal_thr": normal_thr,
        "known_thr": known_thr,
        "rule": "",
        "drivers": pd.DataFrame(),
        "shap_png": None,
        "raw": pd.DataFrame(
            {"Feature": list(raw_values_dict.keys()), "Value": [str(v) for v in raw_values_dict.values()]}
        ),
        "revoked": False,
        "alert_id": None
    }

    if check_normal and prob < normal_thr:
        result["level"] = "clear"
        result["origin"] = "-"
        result["severity"] = "None"
    else:
        expl, snort_rule = analyze_and_mitigate(
            model, flow_df, feature_names, raw_values=raw_values_dict, explainer=explainer
        )
        result["rule"] = snort_rule
        result["origin"] = "Known dataset anomaly" if prob >= known_thr else "Zero-day anomaly"
        result["level"] = "watch" if snort_rule.startswith("#") else "alert"
        result["severity"] = "Benign volume spike" if result["level"] == "watch" else "Harmful"
        result["drivers"] = build_drivers(expl, raw_values_dict)
        result["shap_png"] = render_waterfall(expl)
    st.session_state.ledger.append(
        {
            "Time": result["time"],
            "Source": source,
            "Flow": flow_label,
            "Confidence": f"{prob * 100:.2f}%",
            "Severity": result["severity"],
            "Origin": result["origin"],
            "Status": "Rule active" if result["level"] == "alert" else "No rule issued",
        }
    )
    result["ledger_idx"] = len(st.session_state.ledger) - 1
    if db_conn and cursor:
        try:
            features_json = json.dumps(raw_values_dict)
            cursor.execute("""
                INSERT INTO Threat_Logs (source_ip, destination_ip, confidence_score, threat_severity, anomaly_origin, raw_features) 
                VALUES (:1, :2, :3, :4, :5, :6) RETURNING alert_id INTO :alert_id
            """, (source, "Target_Interface", prob, result["severity"], result["origin"], features_json))
            
            alert_id = cursor.var(oracledb.NUMBER)
            result["alert_id"] = alert_id.getvalue()[0]
            
            if result["rule"] and not result["rule"].startswith("#"):
                cursor.execute("""
                    INSERT INTO Generated_Rules (alert_id, snort_syntax) 
                    VALUES (:1, :2)
                """, (result["alert_id"], result["rule"]))
                
            db_conn.commit()
        except Exception as e:
            print(f"Failed to log to database: {e}")

    st.session_state.result = result

def revoke_rule():
    """Overrides a false positive, adds to allowlist and flags for MLOps retraining."""
    r = st.session_state.result
    if r and not r["revoked"]:
        r["revoked"] = True
        st.session_state.ledger[r["ledger_idx"]]["Status"] = "Revoked (false positive)"
        st.session_state.allowlist_ips.add(r["source"])
        if r.get("alert_id") is not None:
            try:
                cursor.execute("""
                    UPDATE Threat_Logs 
                    SET threat_severity = 'False Positive' 
                    WHERE alert_id = :1
                """, (r["alert_id"],))
                db_conn.commit()
            except Exception as e:
                print(f"Failed to update database: {e}")


def gauge_html(prob, normal_thr, known_thr):
    a, b, c = normal_thr * 100, (known_thr - normal_thr) * 100, (1 - known_thr) * 100
    pos = min(max(prob, 0.0), 1.0) * 100
    return (
        f'<div class="gauge">'
        f'<div class="z c" style="width:{a:.1f}%"></div>'
        f'<div class="z w" style="width:{b:.1f}%"></div>'
        f'<div class="z a" style="width:{c:.1f}%"></div>'
        f'<div class="pin" style="left:{pos:.1f}%"></div></div>'
        f'<div class="gauge-labels">'
        f'<span style="width:{a:.1f}%">Normal</span>'
        f'<span style="width:{b:.1f}%">Unfamiliar pattern</span>'
        f'<span style="width:{c:.1f}%">Known attack</span></div>'
    )


VERDICTS = {
    "clear": (
        "Traffic looks normal",
        "The model found nothing in this capture that resembles an attack. No rule was generated.",
    ),
    "alert": (
        "Harmful anomaly: action required",
        "The model's explanation maps to packet-header conditions, so a deployable firewall rule was generated.",
    ),
    "watch": (
        "Unusual traffic volume: no header-level threat",
        "The anomaly depends on rate and volume features that a packet-header rule cannot block safely. Review before acting.",
    ),
}


def render_verdict(r):
    title, lede = VERDICTS[r["level"]]
    st.markdown(
        f'<div class="verdict {r["level"]}">'
        f"<h2>{title}</h2>"
        f'<p class="lede">{lede}</p>'
        f'{gauge_html(r["prob"], r["normal_thr"], r["known_thr"])}'
        f'<div class="facts">'
        f'<div><span>Threat severity</span><strong>{r["severity"]}</strong></div>'
        f'<div><span>Anomaly origin</span><strong>{r["origin"]}</strong></div>'
        f'<div><span>Model confidence</span><strong>{r["prob"] * 100:.2f}%</strong></div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


def render_report(r):
    st.caption(f'{r["source"]} · {r["flow"]} · analyzed at {r["time"]}')
    render_verdict(r)
    st.write("")

    if r["level"] == "clear":
        st.markdown('<h4 class="center-h">Extracted flow features</h4>', unsafe_allow_html=True)
        html_table(r["raw"])
        return

    tab_why, tab_rule, tab_flow = st.tabs(["Why it was flagged", "Firewall rule", "Flow features"])

    # Waterfall on top, contributing features below, both centered
    with tab_why:
        _, mid, _ = st.columns([1, 3, 1])
        with mid:
            st.markdown('<h4 class="center-h">SHAP waterfall</h4>', unsafe_allow_html=True)
            st.markdown(
                '<p class="center">Each bar shows how much a feature moved the prediction '
                "away from the model's baseline.</p>",
                unsafe_allow_html=True,
            )
            st.image(r["shap_png"], use_container_width=True)

            st.markdown('<h4 class="center-h">Top contributing features</h4>', unsafe_allow_html=True)
            if r["drivers"].empty:
                st.markdown(
                    '<p class="center">Contribution table unavailable for this explanation shape.</p>',
                    unsafe_allow_html=True,
                )
            else:
                html_table(r["drivers"])

    with tab_rule:
        rule = r["rule"]
        if r["revoked"]:
            notice("Rule revoked. This anomaly was recategorized as not harmful and recorded in the ledger.")
            rule = "# REVOKED BY ANALYST (false positive)\n" + "\n".join("# " + ln for ln in rule.splitlines())
        elif r["level"] == "watch":
            notice(
                "No packet-header rule can be derived. The signal comes from volume or rate features, "
                "which would block legitimate traffic if turned into a rule.",
                "warn",
            )
        else:
            notice("SHAP attributes were translated into Snort syntax.", "success")

        st.code(rule, language=None)

        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            st.download_button(
                "Download rule",
                data=rule,
                file_name="generated.rules",
                mime="text/plain",
                disabled=r["level"] != "alert" or r["revoked"],
                use_container_width=True,
            )
        with c2:
            st.button(
                "Mark as false positive",
                on_click=revoke_rule,
                disabled=r["level"] != "alert" or r["revoked"],
                use_container_width=True,
            )

    with tab_flow:
        html_table(r["raw"])


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    f"""
<div class="app-head">
  <div>
    <h1>Closed-Loop NIDS</h1>
    <p>Detects anomalous network flows, explains them, and drafts a firewall rule.</p>
  </div>
  <div class="status">
    <div><b>{len(feature_names)}</b>features</div>
    <div><b>{len(X_test):,}</b>test flows</div>
    <div><b>{len(flagged_idx):,}</b>flagged by model</div>
    <div><b>{len(st.session_state.ledger)}</b>analyzed this session</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Traffic source")
    mode = st.radio(
        "Input method",
        ["Test set flows", "Upload .pcap"],
        label_visibility="collapsed",
    )

    st.divider()

    if mode == "Test set flows":
        st.caption("Only flows the model flags as malicious are listed, so every selection triggers an alert.")
        options = list(flagged_idx[:50])
        selected_idx = st.selectbox(
            "Flagged flow",
            options,
            format_func=lambda i: f"Flow {i}  ({test_proba[i] * 100:.1f}%)",
        )
        if st.button("Analyze flow", type="primary", use_container_width=True):
            with st.spinner("Scoring flow, computing SHAP values, drafting rule..."):
                packet_df = X_test.iloc[[selected_idx]]
                raw_vals = to_raw_values(pipeline, packet_df)
                analyze(
                    packet_df, raw_vals, "Test set", f"Flow {selected_idx}",
                    NORMAL_THR, KNOWN_THR, check_normal=False,
                )

    else:
        if not PCAP_MODULE_READY:
            notice("The pcap parser (Module B) is not linked yet.", "warn")
        else:
            uploaded = st.file_uploader("Capture file", type=["pcap", "pcapng"])
            if uploaded is not None and st.button("Analyze capture", type="primary", use_container_width=True):
                with st.spinner("Parsing packets and scoring flows..."):
                    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
                        tmp.write(uploaded.getbuffer())
                        tmp_path = tmp.name

                    raw_df = extract_flow_features(tmp_path)
                    num_cols = list(pipeline.transformers_[0][2])
                    scaler = pipeline.named_transformers_["num"]
                    scaled_df = raw_df.copy()
                    scaled_df[num_cols] = scaler.transform(raw_df[num_cols])

                    probs = model.predict_proba(scaled_df)[:, 1]
                    worst = int(np.argmax(probs))
                    st.session_state["pcap_note"] = (
                        f"{len(raw_df)} flow(s) extracted. Showing the highest-scoring one (flow {worst})."
                        if len(raw_df) > 1 else None
                    )
                    analyze(
                        scaled_df.iloc[[worst]],
                        raw_df.iloc[worst].to_dict(),
                        f"Capture: {uploaded.name}",
                        f"Flow {worst}",
                        NORMAL_THR, KNOWN_THR, check_normal=True,
                    )

    st.divider()
    st.caption("XGBoost classifier · SHAP TreeExplainer · rule synthesis via explainability engine")


# ----------------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------------
ledger_count = len(st.session_state.ledger)
tab_analysis, tab_ledger = st.tabs(["Analysis", f"Alert ledger ({ledger_count})"])

with tab_analysis:
    if st.session_state.result is None:
        st.markdown(
            """
<div class="empty">
  <h3>No flow analyzed yet</h3>
  <p>Choose a traffic source in the sidebar, then select <b>Analyze flow</b>.
  You'll get a verdict, the features that drove it, and a Snort rule when one can be derived.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        note = st.session_state.get("pcap_note")
        if note and st.session_state.result["source"].startswith("Capture"):
            notice(note)
        render_report(st.session_state.result)

with tab_ledger:
    if not st.session_state.ledger:
        st.markdown(
            '<div class="empty"><h3>The ledger is empty</h3>'
            "<p>Every analyzed flow is recorded here with its severity and rule status, "
            "including rules you revoke as false positives.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        ledger_df = pd.DataFrame(st.session_state.ledger).iloc[::-1]
        html_table(ledger_df)
        st.write("")
        st.download_button(
            "Export ledger as CSV",
            data=ledger_df.to_csv(index=False),
            file_name="nids_ledger.csv",
            mime="text/csv",
        )
