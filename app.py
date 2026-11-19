"""NeuroGuard: local beginner-friendly Streamlit dashboard (no packet capture)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core import FEATURES, check_features, predict_frame
from neural_visualizer import inspect_forward_pass, render_animation_html
from train_model import METRICS_PATH, MODEL_PATH

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data" / "sample_flows.csv"

st.set_page_config(page_title="NeuroGuard IDS", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.8rem; max-width: 1200px;}
.hero {padding: 1.25rem 1.5rem; border-radius: 16px;
       background: linear-gradient(110deg, #102a43 0%, #124e66 65%, #0b6a70 100%);
       color: #fff; margin-bottom: 1.2rem;}
.hero h1 {font-size: 2.3rem; margin: 0 0 .3rem 0; color: white;}
.hero p {margin: 0; color: #dbf4f4; font-size: 1.03rem;}
.small {font-size: .88rem; color: #64748b;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading your trained neural network...")
def load_trained_model(modified_time: float):
    # Only loads the artifact generated locally by our own training script.
    return joblib.load(MODEL_PATH)["model"]


@st.cache_data

def load_sample():
    return pd.read_csv(SAMPLE)


def model_score(model, frame):
    return predict_frame(model, frame)


st.markdown('<div class="hero"><h1>🛡️ NeuroGuard IDS</h1>'
            '<p>AI-powered network-flow classification · Beginner laboratory project</p></div>',
            unsafe_allow_html=True)
st.warning("**Education-only demonstration:** the default model is trained on fictional network-flow data. "
           "Predictions are not reliable real-world intrusion alerts. No packets are captured and no ports are scanned.",
           icon="⚠️")

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("The trained model is missing. Create it with the following commands first:")
    st.code("python generate_data.py\npython train_model.py", language="powershell")
    st.info("For Windows 11, you can also double-click RUN_WINDOWS.bat.")
    st.stop()

try:
    model = load_trained_model(MODEL_PATH.stat().st_mtime)
    report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
except Exception as error:
    st.error(f"Unable to open the model files: {error}")
    st.info("Try running python train_model.py again. Only open model files you created yourself.")
    st.stop()

page = st.sidebar.radio(
    "Go to", ["📊 Overview", "🔍 Check one flow", "📁 Analyze a CSV",
              "📈 Model evaluation", "🎞️ Neural network animation", "🧠 How the AI works"]
)
st.sidebar.divider()
st.sidebar.caption("Model: MLP neural network (32 → 16 hidden neurons)")
st.sidebar.caption("Mode: offline CSV demonstration, no live network access")

if page == "📊 Overview":
    st.subheader("Project overview")
    first, second, third, fourth = st.columns(4)
    first.metric("Training records", f"{report['train_rows']:,}")
    second.metric("Test records", f"{report['test_rows']:,}")
    third.metric("Demo accuracy", f"{report['accuracy']:.1%}")
    fourth.metric("Demo F1 score", f"{report['f1']:.1%}")
    st.caption("Accuracy and F1 are measured only against held-out rows from a similar labeled dataset.")
    if SAMPLE.exists():
        sample = load_sample()
        results = model_score(model, sample)
        left, right = st.columns([1, 2])
        with left:
            st.markdown("#### Built-in example classifications")
            counts = results["prediction"].value_counts().reindex(["Normal", "Suspicious"], fill_value=0)
            st.bar_chart(counts)
        with right:
            st.markdown("#### Sample records")
            st.dataframe(results[["protocol", "dst_port", "connections_per_minute",
                                  "failed_logins", "prediction", "suspicious_score"]],
                         use_container_width=True, hide_index=True)
    st.info("Go to **Check one flow** to change individual values, or **Analyze a CSV** to classify multiple rows.")

elif page == "🔍 Check one flow":
    st.subheader("Check one network-flow summary")
    st.write("Enter summary statistics for a *fictional or authorized* network flow. "
             "These are input features, not commands sent to a network.")
    with st.form("flow_input"):
        a, b, c = st.columns(3)
        with a:
            protocol = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"])
            dst_port = st.number_input("Destination port", 0, 65535, 443, step=1)
            duration_ms = st.number_input("Duration (milliseconds)", 0, 100000000, 1200, step=50)
        with b:
            src_bytes = st.number_input("Source bytes", 0, 1000000000, 1100, step=100)
            dst_bytes = st.number_input("Destination bytes", 0, 1000000000, 2400, step=100)
            packets = st.number_input("Packets in the flow", 0, 100000000, 22, step=1)
        with c:
            syn_count = st.number_input("TCP SYN count", 0, 100000000, 1, step=1)
            failed_logins = st.number_input("Failed login count", 0, 100000000, 0, step=1)
            cpm = st.number_input("Connections per minute", 0, 100000000, 8, step=1)
        submitted = st.form_submit_button("Analyze this flow", use_container_width=True)
    if submitted:
        frame = pd.DataFrame([{
            "protocol": protocol, "duration_ms": duration_ms, "dst_port": dst_port,
            "src_bytes": src_bytes, "dst_bytes": dst_bytes, "packets": packets,
            "syn_count": syn_count, "failed_logins": failed_logins,
            "connections_per_minute": cpm,
        }])
        result = model_score(model, frame).iloc[0]
        label = result["prediction"]
        st.markdown("### Result")
        m1, m2 = st.columns(2)
        m1.metric("AI classification", label)
        m2.metric("Suspicious-class model score", f"{result['suspicious_score']:.1%}")
        st.progress(float(result["suspicious_score"]))
        if label == "Suspicious":
            st.warning("This input resembles the synthetic suspicious examples used for training.")
        else:
            st.success("This input resembles the synthetic normal examples used for training.")
        st.caption("This score is not a calibrated likelihood of an actual cyberattack.")

elif page == "📁 Analyze a CSV":
    st.subheader("Batch classification from CSV")
    st.write("Use the included sample dataset or upload flow summaries with the same nine columns.")
    mode = st.radio("Data source", ["Use included sample file", "Upload my CSV"], horizontal=True)
    if mode == "Use included sample file":
        frame = load_sample() if SAMPLE.exists() else None
    else:
        uploaded = st.file_uploader("Choose a .csv file (10 MB maximum)", type=["csv"])
        if uploaded is None:
            st.info("Choose a file to see its predictions.")
            st.stop()
        if uploaded.size > 10 * 1024 * 1024:
            st.error("File exceeds this demonstration's 10 MB limit.")
            st.stop()
        try:
            frame = pd.read_csv(uploaded)
        except Exception as error:
            st.error(f"Cannot read CSV: {error}")
            st.stop()
    if frame is None or frame.empty:
        st.error("CSV is empty or missing.")
        st.stop()
    if len(frame) > 20000:
        st.error("For this beginner dashboard, limit each CSV to 20,000 rows.")
        st.stop()
    try:
        check_features(frame)
        results = model_score(model, frame)
    except ValueError as error:
        st.error(f"Invalid CSV schema: {error}")
        st.code(",".join(FEATURES), language="text")
        st.stop()
    normal = int((results["prediction"] == "Normal").sum())
    suspicious = int((results["prediction"] == "Suspicious").sum())
    x, y, z = st.columns(3)
    x.metric("Rows analyzed", len(results))
    y.metric("Classified normal", normal)
    z.metric("Classified suspicious", suspicious)
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button("Download CSV with predictions", results.to_csv(index=False).encode("utf-8"),
                       file_name="neuroguard_predictions.csv", mime="text/csv", use_container_width=True)
    st.caption("The downloadable predictions are for classroom demonstration, not incident response.")

elif page == "📈 Model evaluation":
    st.subheader("What happened during training?")
    st.write(f"Dataset: `{report['data_source']}` · Total records: {report['dataset_rows']:,} · "
             f"Held-out test records: {report['test_rows']:,}")
    cols = st.columns(4)
    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1"]):
        col.metric(key.capitalize(), f"{report[key]:.1%}")
    st.markdown("#### Confusion matrix")
    cm = report["confusion_matrix"]
    confusion = pd.DataFrame(
        cm,
        index=["Actual normal", "Actual suspicious"],
        columns=["Predicted normal", "Predicted suspicious"],
    )
    st.dataframe(confusion, use_container_width=True)
    st.markdown("- **Accuracy:** portion of correct labels in the test data.\n"
                "- **Precision:** of rows flagged suspicious, the portion that really have that label.\n"
                "- **Recall:** of rows labeled suspicious, the portion found by the model.\n"
                "- **F1 score:** balance of precision and recall.")
    st.warning("Even a very high result on synthetic data does **not** mean this is ready to protect a real network.")

elif page == "🎞️ Neural network animation":
    st.subheader("🎞️ Watch the trained neural network work")
    st.write("Select one example from the included CSV. The animation shows actual activation values "
             "from your trained model, moving from the input layer to the final prediction.")
    if not SAMPLE.exists():
        st.error("Sample data not found. Run python generate_data.py first.")
        st.stop()
    examples = load_sample()
    choices = list(range(min(40, len(examples))))
    if not choices:
        st.warning("The example CSV is empty.")
        st.stop()
    selected = st.selectbox(
        "Choose a flow record",
        choices,
        format_func=lambda i: (f"Example {i+1} · {examples.iloc[i]['protocol']} "
                               f"· {int(examples.iloc[i]['connections_per_minute'])} connections/minute"),
    )
    chosen = examples.iloc[[selected]]
    try:
        details = inspect_forward_pass(model, chosen)
    except ValueError as error:
        st.error(f"Could not visualize this record: {error}")
        st.stop()
    a, b = st.columns(2)
    a.metric("AI prediction", details["label"])
    b.metric("Suspicious-class model score", f"{details['suspicious_score']:.1%}")
    components.html(render_animation_html(details), height=855, scrolling=False)
    st.caption("Replay, pause, and change speed inside the animation. The colors show real "
               "neuron activations, but the animation timing is a teaching illustration. "
               "It is not a live traffic capture or a backpropagation/training animation.")
    with st.expander("See the actual numbers used by the animation"):
        for i, activations in enumerate(details["layers"]):
            label = ["Encoded input", "Hidden layer 1 (ReLU)", "Hidden layer 2 (ReLU)", "Sigmoid output"][i]
            st.markdown(f"**{label}**")
            names = details["feature_names"] if i == 0 else [f"Neuron {j+1}" for j in range(len(activations))]
            st.dataframe(pd.DataFrame({"Unit": names, "Activation": activations}),
                         hide_index=True, use_container_width=True)

else:
    st.subheader("🧠 How the neural network works")
    st.markdown("""
1. **Input:** one row contains 9 flow fields (protocol, bytes, packets, counts, etc.).
2. **Prepare:** numerical fields are standardized; `protocol` is one-hot encoded.
3. **Hidden layer 1:** 32 artificial neurons combine the input values.
4. **Hidden layer 2:** 16 neurons learn more complex combinations of patterns.
5. **Output:** a score for the `Suspicious` class.
6. **Training:** the model adjusts its weights repeatedly to reduce classification error.

The classifier is a **multi-layer perceptron (MLP)** trained with backpropagation and the Adam optimizer. It does not read raw packet payloads and does not know the identity of an attacker.
""")
    st.code("Original 9 features -> preprocessing -> 32 neurons -> 16 neurons -> score [0..1]", language="text")
    st.markdown("#### Input CSV schema")
    explanation = pd.DataFrame([
        ["protocol", "TCP, UDP, or ICMP"],
        ["duration_ms", "Flow duration in milliseconds"],
        ["dst_port", "Destination port (0–65535)"],
        ["src_bytes", "Bytes from source"],
        ["dst_bytes", "Bytes to source"],
        ["packets", "Number of packets"],
        ["syn_count", "Count of TCP SYN flags, if applicable"],
        ["failed_logins", "Failed login attempts summarized for this event"],
        ["connections_per_minute", "Recent connections observed per minute"],
    ], columns=["Field", "Meaning"])
    st.dataframe(explanation, hide_index=True, use_container_width=True)
    st.write("For a new model, supply a **labeled** CSV with those fields plus a `label` "
             "column (0=Normal, 1=Suspicious), then run `python train_model.py --data your_file.csv`. ")
    st.caption("For real experiments, ensure labels and flow features are reliable, use a separate test set, "
               "and do not assume synthetic accuracy carries over to real traffic.")
