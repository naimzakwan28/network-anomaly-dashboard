import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import plotly.express as px
import shap

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="Network Anomaly Detection Dashboard", layout="wide")
st.title("🚨 Advanced Network Anomaly Detection")
st.write("Advanced Anomaly Analysis using CICIDS2017 with Explainable AI")

# -------------------------------------------------
# Load Models, Scaler, and Feature List
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    rf_model = joblib.load("tuned_rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("final_features.pkl")
    try:
        svm_model = joblib.load("svm_model.pkl")
    except:
        svm_model = None
    return rf_model, svm_model, scaler, features

rf_model, svm_model, scaler, trained_features = load_artifacts()

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
st.sidebar.header("🛠️ Dashboard Controls")

model_choice = st.sidebar.selectbox(
    "Select Active Model:",
    options=["Random Forest", "Support Vector Machine"] if svm_model else ["Random Forest"]
)

top_n = st.sidebar.slider("Top features to show", 5, 30, 20)

# -------------------------------------------------
# Tabs Layout
# -------------------------------------------------
tab1, tab2 = st.tabs([
    "📊 Dataset Analysis Dashboard",
    "🔎 Manual Flow Analysis"
])

# =================================================
# TAB 1 — DATASET ANALYSIS DASHBOARD
# =================================================
with tab1:
    uploaded_file = st.file_uploader(
        "📂 Upload CICIDS2017 Sample CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding="latin1", low_memory=False)
        df.columns = df.columns.str.strip()

        # Binary label
        df["BinaryLabel"] = df["Label"].astype(str).str.strip().apply(
            lambda x: 0 if x.upper() == "BENIGN" else 1
        )

        # Feature alignment
        missing_features = set(trained_features) - set(df.columns)
        for col in missing_features:
            df[col] = 0

        X = df[trained_features]
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        X_scaled = scaler.transform(X)

        # Prediction
        # Prediction probabilities / confidence
if model_choice == "Random Forest":
    y_probs = active_model.predict_proba(X_scaled)[:, 1]

elif model_choice == "Support Vector Machine":
    # LinearSVC has NO predict_proba
    decision_scores = active_model.decision_function(X_scaled)

    # Normalize decision scores to 0–1 range
    y_probs = (decision_scores - decision_scores.min()) / (
        decision_scores.max() - decision_scores.min() + 1e-9
    )


        # -------------------------------------------------
        # KPI Metrics
        # -------------------------------------------------
        st.subheader(f"📊 Global Performance: {model_choice}")

        tn, fp, fn, tp = confusion_matrix(df["BinaryLabel"], y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Flows", f"{len(df):,}")
        c2.metric("Attacks Detected", f"{np.sum(y_pred == 1):,}")
        c3.metric("Accuracy", f"{accuracy_score(df['BinaryLabel'], y_pred):.4f}")
        c4.metric("Precision", f"{precision_score(df['BinaryLabel'], y_pred):.4f}")
        c5.metric("Recall", f"{recall_score(df['BinaryLabel'], y_pred):.4f}")
        c6.metric("False Positive Rate", f"{fpr:.4f}")

        # -------------------------------------------------
        # Performance Visuals
        # -------------------------------------------------
        st.markdown("---")
        st.subheader("🖼️ Model Performance Visuals")

        v1, v2 = st.columns(2)

        with v1:
            report = classification_report(
                df["BinaryLabel"], y_pred, output_dict=True, zero_division=0
            )
            report_df = pd.DataFrame(report).transpose().iloc[:2, :3]
            st.plotly_chart(
                px.bar(report_df, barmode="group", title="Classification Report"),
                use_container_width=True
            )

        with v2:
            cm = confusion_matrix(df["BinaryLabel"], y_pred)
            st.plotly_chart(
                px.imshow(cm, text_auto=True,
                           x=["Normal", "Attack"],
                           y=["Normal", "Attack"],
                           title="Confusion Matrix"),
                use_container_width=True
            )

        # -------------------------------------------------
        # Attack Timeline
        # -------------------------------------------------
        if "Timestamp" in df.columns:
            st.markdown("---")
            st.subheader("📈 Attack Timeline")

            df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
            temp_df = df.dropna(subset=["Timestamp"]).copy()

            if not temp_df.empty:
                temp_df["IsAttack"] = y_pred[:len(temp_df)]
                timeline = temp_df.resample("1T", on="Timestamp")["IsAttack"].sum().reset_index()

                st.plotly_chart(
                    px.line(timeline, x="Timestamp", y="IsAttack",
                            title="Anomalies per Minute",
                            color_discrete_sequence=["red"]),
                    use_container_width=True
                )

        # -------------------------------------------------
        # Mini SOC Panel
        # -------------------------------------------------
        st.markdown("---")
        l, r = st.columns(2)

        with l:
            st.subheader("📋 Detection Priority")
            soc_df = pd.DataFrame({
                "Source IP": df["Source IP"] if "Source IP" in df.columns else "N/A",
                "Prediction": ["Attack" if p == 1 else "Normal" for p in y_pred],
                "Confidence": y_probs.round(4)
            })
            soc_df["Risk Level"] = pd.cut(
                soc_df["Confidence"],
                bins=[-0.1, 0.4, 0.7, 1.1],
                labels=["Low 🟢", "Medium 🟡", "High 🔴"]
            )
            st.dataframe(soc_df.head(50), use_container_width=True)

        with r:
            st.subheader("🕵️ Threat Distribution")
            st.plotly_chart(
                px.pie(values=df["Label"].value_counts().values,
                       names=df["Label"].value_counts().index,
                       hole=0.4),
                use_container_width=True
            )

        # -------------------------------------------------
        # SHAP (Random Forest only)
        # -------------------------------------------------
        if model_choice == "Random Forest":
            st.markdown("---")
            st.subheader("🧠 Explainable AI (SHAP)")

            row_idx = st.number_input("Select row index", 0, len(df) - 1, 0)
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_scaled[row_idx])

            fig, ax = plt.subplots(figsize=(10, 5))
            shap.bar_plot(shap_values[1], trained_features, max_display=10, show=False)
            st.pyplot(fig)

        # -------------------------------------------------
        # Global Feature Importance
        # -------------------------------------------------
        st.markdown("---")
        st.subheader("🔍 Global Feature Importance")

        if model_choice == "Random Forest":
            fi_df = pd.DataFrame({
                "Feature": trained_features,
                "Importance": rf_model.feature_importances_
            }).sort_values(by="Importance", ascending=False).head(top_n)

            st.bar_chart(fi_df.set_index("Feature"))

        else:
            if os.path.exists("svm_feature_importance.csv"):
                svm_df = pd.read_csv("svm_feature_importance.csv") \
                            .sort_values(by="AbsWeight", ascending=False) \
                            .head(top_n)
                st.bar_chart(svm_df.set_index("Feature")["AbsWeight"])
            else:
                st.warning("svm_feature_importance.csv not found.")

    else:
        st.info("📌 Please upload a CICIDS2017 CSV file to begin analysis.")

# =================================================
# TAB 2 — MANUAL FLOW ANALYSIS (NO DATASET KPIs)
# =================================================
with tab2:
    st.subheader("🔎 Manual Flow Analysis (What-If Prediction)")
    st.write("Manually input a single network flow to predict anomaly behavior.")

    with st.form("manual_flow_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            flow_duration = st.number_input("Flow Duration", 0.0, value=1000.0)
            pkt_len_mean = st.number_input("Packet Length Mean", 0.0, value=500.0)
            flow_iat_mean = st.number_input("Flow IAT Mean", 0.0, value=200.0)

        with c2:
            pkt_len_var = st.number_input("Packet Length Variance", 0.0, value=1000.0)
            flow_iat_max = st.number_input("Flow IAT Max", 0.0, value=500.0)
            fwd_pkt_len_max = st.number_input("Fwd Packet Length Max", 0.0, value=1500.0)

        with c3:
            idle_mean = st.number_input("Idle Mean", 0.0, value=0.0)
            idle_max = st.number_input("Idle Max", 0.0, value=0.0)
            protocol = st.selectbox("Protocol", [6, 17],
                                    format_func=lambda x: "TCP (6)" if x == 6 else "UDP (17)")

        submit = st.form_submit_button("🔍 Predict Flow")

if submit:
    # -------------------------------------------------
    # Step 1: Create base manual input
    # -------------------------------------------------
    manual_data = {f: 0 for f in trained_features}

    manual_data.update({
        "Flow Duration": flow_duration,
        "Packet Length Mean": pkt_len_mean,
        "Packet Length Variance": pkt_len_var,
        "Flow IAT Mean": flow_iat_mean,
        "Flow IAT Max": flow_iat_max,
        "Fwd Packet Length Max": fwd_pkt_len_max,
        "Idle Mean": idle_mean,
        "Idle Max": idle_max,
        "Protocol": protocol
    })

    # -------------------------------------------------
    # Step 2: Derived Flood / Attack-like Features
    # -------------------------------------------------
    if "Flow Bytes/s" in trained_features:
        manual_data["Flow Bytes/s"] = pkt_len_mean * 1000

    if "Flow Packets/s" in trained_features:
        manual_data["Flow Packets/s"] = 2000

    if "Total Fwd Packets" in trained_features:
        manual_data["Total Fwd Packets"] = 500

    if "Total Backward Packets" in trained_features:
        manual_data["Total Backward Packets"] = 0

        
        manual_df = pd.DataFrame([manual_data])[trained_features]
        manual_scaled = scaler.transform(manual_df)

        model = rf_model if model_choice == "Random Forest" else svm_model
        pred = model.predict(manual_scaled)[0]

        try:
            conf = model.predict_proba(manual_scaled)[0][1]
        except:
            score = model.decision_function(manual_scaled)[0]
            conf = (score - score.min()) / (score.max() - score.min())

        risk = "Low 🟢" if conf < 0.2 else "Medium 🟡" if conf < 0.5 else "High 🔴"

        st.markdown("### 🧾 Prediction Result")
        a, b, c = st.columns(3)
        a.metric("Prediction", "Attack 🚨" if pred == 1 else "Normal ✅")
        b.metric("Confidence", f"{conf:.4f}")

        c.metric("Risk Level", risk)

