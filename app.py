import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn


# ── Model Architecture (must match training) ──────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    config = joblib.load("model_config.pkl")
    scaler = joblib.load("scaler.pkl")
    iso_forest = joblib.load("isolation_forest.pkl")

    ae = Autoencoder(config["input_dim"])
    ae.load_state_dict(torch.load("autoencoder.pth", map_location="cpu"))
    ae.eval()

    return scaler, iso_forest, ae, config


scaler, iso_forest, autoencoder, config = load_models()
threshold = config["threshold"]
feature_cols = config["feature_cols"]


# ── Prediction Helpers ────────────────────────────────────────────────────────
def predict(X):
    iso_preds = (iso_forest.predict(X) == -1).astype(int)
    tensor = torch.FloatTensor(X)
    with torch.no_grad():
        reconstructed = autoencoder(tensor)
        mse = torch.mean((tensor - reconstructed) ** 2, dim=1).numpy()
    ae_preds = (mse > threshold).astype(int)
    return iso_preds, ae_preds, mse


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Network Anomaly Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Network Intrusion Anomaly Detector")
st.markdown(
    """
Detects malicious network traffic using two unsupervised anomaly detection models trained on the NSL-KDD benchmark:  
**Isolation Forest** (classical ML) vs **Autoencoder** (deep learning, PyTorch)
"""
)

tab1, tab2 = st.tabs(["📁 File Upload", "🔍 Sample Explorer"])

# ── TAB 1: File Upload ────────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload Network Traffic Data")
    st.markdown(
        "Upload a CSV with NSL-KDD features. The app will flag anomalous connections using both models."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df):,} connections")

            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                X_scaled = scaler.transform(df[feature_cols].values)
                iso_preds, ae_preds, mse_scores = predict(X_scaled)

                df["Isolation Forest"] = [
                    "🚨 Attack" if p else "✅ Normal" for p in iso_preds
                ]
                df["Autoencoder"] = [
                    "🚨 Attack" if p else "✅ Normal" for p in ae_preds
                ]
                df["Anomaly Score"] = mse_scores.round(4)

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Connections", f"{len(df):,}")
                col2.metric(
                    "Flagged by Isolation Forest",
                    f"{iso_preds.sum():,}",
                    f"{iso_preds.mean()*100:.1f}%",
                )
                col3.metric(
                    "Flagged by Autoencoder",
                    f"{ae_preds.sum():,}",
                    f"{ae_preds.mean()*100:.1f}%",
                )

                st.divider()
                filter_opt = st.radio(
                    "Show:", ["All", "Attacks only", "Normal only"], horizontal=True
                )
                if filter_opt == "Attacks only":
                    display_df = df[ae_preds == 1]
                elif filter_opt == "Normal only":
                    display_df = df[ae_preds == 0]
                else:
                    display_df = df

                st.dataframe(
                    display_df[
                        ["Isolation Forest", "Autoencoder", "Anomaly Score"]
                        + feature_cols[:5]
                    ],
                    use_container_width=True,
                )

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results", csv, "anomaly_results.csv", "text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info(
            "Upload a raw NSL-KDD CSV file. Download the dataset from Kaggle and upload a subset of KDDTrain+.txt or KDDTest+.txt."
        )

# ── TAB 2: Sample Explorer ────────────────────────────────────────────────────
with tab2:
    st.subheader("Explore Individual Connections")
    st.markdown(
        "Browse network connections from the test set and see how each model classifies them."
    )

    try:
        test_df = pd.read_csv("test_samples.csv")

        col1, col2 = st.columns([1, 2])

        with col1:
            sample_type = st.radio(
                "Sample type:", ["Random", "Known Attack", "Known Normal"]
            )

            if sample_type == "Known Attack":
                pool = test_df[test_df["true_label"] == 1]
            elif sample_type == "Known Normal":
                pool = test_df[test_df["true_label"] == 0]
            else:
                pool = test_df

            if st.button("🎲 Get Sample", use_container_width=True):
                st.session_state["sample"] = pool.sample(1).iloc[0].to_dict()

            if "sample" not in st.session_state:
                st.session_state["sample"] = pool.sample(1).iloc[0].to_dict()

            sample = st.session_state["sample"]
            X = np.array([sample[f] for f in feature_cols]).reshape(1, -1)

            iso_pred = (iso_forest.predict(X) == -1)[0]
            tensor = torch.FloatTensor(X)
            with torch.no_grad():
                recon = autoencoder(tensor)
                mse = torch.mean((tensor - recon) ** 2).item()
            ae_pred = mse > threshold
            true_label = int(sample["true_label"])

            st.divider()
            st.markdown("**Ground Truth:**")
            if true_label == 1:
                st.error("🚨 Attack")
            else:
                st.success("✅ Normal")
            st.markdown("**Isolation Forest:**")
            if iso_pred:
                st.error("🚨 Flagged as Attack")
            else:
                st.success("✅ Classified Normal")

            st.markdown("**Autoencoder:**")
            if ae_pred:
                st.error("🚨 Flagged as Attack")
            else:
                st.success("✅ Classified Normal")

            st.metric("Anomaly Score (MSE)", f"{mse:.4f}")
            st.caption(f"Threshold: {threshold:.4f}")
            st.progress(min(float(mse / (threshold * 2)), 1.0))

        with col2:
            st.markdown("**Connection Features:**")
            feature_display = pd.DataFrame(
                {
                    "Feature": feature_cols,
                    "Value": [round(sample[f], 4) for f in feature_cols],
                }
            )
            st.dataframe(feature_display, use_container_width=True, height=500)

    except FileNotFoundError:
        st.error("test_samples.csv not found. Run the notebook first to generate it.")

st.divider()
st.caption(
    "Built with PyTorch · Scikit-learn · NSL-KDD Dataset · Unsupervised Anomaly Detection"
)
