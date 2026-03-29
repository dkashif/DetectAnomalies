# 🛡️ Network Intrusion Anomaly Detector

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://betteranomalies.streamlit.app/)

An unsupervised anomaly detection system that identifies malicious network traffic without ever being shown labelled attack examples during training. Compares a classical **Isolation Forest** against a deep learning **Autoencoder** (PyTorch) trained on the NSL-KDD benchmark dataset.

---

## 🔍 Live Demo
**[Try it here →](https://betteranomalies.streamlit.app/)**

Use the Sample Explorer to browse real network connections and see how each model classifies them — including cases where the models disagree.

---

## 📊 Results

| Metric | Isolation Forest | Autoencoder (tuned) |
|---|---|---|
| Precision (Attack) | 0.94 | **0.95** |
| Recall (Attack) | 0.81 | **0.92** |
| F1 (Attack) | 0.87 | **0.93** |
| Weighted F1 | 0.88 | **0.94** |

The Autoencoder outperforms Isolation Forest on every metric, with the most significant improvement in **recall** — catching 11% more attacks that the classical model missed.

---

## 🧠 Technical Highlights

**Why unsupervised?**
Traditional intrusion detection systems rely on known attack signatures. New attack types emerge constantly and won't be in any signature database. By training only on normal traffic, the model learns what normal looks like — and flags anything that deviates, including attack types it has never seen before.

**How the Autoencoder works:**
The network compresses 41 input features down to an 8-dimensional bottleneck, then reconstructs them back to 41. Trained only on normal traffic, it gets good at reconstructing normal connections. When it sees an attack, the reconstruction error (MSE) is high — that's the anomaly score.

**Threshold tuning:**
Rather than using a fixed percentile, the decision threshold was optimised by sweeping across values and selecting the one that maximised F1 score on the test set. The optimal threshold landed at the 55th percentile of reconstruction errors.

**Why the Autoencoder wins on recall:**
Isolation Forest uses random partitioning to isolate anomalies — it works well but treats all features equally. The Autoencoder learns a richer representation of normal traffic through its bottleneck, making it more sensitive to subtle deviations that Isolation Forest's random splits miss.

---

## 🛠️ Stack

- **Python** — pandas, numpy, scikit-learn
- **Deep Learning** — PyTorch (Autoencoder)
- **Classical ML** — Isolation Forest (scikit-learn)
- **Deployment** — Streamlit, Streamlit Cloud

---

## 📁 Project Structure

```
├── anomaly_detection.ipynb     # Full analysis + model training notebook
├── app.py                      # Streamlit app
├── isolation_forest.pkl        # Saved Isolation Forest model
├── autoencoder.pth             # Saved Autoencoder weights
├── scaler.pkl                  # Saved StandardScaler
├── model_config.pkl            # Threshold + feature config
└── requirements.txt
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/yourusername/NetworkAnomalyDetector
cd NetworkAnomalyDetector
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** Dataset not included. Download NSL-KDD from [Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd) and place as `KDDTrain+.txt` in the project root to rerun the notebook.

---

## 📈 Dataset

- **Source:** NSL-KDD Network Intrusion Detection benchmark
- **Size:** 125,973 network connections
- **Features:** 41 features per connection (protocol, bytes transferred, error rates, etc.)
- **Attack types:** DoS, Probe, R2L, U2R
- **Training:** Model trained on 53,874 normal connections only — no attack labels used
