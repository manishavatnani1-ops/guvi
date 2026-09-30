"""
💎 Diamond Price Prediction & Market Segmentation — Streamlit App
-------------------------------------------------------------------
Loads the artifacts produced by `diamond_price_prediction_updated.ipynb`:
    - pipeline.pkl        (StandardScaler + XGBRegressor, predicts sqrt(price_usd))
    - encoder.pkl         (OrdinalEncoder fitted on cut/color/clarity)
    - cluster_model.pkl   (KMeans, 3 clusters)
    - cluster_scaler.pkl  (StandardScaler used before KMeans)

Both the price model and the cluster model were trained on the SAME feature
set/order: ['carat','cut','color','clarity','depth','table','x','y','z'],
where carat/x/y/z were sqrt-transformed in the notebook before training
(to fix right-skew) and cut/color/clarity were ordinal-encoded.
This app reproduces that exact preprocessing for any new user input.
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Diamond Price & Market Segment Predictor",
    page_icon="💎",
    layout="centered",
)

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
USD_TO_INR = 83  # same conversion rate used in the notebook

# Human-readable names for each KMeans cluster.
# Derived from the actual cluster means on the training data:
#   cluster 0 -> avg carat ~1.54, avg price ~$9,753  -> large & expensive
#   cluster 1 -> avg carat ~0.39, avg price ~$1,035   -> small & affordable
#   cluster 2 -> avg carat ~0.89, avg price ~$4,345   -> medium / mid-range
# If you retrain the KMeans model, re-check these mappings — cluster
# index-to-meaning can shift between training runs.
CLUSTER_NAMES = {
    0: "Premium Heavy Diamonds",
    1: "Affordable Small Diamonds",
    2: "Mid-Range Diamonds",
}

FEATURE_ORDER = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]


# ------------------------------------------------------------------
# LOAD ARTIFACTS (cached so they only load once per session)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("pipeline.pkl")
    encoder = joblib.load("encoder.pkl")
    cluster_model = joblib.load("cluster_model.pkl")
    cluster_scaler = joblib.load("cluster_scaler.pkl")
    return pipeline, encoder, cluster_model, cluster_scaler


try:
    pipeline, encoder, cluster_model, cluster_scaler = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Could not find one or more model files "
        "(pipeline.pkl, encoder.pkl, cluster_model.pkl, cluster_scaler.pkl). "
        "Make sure they are in the same folder as this app.py.\n\n"
        f"Details: {e}"
    )
    st.stop()

# Category options come straight from the fitted encoder, so the dropdowns
# always match exactly what the model was trained on.
CUT_OPTIONS = list(encoder.categories_[0])
COLOR_OPTIONS = list(encoder.categories_[1])
CLARITY_OPTIONS = list(encoder.categories_[2])


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def build_feature_row(carat, cut, color, clarity, depth, table, x, y, z):
    """Apply the same preprocessing used in training and return a 1-row
    DataFrame in the exact column order the models expect."""
    cut_enc, color_enc, clarity_enc = encoder.transform([[cut, color, clarity]])[0]

    row = pd.DataFrame(
        [{
            "carat": np.sqrt(carat),
            "cut": cut_enc,
            "color": color_enc,
            "clarity": clarity_enc,
            "depth": depth,
            "table": table,
            "x": np.sqrt(x),
            "y": np.sqrt(y),
            "z": np.sqrt(z),
        }]
    )
    return row[FEATURE_ORDER]


def predict_price_inr(row: pd.DataFrame) -> float:
    """Pipeline predicts sqrt(price_usd) — square it, then convert to INR."""
    pred_sqrt_price = pipeline.predict(row)[0]
    price_usd = pred_sqrt_price ** 2
    return price_usd * USD_TO_INR


def predict_cluster(row: pd.DataFrame):
    scaled = cluster_scaler.transform(row)
    cluster_id = int(cluster_model.predict(scaled)[0])
    cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
    return cluster_id, cluster_name


# ------------------------------------------------------------------
# UI — HEADER
# ------------------------------------------------------------------
st.title("💎 Diamond Price & Market Segment Predictor")
st.write(
    "Enter a diamond's attributes below, then use either button to predict "
    "its **price** or its **market segment (cluster)**."
)

# ------------------------------------------------------------------
# UI — SHARED INPUT FORM
# ------------------------------------------------------------------
st.subheader("Diamond Attributes")

col1, col2 = st.columns(2)

with col1:
    carat = st.number_input("Carat", min_value=0.01, max_value=10.0, value=1.00, step=0.01)
    x = st.number_input("x (length, mm)", min_value=0.1, max_value=15.0, value=6.40, step=0.01)
    y = st.number_input("y (width, mm)", min_value=0.1, max_value=15.0, value=6.30, step=0.01)
    z = st.number_input("z (depth, mm)", min_value=0.1, max_value=15.0, value=4.00, step=0.01)

with col2:
    cut = st.selectbox("Cut", CUT_OPTIONS, index=CUT_OPTIONS.index("Ideal") if "Ideal" in CUT_OPTIONS else 0)
    color = st.selectbox("Color", COLOR_OPTIONS)
    clarity = st.selectbox("Clarity", CLARITY_OPTIONS)

# depth (%) and table (%) are required by the trained models even though
# they weren't explicitly listed in the spec — without them the model
# cannot make a prediction, so they're included here as extra numeric inputs.
st.markdown("###### Additional measurements (required by the model)")
col3, col4 = st.columns(2)
with col3:
    depth = st.number_input("Depth (%)", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
with col4:
    table = st.number_input("Table (%)", min_value=40.0, max_value=95.0, value=57.0, step=0.1)

feature_row = build_feature_row(carat, cut, color, clarity, depth, table, x, y, z)

st.divider()

# ------------------------------------------------------------------
# 1) PRICE PREDICTION MODULE
# ------------------------------------------------------------------
st.subheader("🎯 Price Prediction")

if st.button("Predict Price", type="primary"):
    price_inr = predict_price_inr(feature_row)
    st.success(f"💰 Predicted Diamond Price: ₹{price_inr:,.2f}")

st.divider()

# ------------------------------------------------------------------
# 2) MARKET SEGMENT PREDICTION (CLUSTERING) MODULE
# ------------------------------------------------------------------
st.subheader("🎯 Market Segment Prediction")

if st.button("Predict Cluster"):
    cluster_id, cluster_name = predict_cluster(feature_row)
    st.success(f"🏷️ Cluster {cluster_id}: **{cluster_name}**")

st.divider()
st.caption(
    "Model: XGBoost regression pipeline for price · KMeans (k=3) for market segmentation. "
    "Prices are converted from USD to INR at a fixed rate of 1 USD = ₹83."
)
