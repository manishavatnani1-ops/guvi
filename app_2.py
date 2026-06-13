
import streamlit as st
import pandas as pd
import joblib

# LOAD PIPELINE
pipeline = joblib.load("pipeline.pkl")

# PAGE CONFIG
st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎"
)

# TITLE
st.title("💎 Diamond Price Prediction App")

st.write(
    "Enter diamond details below to predict the estimated price."
)

# SIDEBAR
st.sidebar.header("Diamond Features")

# INPUTS
carat = st.sidebar.number_input(
    "Carat",
    min_value=0.0,
    step=0.1
)

cut = st.sidebar.selectbox(
    "Cut",
    ["Fair", "Good", "Very Good", "Premium", "Ideal"]
)

color = st.sidebar.selectbox(
    "Color",
    ["D", "E", "F", "G", "H", "I", "J"]
)

clarity = st.sidebar.selectbox(
    "Clarity",
    ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
)

depth = st.sidebar.number_input(
    "Depth",
    min_value=0.0
)

table = st.sidebar.number_input(
    "Table",
    min_value=0.0
)

x = st.sidebar.number_input(
    "Length (x)",
    min_value=0.0
)

y = st.sidebar.number_input(
    "Width (y)",
    min_value=0.0
)

z = st.sidebar.number_input(
    "Depth (z)",
    min_value=0.0
)

# ENCODING MAPPINGS
cut_mapping = {
    "Fair": 0,
    "Good": 1,
    "Very Good": 2,
    "Premium": 3,
    "Ideal": 4
}

color_mapping = {
    "D": 0,
    "E": 1,
    "F": 2,
    "G": 3,
    "H": 4,
    "I": 5,
    "J": 6
}

clarity_mapping = {
    "I1": 0,
    "SI2": 1,
    "SI1": 2,
    "VS2": 3,
    "VS1": 4,
    "VVS2": 5,
    "VVS1": 6,
    "IF": 7
}

# CONVERT TO NUMERIC
cut = cut_mapping[cut]
color = color_mapping[color]
clarity = clarity_mapping[clarity]

# CREATE DATAFRAME
input_df = pd.DataFrame({
    'carat': [carat],
    'cut': [cut],
    'color': [color],
    'clarity': [clarity],
    'depth': [depth],
    'table': [table],
    'x': [x],
    'y': [y],
    'z': [z]
})

# PREDICTION
if st.button("Predict Price"):

    prediction = pipeline.predict(input_df)

    st.success(
    f"Estimated Diamond Price: ₹{prediction[0] * 83.5:,.2f}"

    )

    st.balloons()


