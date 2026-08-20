"""Streamlit dashboard for the AI House Price Prediction System.

Run with: streamlit run app/streamlit_app.py
"""
import os
import sys

import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import predict_price  # noqa: E402
from utils import load_object  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

st.set_page_config(page_title="AI House Price Predictor", page_icon="🏠", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .price-card {
        background: linear-gradient(135deg, #1f6feb 0%, #0e4c92 100%);
        padding: 28px; border-radius: 16px; text-align: center; color: white;
        margin-bottom: 20px;
    }
    .price-card h1 { font-size: 42px; margin: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

best_model_name = "Best ML Model"
best_name_path = os.path.join(MODELS_DIR, "best_model_name.pkl")
if os.path.exists(best_name_path):
    best_model_name = load_object(best_name_path)

st.title("🏠 AI House Price Prediction System")
st.caption(f"California Housing Dataset · Optimized Model: **{best_model_name}** · 20,640 Censused Records")

with st.sidebar:
    st.header("🔧 House Details")
    median_income = st.slider("Median Income (in $10,000s)", 0.5, 15.0, 5.0, 0.1)
    housing_median_age = st.slider("House Age (years)", 1, 52, 20)
    total_rooms = st.slider("Total Rooms (block)", 2, 15000, 2500)
    total_bedrooms = st.slider("Total Bedrooms (block)", 1, 3000, 500)
    population = st.slider("Population (block)", 3, 20000, 1400)
    households = st.slider("Households (block)", 1, 5000, 500)
    latitude = st.slider("Latitude", 32.5, 42.0, 34.05, 0.01)
    longitude = st.slider("Longitude", -124.5, -114.0, -118.25, 0.01)
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"],
    )
    predict_clicked = st.button("💰 Predict Price", use_container_width=True)

if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Model Comparison Leaderboard")
    lb_path = os.path.join(REPORTS_DIR, "leaderboard.csv")
    if os.path.exists(lb_path):
        lb_df = pd.read_csv(lb_path)
        st.dataframe(lb_df, use_container_width=True)
    else:
        st.info("Run `python src/train.py` first to generate the leaderboard.")

with col2:
    st.subheader("🧠 Price Prediction")
    if predict_clicked:
        try:
            record = {
                "longitude": longitude, "latitude": latitude,
                "housing_median_age": housing_median_age, "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms, "population": population,
                "households": households, "median_income": median_income,
                "ocean_proximity": ocean_proximity,
            }
            price = predict_price(record)

            st.markdown(
                f"""<div class="price-card"><p>Estimated House Value</p>
                <h1>${price:,.2f}</h1></div>""",
                unsafe_allow_html=True,
            )
            st.session_state.history.insert(0, {**record, "predicted_price": round(price, 2)})
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Configure house features in the sidebar and click **Predict Price**.")

st.divider()
st.subheader("📈 Model Visualizations & EDA Insights")

c1, c2, c3 = st.columns(3)
with c1:
    path = os.path.join(IMAGES_DIR, "geographical_distribution.png")
    if os.path.exists(path):
        st.image(path, caption="Geographical Price Distribution", use_container_width=True)
with c2:
    path = os.path.join(IMAGES_DIR, "correlation_heatmap.png")
    if os.path.exists(path):
        st.image(path, caption="Feature Correlation Heatmap", use_container_width=True)
with c3:
    path = os.path.join(IMAGES_DIR, "feature_importance.png")
    if os.path.exists(path):
        st.image(path, caption="Feature Importance", use_container_width=True)

c4, c5, c6 = st.columns(3)
with c4:
    path = os.path.join(IMAGES_DIR, "actual_vs_predicted.png")
    if os.path.exists(path):
        st.image(path, caption="Actual vs Predicted Prices", use_container_width=True)
with c5:
    path = os.path.join(IMAGES_DIR, "residual_distribution.png")
    if os.path.exists(path):
        st.image(path, caption="Residual Distribution", use_container_width=True)
with c6:
    path = os.path.join(IMAGES_DIR, "feature_distributions.png")
    if os.path.exists(path):
        st.image(path, caption="Feature Distributions", use_container_width=True)

if st.session_state.history:
    st.divider()
    st.subheader("🕑 Recent Prediction History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)
    fig = px.line(hist_df.iloc[::-1], y="predicted_price", markers=True,
                  title="Prediction History Trend")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "⬇️ Download Predictions as CSV",
        hist_df.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )
