"""Streamlit dashboard for the AI House Price Prediction System.

Run with: streamlit run app/streamlit_app.py
"""
import os
import sys

import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import load_artifacts  # noqa: E402

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
    }
    .price-card h1 { font-size: 42px; margin: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏠 AI House Price Prediction System")
st.caption("California Housing Dataset · XGBoost · Trained on 20,640 real records")

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
    st.subheader("📊 Model Leaderboard")
    lb_path = os.path.join(REPORTS_DIR, "leaderboard.csv")
    if os.path.exists(lb_path):
        st.dataframe(pd.read_csv(lb_path), use_container_width=True)
    else:
        st.info("Run `python src/train.py` first to generate the leaderboard.")

with col2:
    st.subheader("🧠 Prediction")
    if predict_clicked:
        try:
            model, scaler, feature_columns = load_artifacts()
            record = {
                "longitude": longitude, "latitude": latitude,
                "housing_median_age": housing_median_age, "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms, "population": population,
                "households": households, "median_income": median_income,
                "ocean_proximity": ocean_proximity,
            }
            df = pd.DataFrame([record])
            df["rooms_per_household"] = df["total_rooms"] / df["households"].replace(0, 1)
            df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"].replace(0, 1)
            df["population_per_household"] = df["population"] / df["households"].replace(0, 1)
            df = pd.get_dummies(df, columns=["ocean_proximity"])
            df.columns = (
                df.columns.str.replace("<", "under_", regex=False)
                .str.replace(">", "over_", regex=False)
                .str.replace("[", "", regex=False)
                .str.replace("]", "", regex=False)
                .str.replace(" ", "_", regex=False)
            )
            df = df.reindex(columns=feature_columns, fill_value=0)
            scaled = scaler.transform(df)
            price = float(model.predict(scaled)[0])

            st.markdown(
                f"""<div class="price-card"><p>Estimated House Value</p>
                <h1>${price:,.0f}</h1></div>""",
                unsafe_allow_html=True,
            )
            st.session_state.history.insert(0, {**record, "predicted_price": round(price, 2)})
        except FileNotFoundError:
            st.error("No trained model found. Run `python src/train.py` first.")
    else:
        st.info("Set the house details in the sidebar and click **Predict Price**.")

st.divider()
c1, c2, c3 = st.columns(3)
for col, img, caption in [
    (c1, "correlation_heatmap.png", "Feature Correlation"),
    (c2, "feature_importance.png", "Feature Importance"),
    (c3, "actual_vs_predicted.png", "Actual vs Predicted"),
]:
    path = os.path.join(IMAGES_DIR, img)
    if os.path.exists(path):
        col.image(path, caption=caption, use_container_width=True)

if st.session_state.history:
    st.subheader("🕑 Recent Predictions")
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
