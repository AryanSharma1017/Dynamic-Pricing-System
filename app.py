import os
import joblib
import gdown
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Dynamic Pricing System",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom Styling
# =========================
st.markdown("""
<style>
    .main {
        background-color: #0f1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero-card {
        background: linear-gradient(135deg, #1f2937, #111827);
        padding: 28px;
        border-radius: 22px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }

    .section-card {
        background: #161b22;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
        margin-bottom: 1rem;
    }

    .small-muted {
        color: #9ca3af;
        font-size: 0.95rem;
    }

    .price-box {
        background: linear-gradient(135deg, #065f46, #047857);
        color: white;
        padding: 24px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 24px rgba(4,120,87,0.25);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .price-label {
        font-size: 1rem;
        opacity: 0.9;
        margin-bottom: 8px;
    }

    .price-value {
        font-size: 2.2rem;
        font-weight: 700;
    }

    .feature-pill {
        display: inline-block;
        background: #1f2937;
        color: #e5e7eb;
        padding: 8px 12px;
        border-radius: 999px;
        margin: 4px 6px 4px 0;
        font-size: 0.9rem;
        border: 1px solid rgba(255,255,255,0.05);
    }

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        height: 3rem;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
    }

    .stButton > button:hover {
        filter: brightness(1.08);
    }

    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid rgba(255,255,255,0.06);
        padding: 14px;
        border-radius: 16px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Google Drive Model Setup
# =========================
MODEL_PATH = "Models/BestModel.pkl"
MODEL_FILE_ID = "1pX_MEzh3FOsy_QIz55_fnh3sx2jEFGvo"

# =========================
# Load Artifacts
# =========================
@st.cache_resource
def load_artifacts():
    os.makedirs("Models", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... this may take a minute ⏳"):
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

    model = joblib.load(MODEL_PATH)
    kmeans = joblib.load("Models/kmeans.pkl")
    feature_columns = joblib.load("Models/FeatureColumns.pkl")
    lat_long_map = joblib.load("Models/LatLongMap.pkl")
    neighbourhood_mean_map = joblib.load("Models/NeighbourhoodMeanMap.pkl")
    global_mean = joblib.load("Models/GlobalMean.pkl")
    default_lat = joblib.load("Models/DefaultLat.pkl")
    default_long = joblib.load("Models/DefaultLong.pkl")

    return model, kmeans, feature_columns, default_lat, default_long, lat_long_map, neighbourhood_mean_map, global_mean


model, kmeans, feature_columns, default_lat, default_long, lat_long_map, neighbourhood_mean_map, global_mean = load_artifacts()

# =========================
# Hero Section
# =========================
st.markdown("""
<div class="hero-card">
    <h1 style="margin-bottom: 0.4rem;">🏡 Dynamic Pricing System</h1>
    <p class="small-muted" style="margin-bottom: 1rem;">
        Estimate a recommended nightly price for short-term rental listings using a machine learning model
        trained on Airbnb-style listing data.
    </p>
    <div>
        <span class="feature-pill">ML-Powered Prediction</span>
        <span class="feature-pill">Neighbourhood Encoding</span>
        <span class="feature-pill">Location Clustering</span>
        <span class="feature-pill">Real-Time Input Pricing</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("⚙️ Listing Configuration")
st.sidebar.caption("Adjust the listing details to generate a price recommendation.")

neighbourhood = st.sidebar.selectbox(
    "Neighbourhood",
    sorted(neighbourhood_mean_map.keys())
)

city = st.sidebar.selectbox(
    "City",
    [
        'Asheville', 'Austin', 'Boston', 'Broward County', 'Cambridge',
        'Chicago', 'Clark County', 'Columbus', 'Denver', 'Hawaii',
        'Jersey City', 'Los Angeles', 'Nashville', 'New Orleans',
        'New York City', 'Oakland', 'Pacific Grove', 'Portland',
        'Rhode Island', 'Salem', 'San Clara Country', 'San Diego',
        'San Francisco', 'San Mateo County', 'Santa Cruz County',
        'Seattle', 'Twin Cities MSA', 'Washington D.C.'
    ]
)

room_type = st.sidebar.selectbox(
    "Room Type",
    ['Private room', 'Entire home/apt', 'Hotel room', 'Shared room']
)

minimum_nights = st.sidebar.number_input(
    "Minimum Nights",
    min_value=1,
    max_value=365,
    value=2
)

number_of_reviews = st.sidebar.number_input(
    "Number of Reviews",
    min_value=0,
    value=10
)

calculated_host_listings_count = st.sidebar.number_input(
    "Host Listings Count",
    min_value=1,
    value=1
)

availability_365 = st.sidebar.slider(
    "Availability (days/year)",
    min_value=0,
    max_value=365,
    value=180
)

latitude = lat_long_map["latitude"].get(neighbourhood, default_lat)
longitude = lat_long_map["longitude"].get(neighbourhood, default_long)

predict_clicked = st.sidebar.button("Predict Price")

# =========================
# Helper / Preprocessing
# =========================
def preprocess_input():
    df = pd.DataFrame([{
        "neighbourhood": neighbourhood,
        "city": city,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365,
        "latitude": latitude,
        "longitude": longitude
    }])

    df["minimum_nights"] = np.log1p(df["minimum_nights"])
    df["number_of_reviews"] = np.log1p(df["number_of_reviews"])
    df["calculated_host_listings_count"] = np.log1p(df["calculated_host_listings_count"])

    df["Encoded_Neighbourhood"] = df["neighbourhood"].map(neighbourhood_mean_map)
    df["Encoded_Neighbourhood"] = df["Encoded_Neighbourhood"].fillna(global_mean)

    df["location_cluster"] = kmeans.predict(df[["latitude", "longitude"]])

    df = df.drop(columns=["neighbourhood"])
    df = pd.get_dummies(df, columns=["city", "room_type", "location_cluster"], drop_first=True)
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df

# =========================
# Main Layout
# =========================
left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📋 Listing Overview")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Selected City", city)
    metric_col2.metric("Room Type", room_type)
    metric_col3.metric("Min Nights", minimum_nights)

    metric_col4, metric_col5, metric_col6 = st.columns(3)
    metric_col4.metric("Reviews", number_of_reviews)
    metric_col5.metric("Availability", f"{availability_365} days")
    metric_col6.metric("Host Listings", calculated_host_listings_count)

    st.markdown("#### Location Details")
    info1, info2, info3 = st.columns(3)
    info1.info(f"**Neighbourhood**\n\n{neighbourhood}")
    info2.info(f"**Latitude**\n\n{latitude:.4f}")
    info3.info(f"**Longitude**\n\n{longitude:.4f}")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧠 How the Prediction Works")
    st.write(
        """
        This system estimates a nightly rental price by combining:
        - neighbourhood-based encoded pricing signals
        - host and review activity
        - availability patterns
        - geographic clustering using latitude and longitude
        - room type and city-level categorical features
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("💵 Prediction Panel")

    if predict_clicked:
        try:
            final_input = preprocess_input()
            pred_log = model.predict(final_input)[0]
            pred_price = np.expm1(pred_log)

            st.markdown(f"""
            <div class="price-box">
                <div class="price-label">Recommended Nightly Price</div>
                <div class="price-value">${pred_price:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.success("Prediction generated successfully.")
            st.caption("This value is an ML-based estimate and should be used as a pricing guide.")

            with st.expander("View Processed Input Data"):
                st.dataframe(final_input, use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.markdown("""
        <div style="padding: 1rem 0 0.5rem 0;">
            <p class="small-muted">
                Configure your listing details in the sidebar and click <b>Predict Price</b> to generate a recommendation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.markdown("""
<hr style="margin-top: 2rem; margin-bottom: 1rem; border: 0.5px solid rgba(255,255,255,0.08);">
<p style="text-align:center; color:#9ca3af; font-size:0.9rem;">
Built with Streamlit • Machine Learning-Based Dynamic Pricing for Short-Term Rentals
</p>
""", unsafe_allow_html=True)