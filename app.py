import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Dynamic Pricing System", page_icon="📊", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("Models/BestModel.pkl")
    kmeans = joblib.load("Models/kmeans.pkl")
    feature_columns = joblib.load("Models/FeatureColumns.pkl")
    lat_long_map = joblib.load("Models/LatLongMap.pkl")
    neighbourhood_mean_map = joblib.load("Models/NeighbourhoodMeanMap.pkl")
    global_mean = joblib.load("Models/GlobalMean.pkl")
    default_lat = joblib.load("Models/DefaultLat.pkl")
    default_long = joblib.load("Models/DefaultLong.pkl")

    return model, kmeans, feature_columns, default_lat, default_long, lat_long_map, neighbourhood_mean_map, global_mean


model, kmeans, feature_columns, default_lat, default_long, lat_long_map, neighbourhood_mean_map, global_mean = load_artifacts()

st.title("📊 Dynamic Pricing System for Short-Term Rentals")
st.write(
    "Predict a recommended nightly price for a short-term rental listing "
    "using a machine learning model trained on Airbnb-style listing data."
)

st.subheader("Enter Listing Details")

col1, col2 = st.columns(2)

with col1:
    neighbourhood = st.selectbox("Neighbourhood", sorted(neighbourhood_mean_map.keys()))
    city = st.selectbox(
        "City",
        ['Asheville', 'Austin', 'Boston', 'Broward County', 'Cambridge',
         'Chicago', 'Clark County', 'Columbus', 'Denver', 'Hawaii',
         'Jersey City', 'Los Angeles', 'Nashville', 'New Orleans',
         'New York City', 'Oakland', 'Pacific Grove', 'Portland',
         'Rhode Island', 'Salem', 'San Clara Country', 'San Diego',
         'San Francisco', 'San Mateo County', 'Santa Cruz County',
         'Seattle', 'Twin Cities MSA', 'Washington D.C.']
    )
    room_type = st.selectbox(
        "Room Type",
        ['Private room', 'Entire home/apt', 'Hotel room', 'Shared room']
    )
    minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=2)
    number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=10)

with col2:
    calculated_host_listings_count = st.number_input(
        "Calculated Host Listings Count", min_value=1, value=1
    )
    availability_365 = st.slider("Availability (days/year)", min_value=0, max_value=365, value=180)
    latitude = lat_long_map["latitude"].get(neighbourhood, default_lat)
    longitude = lat_long_map["longitude"].get(neighbourhood, default_long)


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


if st.button("Predict Price"):
    try:
        final_input = preprocess_input()
        pred_log = model.predict(final_input)[0]
        pred_price = np.expm1(pred_log)

        st.success(f"Recommended Nightly Price: ${pred_price:,.2f}")
        st.subheader("Input Summary")
        st.dataframe(final_input)

    except Exception as e:
        st.error(f"Error during prediction: {e}")