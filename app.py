import streamlit as st
import pandas as pd
import joblib

from dss import RealEstateDSS

# Load model
model = joblib.load("house_model.pkl")

# Load DSS
dss = RealEstateDSS("vietnam_housing_dataset.csv")

st.title("🏠 Real Estate DSS")
st.subheader("Decision Support System for Real Estate Sellers")

# =====================
# INPUT
# =====================

province = st.text_input("Province", "Hồ Chí Minh")
district = st.text_input("District", "Gò Vấp")
ward = st.text_input("Ward", "Phường 12")
road = st.text_input("Road", "Đường Quang Trung")

area = st.number_input("Area (m²)", value=40)
frontage = st.number_input("Frontage (m)", value=5)
access_road = st.number_input("Access Road (m)", value=6)

house_direction = st.selectbox(
    "House direction",
    ["Đông", "Tây", "Nam", "Bắc", "Đông - Nam"]
)

balcony_direction = st.selectbox(
    "Balcony direction",
    ["Đông", "Tây", "Nam", "Bắc", "Đông - Nam"]
)

floors = st.number_input("Floors", value=4)
bedrooms = st.number_input("Bedrooms", value=4)
bathrooms = st.number_input("Bathrooms", value=5)

legal_status = st.selectbox(
    "Legal status",
    ["Have certificate"]
)

furniture_state = st.selectbox(
    "Furniture state",
    ["Full", "Basic", "Empty"]
)

# =====================
# PREDICT BUTTON
# =====================

if st.button("Predict Price"):

    new_house = pd.DataFrame([{
        "Province": province,
        "District": district,
        "Ward": ward,
        "Road": road,
        "Area": area,
        "Frontage": frontage,
        "Access Road": access_road,
        "House direction": house_direction,
        "Balcony direction": balcony_direction,
        "Floors": floors,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Legal status": legal_status,
        "Furniture state": furniture_state
    }])

    # Predict
    predicted_price = model.predict(new_house)[0]

    # DSS
    recommend = dss.recommend_price(predicted_price)

    market_avg = dss.get_market_average(district)

    market_comment = dss.compare_with_market(
        predicted_price,
        district
    )

    advice = dss.generate_advice(
        predicted_price,
        district
    )

    # =====================
    # OUTPUT
    # =====================

    st.success(
        f"Predicted Price: {predicted_price:.2f} billion VND"
    )

    st.subheader("📊 Price Recommendation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Sell Fast",
            f"{recommend['sell_fast']:.2f} B"
        )

    with col2:
        st.metric(
            "Normal Price",
            f"{recommend['normal']:.2f} B"
        )

    with col3:
        st.metric(
            "Max Profit",
            f"{recommend['maximize_profit']:.2f} B"
        )

    # Market Average

    if market_avg is not None:
        st.subheader("🏘 Market Analysis")

        st.write(
            f"Average price in {district}: "
            f"{market_avg:.2f} billion VND"
        )

        st.info(market_comment)

    # DSS Advice

    st.subheader("💡 DSS Recommendation")
    st.write(advice)

    # Similar Properties

    st.subheader("🏠 Similar Properties")

    similar = dss.find_similar_properties(
        district=district,
        area=area
    )

    if len(similar) > 0:
        st.dataframe(similar)
    else:
        st.warning("No similar properties found.")