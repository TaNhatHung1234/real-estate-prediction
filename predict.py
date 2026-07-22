import pandas as pd
import joblib

model = joblib.load("house_model.pkl")

new_house = pd.DataFrame([{
    "Province": "Hồ Chí Minh",
    "District": "Gò Vấp",
    "Ward": "Phường 12",
    "Road": "Đường Quang Trung",
    "Area": 40,
    "Frontage": 5,
    "Access Road": 6,
    "House direction": "Đông - Nam",
    "Balcony direction": "Đông - Nam",
    "Floors": 4,
    "Bedrooms": 4,
    "Bathrooms": 5,
    "Legal status": "Have certificate",
    "Furniture state": "Full"
}])

price = model.predict(new_house)

print(f"Giá dự đoán: {price[0]:.2f} tỷ đồng")