# Real Estate Price Prediction

Hệ thống dự đoán giá bất động sản sử dụng Machine Learning (XGBoost).

## Cấu trúc thư mục

real-estate-prediction/
│
├── vietnam_housing_dataset.csv
├── house_price_model.py
├── predict.py
├── house_model.pkl
├── requirements.txt
└── README.md


---

## Cài đặt

Tạo môi trường ảo:
python -m venv venv


Kích hoạt:

Windows:
venv\Scripts\activate

Cài thư viện:

pip install -r requirements.txt

---

## Huấn luyện mô hình

python house_price_model.py


Sau khi train thành công sẽ sinh ra:


house_model.pkl

---

## Dự đoán

Chỉnh dữ liệu trong file:

```python
house = pd.DataFrame([{
    "Province": "Hồ Chí Minh",
    "District": "Gò Vấp",
    "Ward": "Phường 12",
    "Road": "Đường Quang Trung",
    "Area": 70,
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
```

Chạy:
python predict.py


Ví dụ kết quả:

Giá dự đoán: 6.71 tỷ đồng

---

## Công nghệ sử dụng

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Joblib

---

## Tác giả

Hưng Tạ

Đồ án: Decision Support System (DSS) for Real Estate Price Prediction.