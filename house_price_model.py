import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================
# Đọc dữ liệu
# =====================
df = pd.read_csv("vietnam_housing_dataset.csv")

# =====================
# Xóa dòng thiếu giá
# =====================
df = df.dropna(subset=["Price"])

# =====================
# Target
# =====================
y = df["Price"]

# =====================
# Features
# =====================
X = df.drop("Price", axis=1)

# =====================
# Cột số
# =====================
numeric_features = [
    "Area",
    "Frontage",
    "Access Road",
    "Floors",
    "Bedrooms",
    "Bathrooms"
]

# =====================
# Cột text
# =====================
categorical_features = [
    "Province",
    "District",
    "Ward",
    "Road",
    "House direction",
    "Balcony direction",
    "Legal status",
    "Furniture state"
]

# =====================
# Pipeline số
# =====================
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# =====================
# Pipeline text
# =====================
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# =====================
# Preprocessor
# =====================
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# =====================
# XGBoost
# =====================
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# =====================
# Pipeline hoàn chỉnh
# =====================
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# =====================
# Train/Test
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================
# Train
# =====================
pipeline.fit(X_train, y_train)

# =====================
# Predict
# =====================
y_pred = pipeline.predict(X_test)

# =====================
# Metrics
# =====================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("="*40)
print("KẾT QUẢ MÔ HÌNH")
print("="*40)
print(f"MAE  : {mae:.3f} tỷ")
print(f"RMSE : {rmse:.3f} tỷ")
print(f"R²   : {r2:.3f}")

joblib.dump(pipeline, "house_model.pkl")
print("Đã lưu model thành house_model.pkl")