import os
import pandas as pd


class RealEstateDSS:
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = os.path.join(
                os.path.dirname(__file__),
                "vietnam_housing_dataset.csv"
            )

        self.df = pd.read_csv(data_path)
        self.df = self.df.copy()

        if "District" in self.df.columns:
            self.df["District"] = self.df["District"].astype(str).str.strip()

        if "Area" in self.df.columns:
            self.df["Area"] = pd.to_numeric(self.df["Area"], errors="coerce")

        if "Price" in self.df.columns:
            self.df["Price"] = pd.to_numeric(self.df["Price"], errors="coerce")

        self.df = self.df.dropna(subset=["District", "Price"]).reset_index(drop=True)

        for column in ["Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"]:
            if column in self.df.columns:
                self.df[column] = pd.to_numeric(self.df[column], errors="coerce")
                self.df[column] = self.df[column].fillna(self.df[column].median())

        for column in [
            "Province",
            "Ward",
            "Road",
            "House direction",
            "Balcony direction",
            "Legal status",
            "Furniture state"
        ]:
            if column in self.df.columns:
                self.df[column] = self.df[column].fillna("Unknown")

    # =====================
    # 1. Đề xuất giá bán
    # =====================
    def recommend_price(self, predicted_price):
        return {
            "sell_fast": round(predicted_price * 0.95, 2),
            "normal": round(predicted_price, 2),
            "maximize_profit": round(predicted_price * 1.05, 2)
        }

    # =====================
    # 2. Giá trung bình khu vực
    # =====================
    def get_market_average(self, district):
        district_key = str(district).strip().lower()

        area_data = self.df[
            self.df["District"].astype(str).str.strip().str.lower() == district_key
        ]

        if len(area_data) == 0:
            return None

        return round(area_data["Price"].mean(), 2)

    # =====================
    # 3. So sánh thị trường
    # =====================
    def compare_with_market(self, predicted_price, district):

        market_avg = self.get_market_average(district)

        if market_avg is None:
            return "Không có dữ liệu thị trường."

        diff = (
            (predicted_price - market_avg)
            / market_avg
        ) * 100

        if diff > 10:
            return (
                f"Giá cao hơn thị trường "
                f"{diff:.1f}%."
            )

        elif diff < -10:
            return (
                f"Giá thấp hơn thị trường "
                f"{abs(diff):.1f}%."
            )

        else:
            return (
                "Giá phù hợp với mặt bằng "
                "thị trường hiện tại."
            )

    # =====================
    # 4. Tìm BĐS tương tự
    # =====================
    def find_similar_properties(
            self,
            district,
            area,
            tolerance=20):

        district_key = str(district).strip().lower()

        similar = self.df[
            (self.df["District"].astype(str).str.strip().str.lower() == district_key)
            &
            (self.df["Area"].between(
                area - tolerance,
                area + tolerance
            ))
        ]

        if len(similar) == 0:
            return pd.DataFrame()

        columns = [
            "District",
            "Area",
            "Bedrooms",
            "Bathrooms",
            "Price"
        ]

        available_columns = [
            col for col in columns
            if col in similar.columns
        ]

        return similar[
            available_columns
        ].head(5)

    # =====================
    # 5. Khuyến nghị DSS
    # =====================
    def generate_advice(
            self,
            predicted_price,
            district):

        market_avg = self.get_market_average(district)

        if market_avg is None:
            return (
                "Không đủ dữ liệu "
                "để đưa ra khuyến nghị."
            )

        if predicted_price > market_avg * 1.1:
            return (
                "Bất động sản có giá trị cao "
                "hơn mặt bằng thị trường. "
                "Có thể đăng bán ở mức cao."
            )

        elif predicted_price < market_avg * 0.9:
            return (
                "Giá dự đoán thấp hơn "
                "so với thị trường. "
                "Nên xem xét lại thông tin "
                "hoặc cân nhắc điều chỉnh giá."
            )

        else:
            return (
                "Giá dự đoán phù hợp "
                "với mặt bằng thị trường. "
                "Có thể đăng bán ở mức đề xuất."
            )