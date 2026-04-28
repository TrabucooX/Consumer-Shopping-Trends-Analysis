from fastapi import FastAPI
import joblib
from pydantic import BaseModel,Field
import pandas as pd

app = FastAPI(title="Customer Shopping Trends Prediction API")

def load_model_encoder():
    global model, encoder, model_columns
    model = joblib.load("data/final_xgb_model.pkl")
    encoder = joblib.load("data/label_encoder.pkl")
    model_columns = joblib.load("data/model_columns.pkl")
    print("Model and encoder loaded successfully.")

load_model_encoder()


class CustomerData(BaseModel):
    age: int = Field(..., example=50)
    monthly_income:int = Field(..., example=54000)
    daily_internet_hours: float = Field(..., example=5.5)
    smartphone_usage_years: int = Field(..., example=4)
    social_media_hours: float = Field(..., example=2.3)
    online_payment_trust_score: int = Field(..., example=8, ge=1, le=10) # Assuming 1-10 scale
    tech_savvy_score: int = Field(..., example=7)
    monthly_online_orders: int = Field(..., example=12)
    monthly_store_visits: int = Field(..., example=3)
    avg_online_spend: int = Field(..., example=150)
    avg_store_spend: int = Field(..., example=80)
    discount_sensitivity: int = Field(..., example=5)
    return_frequency: int = Field(..., example=2)
    avg_delivery_days: int = Field(..., example=3)
    delivery_fee_sensitivity: int = Field(..., example=9)
    free_return_importance: int = Field(..., example=10)
    product_availability_online: int = Field(..., example=8)
    impulse_buying_score: int = Field(..., example=6)
    need_touch_feel_score: int = Field(..., example=4)
    brand_loyalty_score: int = Field(..., example=7)
    environmental_awareness: int = Field(..., example=8)
    time_pressure_level: int = Field(..., example=5)

    gender: str = Field(..., example="Female")
    city_tier: str = Field(..., example="Tier 1")

    class Config:
        # This helps FastAPI documentation show a nice example
        schema_extra = {
            "example": {
                "age": 47,
                "monthly_income":67899,
                "daily_internet_hours": 4.5,
                "smartphone_usage_years": 5,
                "social_media_hours": 3.0,
                "online_payment_trust_score": 7,
                "tech_savvy_score": 8,
                "monthly_online_orders": 10,
                "monthly_store_visits": 2,
                "avg_online_spend": 200,
                "avg_store_spend": 50,
                "discount_sensitivity": 6,
                "return_frequency": 1,
                "avg_delivery_days": 2,
                "delivery_fee_sensitivity": 8,
                "free_return_importance": 9,
                "product_availability_online": 7,
                "impulse_buying_score": 4,
                "need_touch_feel_score": 3,
                "brand_loyalty_score": 8,
                "environmental_awareness": 9,
                "time_pressure_level": 4,
                "gender": "Non-binary",
                "city_tier": "Tier 2"
            }
        }

@app.get("/")
def home():
    return {"message":"Welcome to the Customer Shopping Trends Prediction API"}

@app.post("/predict")
def predict(data: CustomerData):
    input_df = pd.DataFrame([data.model_dump()])

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    predictions = model.predict(input_df)
    string_prediction = encoder.inverse_transform(predictions[0].ravel())
    return{"predictions": string_prediction[0]}
