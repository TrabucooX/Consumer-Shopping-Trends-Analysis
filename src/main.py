import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

FEATURES = ["age", 
            "monthly_income",
            "daily_internet_hours",
            "smartphone_usage_years",
            "social_media_hours",
            "online_payment_trust_score",
            "tech_savvy_score",
            "monthly_online_orders",
            "monthly_store_visits",
            "avg_online_spend",
            "avg_store_spend",
            "discount_sensitivity",
            "return_frequency",
            "avg_delivery_days",
            "delivery_fee_sensitivity",
            "free_return_importance",
            "product_availability_online",
            "impulse_buying_score",
            "need_touch_feel_score",
            "brand_loyalty_score",
            "environmental_awareness",
            "time_pressure_level",
            "gender",
            "city_tier"]

def evaluate_model():
    df = pd.read_csv("data/Consumer_Shopping_Trends_2026.csv")

    model = joblib.load("data/final_xgb_model.pkl")
    encoder = joblib.load("data/label_encoder.pkl")

    # We prepare data to evaluate
    df_encoded = pd.get_dummies(df, columns=["city_tier", "gender"],
                                  dtype="int32")
    X = df_encoded.drop("shopping_preference", axis=1)
    y = encoder.transform(df_encoded["shopping_preference"])

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    predictions = model.predict(X_test)

    # Classification report
    print("Global Performance")
    print(classification_report(y_test, predictions, target_names=["Hybrid", "Online", "Store"]))

    # Confusion matrix
    print("Confusion matrix across the three labels")
    print(confusion_matrix(y_test, predictions))

    # Displaying confusion matrix for better visualization
    display_confusion_matrix = ConfusionMatrixDisplay.from_predictions(y_test, predictions, 
                                        display_labels=["Hybrid", "Online", "Store"])
    plt.savefig("visuals/confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    evaluate_model()