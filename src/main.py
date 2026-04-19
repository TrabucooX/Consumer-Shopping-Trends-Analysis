import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def evaluate_model():
    df = pd.read_csv("../data/Consumer_Shopping_Trends_2026.csv")

    model = joblib.load("../data/final_xgb_model.pkl")
    encoder = joblib.load("../data/label_encoder.pkl")

    # We prepare data to evaluate
    df_encoded = encoder.transform(df)
    X = df_encoded.drop("shopping_preference", axis=1)
    y = df_encoded["shopping_preference"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    predictions = model.predict(X_test)

    # Classification report
    print("Global Performance")
    print(classification_report(y_test, predictions, target_names=["Hybrid", "Online", "Store"]))

    # Confusion matrix
    print("Confusion matrix across the three labels")
    print(confusion_matrix(y_test, predictions, labels=["Hybrid", "Online", "Store"]))

    # Displaying confusion matrix for better visualization
    display_confusion_matrix = ConfusionMatrixDisplay.from_predictions(y_test, predictions, 
                                        display_labels=["Hybrid", "Online", "Store"])
    plt.savefig("../visuals/confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    evaluate_model()