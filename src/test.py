import pandas as pd
import joblib
from pathlib import Path

def run():
    root_path = Path(__file__).resolve().parents[1]
    processed_path = root_path / "data" / "processed"
    models_path = root_path / "models"
    outputs_path = root_path / "data" / "outputs"

    # **🔹 Load the trained Random Forest model**
    model = joblib.load(models_path / "random_forest_tuned.pkl")

    # **🔹 Load test data**
    test_data = pd.read_csv(processed_path / "processed_test.csv")

    # **🔹 Remove target column (since we want to predict churn)**
    X_test = test_data.drop(columns=["Churn"], errors="ignore")  # Remove "Churn" if it exists

    # **🔹 Make predictions**
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # Churn probability

    # **🔹 Save predictions**
    X_test["Predicted_Churn"] = predictions
    X_test["Churn_Probability"] = probabilities
    X_test.to_csv(outputs_path / "churn_predictions.csv", index=False)

    print("✅ Predictions saved in 'churn_predictions.csv'.")