import pandas as pd
import os
import joblib
from pathlib import Path

def run():
    root_path = Path(__file__).resolve().parents[1]
    models_path = root_path / "models"
    processed_path = root_path / "data" / "processed"
    outputs_path = root_path / "data" / "outputs"

    # Check if model and preprocessed data exist
    if not (models_path / "random_forest_model.pkl").exists() or not (processed_path / "processed_train.csv"):
        print("Model or preprocessed data not found! Train the model first.\n")
        exit()

    print("Model and data found. Proceeding to export feature importance...\n")

    # Load trained model and data
    rf_model = joblib.load(models_path / "random_forest_model.pkl")
    train_df = pd.read_csv(processed_path / "processed_train.csv")

    # Extract feature names & importance
    feature_names = train_df.drop("Churn", axis=1).columns
    feature_importances = rf_model.feature_importances_

    # Create DataFrame for Export
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Save to CSV for Tableau or Excel
    importance_df.to_csv(outputs_path / "feature_importance.csv", index=False)
    print("Feature importance saved as 'feature_importance.csv'.\n")