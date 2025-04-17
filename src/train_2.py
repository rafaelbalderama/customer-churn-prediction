import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from pathlib import Path

def run():
    root_path = Path(__file__).resolve().parents[1]
    processed_path = root_path / "data" / "processed"
    models_path = root_path / "models"

    # **🔹 Check if preprocessed data exists**
    if not all((processed_path / f).exists() for f in ["processed_train.csv", "processed_test.csv", "processed_val.csv"]):
        print("❌ Preprocessed data not found! Please run the preprocessing script first.")
        exit()

    print("✅ Preprocessed data found. Proceeding with model training...")

    # **🔹 Load preprocessed data**
    train_df = pd.read_csv(processed_path / "processed_train.csv")
    test_df = pd.read_csv(processed_path / "processed_test.csv")
    val_df = pd.read_csv(processed_path / "processed_val.csv")  # Include validation data

    # Split into features (X) and target (y)
    X_train, y_train = train_df.drop("Churn", axis=1), train_df["Churn"]
    X_test, y_test = test_df.drop("Churn", axis=1), test_df["Churn"]
    X_val, y_val = val_df.drop("Churn", axis=1), val_df["Churn"]  # Validation set

    # **🔹 Manually Adjusted Hyperparameters**
    best_rf_model = RandomForestClassifier(
        n_estimators=500,      # Increased from 300 ➝ Better learning
        max_depth=25,          # Adjusted for complexity control
        min_samples_split=5,   # Balancing bias-variance
        min_samples_leaf=2,    # Avoid too small leaf nodes
        random_state=42,
        n_jobs=-1
    )

    # Train the model
    best_rf_model.fit(X_train, y_train)

    # **🔹 Make Predictions on Validation and Test Data**
    y_pred_val = best_rf_model.predict(X_val)
    y_pred_test = best_rf_model.predict(X_test)

    # **🔹 Evaluate Performance on Validation Set**
    val_accuracy = accuracy_score(y_val, y_pred_val)
    val_precision = precision_score(y_val, y_pred_val)
    val_recall = recall_score(y_val, y_pred_val)
    val_f1 = f1_score(y_val, y_pred_val)

    print("\n📊 **Validation Set Performance:**")
    print(f"✅ Accuracy: {val_accuracy:.4f}")
    print(f"✅ Precision: {val_precision:.4f}")
    print(f"✅ Recall: {val_recall:.4f}")
    print(f"✅ F1 Score: {val_f1:.4f}")

    # **🔹 Evaluate Performance on Test Set**
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)

    print("\n📊 **Test Set Performance:**")
    print(f"✅ Accuracy: {test_accuracy:.4f}")
    print(f"✅ Precision: {test_precision:.4f}")
    print(f"✅ Recall: {test_recall:.4f}")
    print(f"✅ F1 Score: {test_f1:.4f}")

    # Save the improved model
    joblib.dump(best_rf_model, models_path / "random_forest_tuned.pkl")
    print("✅ Improved model saved as 'random_forest_tuned.pkl'")