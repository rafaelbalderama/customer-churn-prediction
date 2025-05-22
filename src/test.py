import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run():
    root_path = Path(__file__).resolve().parents[1]
    processed_path = root_path / "data" / "processed"
    models_path = root_path / "models"
    outputs_path = root_path / "data" / "outputs"

    # Load the trained Random Forest model
    model = joblib.load(models_path / "random_forest_model.pkl")

    # Load test data
    test_data = pd.read_csv(processed_path / "processed_test.csv")

    # Save target column temporarily and (since we want to predict churn)
    actual_churn = test_data["Churn"].copy()
    X_test = test_data.drop(columns=["Churn"], errors="ignore")  # Remove "Churn" if it exists

    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # Churn probability

    # Evaluate the model
    print("Test Set Evaluation:")
    print(f"Accuracy:  {accuracy_score(actual_churn, predictions):.4f}")
    print(f"Precision: {precision_score(actual_churn, predictions):.4f}")
    print(f"Recall:    {recall_score(actual_churn, predictions):.4f}")
    print(f"F1 Score:  {f1_score(actual_churn, predictions):.4f}\n")
    print("Confusion Matrix:")
    print(confusion_matrix(actual_churn, predictions), "\n")

    # For plotting
    accuracy = accuracy_score(actual_churn, predictions)
    precision = precision_score(actual_churn, predictions)
    recall = recall_score(actual_churn, predictions)
    f1 = f1_score(actual_churn, predictions)

    # Metrics dictionary
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    # Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis")
    plt.ylim(0, 1)
    plt.title("Model Evaluation Metrics")
    plt.ylabel("Score")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Insert back actual churn and save predictions
    X_test["Churn"] = actual_churn
    X_test["Predicted_Churn"] = predictions
    X_test["Churn_Probability"] = probabilities
    X_test.sort_values(by="Churn_Probability", ascending=False, inplace=True)
    X_test.to_csv(outputs_path / "churn_predictions.csv", index=False)

    print("Predictions saved in 'churn_predictions.csv'.")