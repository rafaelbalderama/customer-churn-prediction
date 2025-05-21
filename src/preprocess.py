import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def run():
    root_path = Path(__file__).resolve().parents[1]
    raw_path = root_path / "data" / "raw"
    processed_path = root_path / "data" / "processed"

    # File paths
    train_path = raw_path / "train.csv"
    test_path = raw_path / "test.csv"
    val_path = raw_path / "validation.csv"

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)

    ### Step 1: Drop Unnecessary Columns ###
    drop_cols = ["Customer ID", "City", "Zip Code", "Lat Long", "State", "Country", "Churn Category", "Churn Reason", "Customer Status", "Churn Score"]
    train_df.drop(columns=drop_cols, inplace=True)
    test_df.drop(columns=drop_cols, inplace=True)
    val_df.drop(columns=drop_cols, inplace=True)

    ### Step 2: Split Features and Target Before Preprocessing ###
    X_train, y_train = train_df.drop("Churn", axis=1), train_df["Churn"]
    X_test, y_test = test_df.drop("Churn", axis=1), test_df["Churn"]
    X_val, y_val = val_df.drop("Churn", axis=1), val_df["Churn"]

    ### Step 3: Handle Missing Values ###
    # Fill numerical columns with median
    # Only compute the median from the train ds to avoid data leakage
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    num_medians = X_train[num_cols].median()
    X_train[num_cols] = X_train[num_cols].fillna(num_medians)
    X_test[num_cols] = X_test[num_cols].fillna(num_medians)
    X_val[num_cols] = X_val[num_cols].fillna(num_medians)


    # Fill categorical columns with mode
    # Only compute the mode from the train ds to avoid data leakage
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    cat_mode = X_train[cat_cols].mode().iloc[0]
    X_train[cat_cols] = X_train[cat_cols].fillna(cat_mode)
    X_test[cat_cols] = X_test[cat_cols].fillna(cat_mode)
    X_val[cat_cols] = X_val[cat_cols].fillna(cat_mode)


    ### Step 4: Encode Categorical Features ###
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        X_val[col] = le.transform(X_val[col])

    ### Step 5: Remove Outliers Using IQR ###
    def remove_outliers(df, cols):
        Q1 = df[cols].quantile(0.25)
        Q3 = df[cols].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((df[cols] < (Q1 - 1.5 * IQR)) | (df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        return df[mask]

    # Combine back y for outlier filtering
    train_combined = X_train.copy()
    train_combined["Churn"] = y_train
    train_combined = remove_outliers(train_combined, num_cols)
    X_train = train_combined.drop("Churn", axis=1)
    y_train = train_combined["Churn"]

    ### Step 6: Save Processed Data ###
    train_processed = pd.concat([X_train, y_train], axis=1)
    test_processed = pd.concat([X_test, y_test], axis=1)
    val_processed = pd.concat([X_val, y_val], axis=1)

    train_processed.to_csv(processed_path / "processed_train.csv", index=False)
    test_processed.to_csv(processed_path / "processed_test.csv", index=False)
    val_processed.to_csv(processed_path / "processed_val.csv", index=False)

    print("Data preprocessing complete! Cleaned files saved.")