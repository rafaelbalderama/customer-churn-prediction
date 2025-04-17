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
    drop_cols = ["Customer ID", "City", "Zip Code", "Lat Long", "State", "Country", "Churn Category", "Churn Reason"]
    train_df.drop(columns=drop_cols, inplace=True)
    test_df.drop(columns=drop_cols, inplace=True)
    val_df.drop(columns=drop_cols, inplace=True)

    ### Step 2: Handle Missing Values ###
    # Fill numerical columns with median
    num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns
    train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].median())
    test_df[num_cols] = test_df[num_cols].fillna(test_df[num_cols].median())
    val_df[num_cols] = val_df[num_cols].fillna(val_df[num_cols].median())

    # Fill categorical columns with mode
    cat_cols = train_df.select_dtypes(include=["object"]).columns
    train_df[cat_cols] = train_df[cat_cols].fillna(train_df[cat_cols].mode().iloc[0])
    test_df[cat_cols] = test_df[cat_cols].fillna(test_df[cat_cols].mode().iloc[0])
    val_df[cat_cols] = val_df[cat_cols].fillna(val_df[cat_cols].mode().iloc[0])

    ### Step 3: Encode Categorical Features ###
    label_encoder = LabelEncoder()
    for col in cat_cols:
        train_df[col] = label_encoder.fit_transform(train_df[col])
        test_df[col] = label_encoder.transform(test_df[col])
        val_df[col] = label_encoder.transform(val_df[col])

    ### Step 4: Remove Outliers Using IQR ###
    def remove_outliers(df):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    train_df = remove_outliers(train_df)
    test_df = remove_outliers(test_df)
    val_df = remove_outliers(val_df)

    ### Step 5: Split Features (X) and Target (y) ###
    X_train, y_train = train_df.drop("Churn", axis=1), train_df["Churn"]
    X_test, y_test = test_df.drop("Churn", axis=1), test_df["Churn"]
    X_val, y_val = val_df.drop("Churn", axis=1), val_df["Churn"]

    ### Step 6: Save Processed Data ###
    train_df.to_csv(processed_path / "processed_train.csv", index=False)
    test_df.to_csv(processed_path / "processed_test.csv", index=False)
    val_df.to_csv(processed_path / "processed_val.csv", index=False)

    print("✅ Data preprocessing complete! Cleaned files saved.")