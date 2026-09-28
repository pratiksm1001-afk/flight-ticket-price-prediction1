# ============================================================
# FLIGHT TICKET PRICE PREDICTION USING XGBOOST
# train_model.py
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = r"D:\mini project\flight ticket price prediction"

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Clean_Dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("        FLIGHT TICKET PRICE PREDICTION")
print("              USING XGBOOST")
print("=" * 70)

print("\n🤖 Initializing Machine Learning Engine...")

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print("\n❌ Dataset not found!")
    print(f"Expected location:\n{DATASET_PATH}")
    exit()


print("\n✅ Dataset loaded successfully.")

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 3. REMOVE UNNECESSARY COLUMN
# ============================================================

if "Unnamed: 0" in df.columns:
    df.drop(
        "Unnamed: 0",
        axis=1,
        inplace=True
    )

print("\nDataset columns:")
print(df.columns.tolist())


# ============================================================
# 4. CHECK TARGET COLUMN
# ============================================================

if "price" not in df.columns:
    print("\n❌ ERROR: 'price' column not found.")
    print("Available columns:")
    print(df.columns.tolist())
    exit()


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

print("\n🔍 Checking missing values...")

missing_values = df.isnull().sum()

total_missing = missing_values.sum()

if total_missing > 0:

    print("\nMissing values found:")

    print(
        missing_values[
            missing_values > 0
        ]
    )

    # Fill numerical columns
    numerical_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for column in numerical_columns:

        if column != "price":

            df[column] = df[column].fillna(
                df[column].median()
            )

    # Fill categorical columns
    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in categorical_columns:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )

    print("\n✅ Missing values handled.")

else:

    print("✅ No missing values found.")


# ============================================================
# 6. REMOVE INVALID TARGET VALUES
# ============================================================

df = df.dropna(
    subset=["price"]
)

df = df[
    df["price"] > 0
]

print(
    f"\nRows after cleaning: {len(df)}"
)


# ============================================================
# 7. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    "price",
    axis=1
)

y = df["price"]


print("\n🎯 Target variable: price")

print(
    f"Minimum price : ₹{y.min():,.2f}"
)

print(
    f"Maximum price : ₹{y.max():,.2f}"
)

print(
    f"Average price : ₹{y.mean():,.2f}"
)


# ============================================================
# 8. IDENTIFY CATEGORICAL COLUMNS
# ============================================================

categorical_cols = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    include=[np.number]
).columns.tolist()


print("\nCategorical columns:")
print(categorical_cols)

print("\nNumerical columns:")
print(numerical_cols)


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n📊 Data split completed.")

print(
    f"Training samples : {len(X_train)}"
)

print(
    f"Testing samples  : {len(X_test)}"
)


# ============================================================
# 10. ENCODE CATEGORICAL FEATURES
# ============================================================

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()


if len(categorical_cols) > 0:

    X_train_encoded[
        categorical_cols
    ] = encoder.fit_transform(
        X_train[categorical_cols]
    )

    X_test_encoded[
        categorical_cols
    ] = encoder.transform(
        X_test[categorical_cols]
    )


# ============================================================
# 11. TRAIN XGBOOST MODEL
# ============================================================

print("\n🧠 Training XGBoost model...")
print("Please wait...\n")


model = XGBRegressor(

    n_estimators=500,

    learning_rate=0.05,

    max_depth=8,

    min_child_weight=2,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train_encoded,
    y_train
)


print("✅ XGBoost model training completed.")


# ============================================================
# 12. MAKE PREDICTIONS
# ============================================================

print("\n🔮 Making predictions...")

y_pred = model.predict(
    X_test_encoded
)


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n")
print("=" * 70)
print("                 MODEL EVALUATION")
print("=" * 70)

print(
    f"\nMAE  : ₹{mae:,.2f}"
)

print(
    f"RMSE : ₹{rmse:,.2f}"
)

print(
    f"R²   : {r2:.4f}"
)

print("=" * 70)


# ============================================================
# 14. DISPLAY SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({

    "Actual Price": y_test.values,

    "Predicted Price": y_pred

})

results["Difference"] = (
    results["Actual Price"]
    - results["Predicted Price"]
)


print("\n📋 SAMPLE PREDICTIONS")

print(
    results.head(10).to_string(
        index=False
    )
)


# ============================================================
# 15. SAVE ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.xlabel(
    "Actual Flight Price"
)

plt.ylabel(
    "Predicted Flight Price"
)

plt.title(
    "Actual vs Predicted Flight Ticket Prices"
)

# Perfect prediction line
min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.tight_layout()

actual_predicted_path = os.path.join(
    OUTPUT_DIR,
    "actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_path,
    dpi=300
)

plt.close()


print(
    f"\n📈 Actual vs Predicted graph saved:"
)

print(
    actual_predicted_path
)


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\n")
print("=" * 70)
print("                 FEATURE IMPORTANCE")
print("=" * 70)

print(
    feature_importance.to_string(
        index=False
    )
)


# ============================================================
# 17. SAVE FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "XGBoost Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

feature_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.png"
)

plt.savefig(
    feature_path,
    dpi=300
)

plt.close()


print(
    f"\n📊 Feature importance graph saved:"
)

print(
    feature_path
)


# ============================================================
# 18. SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "xgboost_flight_model.pkl"
)

with open(
    model_path,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


print(
    f"\n💾 Model saved:"
)

print(
    model_path
)


# ============================================================
# 19. SAVE ENCODER
# ============================================================

encoder_path = os.path.join(
    MODEL_DIR,
    "ordinal_encoder.pkl"
)

with open(
    encoder_path,
    "wb"
) as file:

    pickle.dump(
        encoder,
        file
    )


print(
    f"\n💾 Encoder saved:"
)

print(
    encoder_path
)


# ============================================================
# 20. SAVE FEATURE INFORMATION
# ============================================================

feature_info = {

    "all_columns": X.columns.tolist(),

    "categorical_columns": categorical_cols,

    "numerical_columns": numerical_cols

}


feature_info_path = os.path.join(
    MODEL_DIR,
    "feature_info.pkl"
)

with open(
    feature_info_path,
    "wb"
) as file:

    pickle.dump(
        feature_info,
        file
    )


print(
    f"\n💾 Feature information saved:"
)

print(
    feature_info_path
)


# ============================================================
# 21. SAVE EVALUATION RESULTS
# ============================================================

evaluation = {

    "MAE": mae,

    "RMSE": rmse,

    "R2_Score": r2

}


evaluation_path = os.path.join(
    OUTPUT_DIR,
    "evaluation_results.pkl"
)

with open(
    evaluation_path,
    "wb"
) as file:

    pickle.dump(
        evaluation,
        file
    )


# ============================================================
# 22. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)

print(
    "🎉 FLIGHT PRICE PREDICTION MODEL READY!"
)

print("=" * 70)

print("\nGenerated files:")

print(
    "📁 models/"
)

print(
    "   ├── xgboost_flight_model.pkl"
)

print(
    "   ├── ordinal_encoder.pkl"
)

print(
    "   └── feature_info.pkl"
)

print(
    "\n📁 outputs/"
)

print(
    "   ├── actual_vs_predicted.png"
)

print(
    "   ├── feature_importance.png"
)

print(
    "   └── evaluation_results.pkl"
)

print("\n✅ Training completed successfully!")