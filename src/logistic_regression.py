
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# 1. Set up project paths

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_DIR / "data" / "insurance_claim_fraud_dataset.csv"
)

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load and inspect the dataset

df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)

TARGET = "fraudulent"

print("\nFraud distribution:")
print(df[TARGET].value_counts(dropna=False))

print("\nFraud distribution (%):")
print((df[TARGET].value_counts(normalize=True) * 100).round(2))


# 3. Convert the target to binary labels

y = (
    df[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)

if y.isna().any():
    raise ValueError(
        "Unexpected values in the fraudulent column. "
        f"Found: {df[TARGET].unique()}"
    )


# 4. Remove identifiers, dates, and potential leakage

columns_to_drop = [
    "claim_id",
    "policy_id",
    "customer_id",
    "customer_name",
    "claim_date",
    "incident_date",
    "policy_start_date",
    "claim_status",
    "approved_amount",
    "payout_amount",
    "fraud_risk_score",
    TARGET
]

X = df.drop(columns=columns_to_drop, errors="ignore")

print("\nExcluded columns:")
print(columns_to_drop)

print("\nRemaining features:", X.shape[1])


# 5. Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 6. Identify numerical and categorical features

numeric_features = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude="number"
).columns.tolist()

print("\nNumerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# 7. Create preprocessing pipelines

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# 8. Create and train Logistic Regression

logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("\nTraining Logistic Regression...")

logistic_model.fit(X_train, y_train)

print("Training complete!")


# 9. Make predictions on unseen test records

y_pred = logistic_model.predict(X_test)

# Probability that each claim is fraudulent
y_prob = logistic_model.predict_proba(X_test)[:, 1]


# 10. Evaluate the model

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 45)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 45)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Legitimate", "Fraudulent"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


# 11. Save evaluation results

results = pd.DataFrame([{
    "Algorithm": "Logistic Regression",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "ROC-AUC": roc_auc
}])

results_path = RESULTS_DIR / "logistic_regression_results.csv"

results.to_csv(results_path, index=False)

print(f"\nResults saved to: {results_path}")
