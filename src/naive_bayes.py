import numpy as np
import pandas as pd
import datetime
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_DIR / "insurance_fraud_data" / "insurance_claim_fraud_dataset.csv"
)

RESULTS_DIR = PROJECT_DIR / "fraud_prediction_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)

TARGET = "fraudulent"
print("\nFraud distribution:")
print(df[TARGET].value_counts(dropna=False))
print("\nFraud distribution (%):")
print((df[TARGET].value_counts(normalize=True) * 100).round(2))

dropped_cols = [
    "claim_id", "policy_id", "customer_id", "customer_name",
    "claim_status", "gps_location_match", "repair_shop_verified", 
    "identity_verified", "delayed_reporting", "claim_history_risk",
    "suspicious_documents", "claim_submitted_at_night",
    "inconsistent_statements", "fraud_risk_score" 
]
df= df.drop(columns=dropped_cols)

df['claim_date'] = pd.to_datetime(df['claim_date'])
df['incident_date'] = pd.to_datetime(df['incident_date'])
df['policy_start_date'] = pd.to_datetime(df['policy_start_date'])


df['claim_date_dow'] = df['claim_date'].dt.day_of_week
df['claim_date_wom'] = (df['claim_date'].dt.days_in_month // 7)

df['incident_date_dow'] = df['incident_date'].dt.day_of_week
df['incident_date_wom'] = (df['incident_date'].dt.days_in_month // 7)

df['policy_start_date_dow'] = df['policy_start_date'].dt.day_of_week
df['policy_start_date_wom'] = (df['policy_start_date'].dt.days_in_month // 7)

date_categories = ['claim_date', 'incident_date', 'policy_start_date']
df= df.drop(columns=date_categories)

# ordinal categories
edu_mapper = {"High School":1, "Associate":2, "Bachelor":3, "Master": 4, "Doctorate": 5}
df["education_level"] = df["education_level"].replace(edu_mapper)
convert_edu = {"education_level":int}
df = df.astype(convert_edu)

deductible_mapper = {"Low ($250)": 1, "Standard ($500)": 2, "High ($1,000)": 3, "Premium ($2,500)": 4}
df["deductible_plan"] = df["deductible_plan"].replace(deductible_mapper)
con_deductible = {"deductible_plan":int}
df = df.astype(con_deductible)

acc_sev_mapper = {"Minor": 1, "Moderate": 2, "Major": 3, "Total Loss": 4}
df["accident_severity"] = df["accident_severity"].replace(acc_sev_mapper)
con_acc = {"accident_severity": int}
df = df.astype(con_acc)

unordered_categories = ['gender', 'marital_status', 'occupation', 'policy_type', 
                        'payment_method', 'incident_type', 'incident_location', 'weather_condition',
                        'vehicle_damage', 'vehicle_make', 'vehicle_model', 'ownership_type'
                        ]
df = pd.get_dummies(df, columns=unordered_categories, dtype='int')

yesno_categories = ['policy_active', 'police_report', 'hospital_visit', 'property_damage', 'multiple_claims_last_year',
                     'total_loss', 'fraudulent']
yesno_mapper = {"Yes": 1, "No": 0}
df[yesno_categories] = df[yesno_categories].replace(yesno_mapper)
convert_map = {'policy_active': int, 'police_report':int, 'hospital_visit':int,
                'property_damage':int, 'multiple_claims_last_year':int,
                'total_loss':int, 'fraudulent':int
                }
df = df.astype(convert_map)

df = df[[col for col in df.columns if col != 'fraudulent'] + ['fraudulent']]

# Train and Test Naive Bayes Model
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)
y_pred = naive_bayes_model.predict(X_test)
y_prob = naive_bayes_model.predict_proba(X_test)[:, 1]

# Model Evaluation
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
print("Naive Bayes Classification Results")
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


# Save evaluation results
results = pd.DataFrame([{
    "Algorithm": "Naive Bayes",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "ROC-AUC": roc_auc
}])

results_path = RESULTS_DIR / "naive_bayes_results.csv"

results.to_csv(results_path, index=False)

print(f"\nResults saved to: {results_path}")