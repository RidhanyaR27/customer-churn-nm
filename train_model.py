import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# === Load and preprocess data ===
df = pd.read_csv("preprocessed_train.csv")

# Drop unwanted columns
columns_to_drop = [
    'customer_id',
    'signup_date',
    'weekly_songs_played',
    'num_platform_friends',
    'num_playlists_created',
    'num_shared_playlists'
]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Split into features and labels
X = df.drop(columns=["churned"])
y = df["churned"]

# Save the list of features for use in app.py
joblib.dump(X.columns.tolist(), "model_features.pkl")

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Train Random Forest ===
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)

print("\n🔍 Random Forest Results:")
print("Accuracy:", accuracy_score(y_val, rf_preds))
print("Confusion Matrix:\n", confusion_matrix(y_val, rf_preds))
print("Classification Report:\n", classification_report(y_val, rf_preds))

# Save Random Forest model
joblib.dump(rf_model, "model_rf.pkl")

# === Train XGBoost ===
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_val)

print("\n⚡ XGBoost Results:")
print("Accuracy:", accuracy_score(y_val, xgb_preds))
print("Confusion Matrix:\n", confusion_matrix(y_val, xgb_preds))
print("Classification Report:\n", classification_report(y_val, xgb_preds))

# Save XGBoost model
joblib.dump(xgb_model, "model_xgb.pkl")

print("\n✅ Both models and feature list saved successfully!")

