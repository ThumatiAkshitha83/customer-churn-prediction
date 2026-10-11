import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("Churn_Modelling.csv")

# -----------------------------
# 2. Drop unnecessary columns
# -----------------------------
df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

# -----------------------------
# 3. Features & target
# -----------------------------
X = df.drop("Exited", axis=1)
y = df["Exited"]

# -----------------------------
# 4. Column types
# -----------------------------
numeric_features = [
    "CreditScore", "Age", "Tenure",
    "Balance", "EstimatedSalary", "NumOfProducts"
]

categorical_features = ["Geography", "Gender"]

# -----------------------------
# 5. Preprocessing
# -----------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------------
# 6. Pipeline
# -----------------------------
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        class_weight={0:1, 1:2},
        random_state=42
    ))
])

# -----------------------------
# 7. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 8. Train model
# -----------------------------
model_pipeline.fit(X_train, y_train)

# -----------------------------
# 9. Evaluate model
# -----------------------------
y_pred = model_pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =====================================================
# 🔥 10. FEATURE IMPORTANCE (THIS IS WHAT YOU ASKED)
# =====================================================

# Get feature names after preprocessing
feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()

# Get importance values from Random Forest
importances = model_pipeline.named_steps["classifier"].feature_importances_

# Create DataFrame
feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop Features Influencing Churn:\n")
print(feat_df.head(10))

# -----------------------------
# 11. Save full pipeline
# -----------------------------
with open("churn_pipeline.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("\n✅ churn_pipeline.pkl created successfully")