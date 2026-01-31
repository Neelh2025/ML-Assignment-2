import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    classification_report
)

# --------------------------------------------------
# App UI
# --------------------------------------------------
st.title("ML Assignment 2 – Classification Models")
st.info(
    "This interactive application allows users to compare multiple machine learning "
    "classification models on the Adult Income dataset."
)

# --------------------------------------------------
# Upload CSV
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
data_df = pd.read_csv(uploaded_file)

# Clean column names and string values
data_df.columns = data_df.columns.str.strip()
for col in data_df.select_dtypes(include="object").columns:
    data_df[col] = data_df[col].str.strip()

st.subheader("Dataset Preview")
st.dataframe(data_df.head())

# --------------------------------------------------
# Detect target column robustly
# --------------------------------------------------
target_candidates = [
    col for col in data_df.columns if col.strip().lower() == "income"
]

if len(target_candidates) == 0:
    st.success("Dataset uploaded successfully.")
    st.warning(
        "Target column 'income' not found in the uploaded CSV. "
        "Metrics and classification report are shown only when the target column is available."
    )
    st.stop()

target_col = target_candidates[0]

# --------------------------------------------------
# Prepare features and target
# --------------------------------------------------
features_df = data_df.drop(columns=[target_col])
target_series = data_df[target_col]
target_binary = (target_series == ">50K").astype(int)

# --------------------------------------------------
# Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    features_df,
    target_binary,
    test_size=0.2,
    random_state=42,
    stratify=target_binary
)

# --------------------------------------------------
# Column processing
# --------------------------------------------------
categorical_cols = X_train.select_dtypes(include="object").columns
numeric_cols = X_train.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# --------------------------------------------------
# Model selection
# --------------------------------------------------
st.subheader("Select Classification Model")

model_choice = st.selectbox(
    "Choose a model",
    [
        "Logistic Regression",
        "Decision Tree",
        "k-NN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

model_dict = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
}

selected_model = model_dict[model_choice]

# --------------------------------------------------
# Build and train pipeline
# --------------------------------------------------
model_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", selected_model)
    ]
)

model_pipeline.fit(X_train, y_train)

# --------------------------------------------------
# Predictions
# --------------------------------------------------
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# Metrics
# --------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
mcc = matthews_corrcoef(y_test, y_pred)

# --------------------------------------------------
# Display metrics
# --------------------------------------------------
st.subheader("Model Evaluation Metrics")
st.write(f"Accuracy: {accuracy:.4f}")
st.write(f"Precision: {precision:.4f}")
st.write(f"Recall: {recall:.4f}")
st.write(f"F1: {f1:.4f}")
st.write(f"AUC: {auc:.4f}")
st.write(f"MCC: {mcc:.4f}")

# --------------------------------------------------
# Classification report
# --------------------------------------------------
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))
