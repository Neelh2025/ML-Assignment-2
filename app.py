import streamlit as st
import pandas as pd

from sklearn.metrics import classification_report

st.title("ML Assignment 2 – Classification Models")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

# Precomputed metrics from your notebook
model_metrics = {
    "Logistic Regression": {"Accuracy": 0.8477, "Precision": 0.7215, "Recall": 0.5982, "F1": 0.6541, "AUC": 0.8913, "MCC": 0.5616},
    "Decision Tree": {"Accuracy": 0.8151, "Precision": 0.6112, "Recall": 0.6378, "F1": 0.6242, "AUC": 0.7546, "MCC": 0.5019},
    "k-NN": {"Accuracy": 0.7826, "Precision": 0.5844, "Recall": 0.3355, "F1": 0.4263, "AUC": 0.6794, "MCC": 0.3219},
    "Naive Bayes": {"Accuracy": 0.7993, "Precision": 0.6776, "Recall": 0.3176, "F1": 0.4325, "AUC": 0.8402, "MCC": 0.3644},
    "Random Forest": {"Accuracy": 0.8569, "Precision": 0.7366, "Recall": 0.6314, "F1": 0.6799, "AUC": 0.9074, "MCC": 0.5914},
    "XGBoost": {"Accuracy": 0.8775, "Precision": 0.7917, "Recall": 0.6665, "F1": 0.7237, "AUC": 0.9314, "MCC": 0.6497},
}

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # fix Adult column spacing issues

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Select Classification Model")
    model_name = st.selectbox("Choose a model", list(model_metrics.keys()))

    st.subheader("Model Evaluation Metrics")
    metrics = model_metrics[model_name]
    for metric, value in metrics.items():
        st.write(f"**{metric}:** {value}")

    st.subheader("Classification Report")

    # Identify target column safely (Adult dataset usually has income as last column)
    target_col = "income" if "income" in df.columns else df.columns[-1]

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Binary encoding for target (Adult labels are typically <=50K and >50K)
    y_binary = (y.astype(str).str.strip() == ">50K").astype(int)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    # One-hot encode categorical variables
    X_train_enc = pd.get_dummies(X_train)
    X_test_enc = pd.get_dummies(X_test)

    # Align columns so train/test have identical features
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

    # Build the selected model
    if model_name == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "k-NN":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Naive Bayes":
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    elif model_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
    else:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )

    model.fit(X_train_enc, y_train)
    y_pred = model.predict(X_test_enc)

    report = classification_report(y_test, y_pred)
    st.text(report)

else:
    st.info("Please upload a CSV file to continue.")
