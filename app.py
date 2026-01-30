import streamlit as st
import pandas as pd

st.title("ML Assignment 2 – Classification Models")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

# Precomputed metrics from ML notebook
model_metrics = {
    "Logistic Regression": {
        "Accuracy": 0.8477,
        "Precision": 0.7215,
        "Recall": 0.5982,
        "F1": 0.6541,
        "AUC": 0.8913,
        "MCC": 0.5616
    },
    "Decision Tree": {
        "Accuracy": 0.8151,
        "Precision": 0.6112,
        "Recall": 0.6378,
        "F1": 0.6242,
        "AUC": 0.7546,
        "MCC": 0.5019
    },
    "k-NN": {
        "Accuracy": 0.7826,
        "Precision": 0.5844,
        "Recall": 0.3355,
        "F1": 0.4263,
        "AUC": 0.6794,
        "MCC": 0.3219
    },
    "Naive Bayes": {
        "Accuracy": 0.7993,
        "Precision": 0.6776,
        "Recall": 0.3176,
        "F1": 0.4325,
        "AUC": 0.8402,
        "MCC": 0.3644
    },
    "Random Forest": {
        "Accuracy": 0.8569,
        "Precision": 0.7366,
        "Recall": 0.6314,
        "F1": 0.6799,
        "AUC": 0.9074,
        "MCC": 0.5914
    },
    "XGBoost": {
        "Accuracy": 0.8775,
        "Precision": 0.7917,
        "Recall": 0.6665,
        "F1": 0.7237,
        "AUC": 0.9314,
        "MCC": 0.6497
    }
}

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Select Classification Model")
    model_name = st.selectbox("Choose a model", list(model_metrics.keys()))

    st.subheader("Model Evaluation Metrics")

    metrics = model_metrics[model_name]
    for metric, value in metrics.items():
        st.write(f"**{metric}:** {value}")

else:
    st.info("Please upload a CSV file to continue.")
