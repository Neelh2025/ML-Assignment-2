## Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models on a real-world dataset and deploy the results using an interactive Streamlit web application. 

The project involves implementing six different classification algorithms, comparing their performance using standard evaluation metrics, and presenting the results through a user-friendly interface. This assignment demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation, visualization, and deployment.

## Dataset Description

The dataset used for this assignment is the Adult Income Dataset obtained from the UCI Machine Learning Repository. The dataset is a binary classification problem where the objective is to predict whether an individual's annual income exceeds $50K based on demographic and employment-related attributes.

The dataset contains 32,561 instances and 14 input features along with one target variable. The features include both numerical and categorical attributes such as age, education level, occupation, working hours per week, marital status, and capital gain/loss.

The target variable is:
- **Income**: >50K or <=50K

The dataset satisfies the assignment requirements of having more than 500 instances and more than 12 features, making it suitable for training and evaluating multiple classification models.

## Models Used and Performance Comparison

The following six machine learning classification models were implemented and evaluated on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. k-Nearest Neighbors (k-NN)  
4. Naive Bayes Classifier  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

Each model was evaluated using the following metrics:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Performance Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8477 | 0.8913 | 0.7215 | 0.5982 | 0.6541 | 0.5616 |
| Decision Tree | 0.8151 | 0.7546 | 0.6112 | 0.6378 | 0.6242 | 0.5019 |
| k-NN | 0.7826 | 0.6794 | 0.5844 | 0.3355 | 0.4263 | 0.3219 |
| Naive Bayes | 0.7993 | 0.8402 | 0.6776 | 0.3176 | 0.4325 | 0.3644 |
| Random Forest | 0.8569 | 0.9074 | 0.7366 | 0.6314 | 0.6799 | 0.5914 |
| XGBoost | 0.8775 | 0.9314 | 0.7917 | 0.6665 | 0.7237 | 0.6497 |


## Observations on Model Performance

During this assignment, I observed that ensemble models such as Random Forest and XGBoost performed better due to their ability to capture complex feature interactions.

| Model | Observation |
|------|-------------|
| Logistic Regression | Logistic Regression provided a strong baseline performance with good overall accuracy and AUC. It performed well in distinguishing between income classes but showed moderate recall for the high-income class. |
| Decision Tree | The Decision Tree model captured non-linear relationships in the data but showed signs of overfitting, leading to slightly lower generalization performance compared to ensemble models. |
| k-NN | The k-NN classifier showed lower performance, particularly in recall and F1 score, indicating sensitivity to feature scaling and the curse of dimensionality in higher-dimensional data. |
| Naive Bayes | Naive Bayes achieved reasonable accuracy and AUC but had low recall for the positive class due to its strong independence assumptions among features. |
| Random Forest | Random Forest significantly improved performance by combining multiple decision trees, resulting in higher accuracy, better recall, and strong overall robustness. |
| XGBoost | XGBoost achieved the best overall performance across all metrics, benefiting from gradient boosting, regularization, and effective handling of complex feature interactions. |

