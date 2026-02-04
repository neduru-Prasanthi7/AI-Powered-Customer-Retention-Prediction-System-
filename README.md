# 📊 AI Powered Customer Churn Prediction 

## 📌 Project Overview
Customer churn is a major challenge for telecom companies, as retaining existing customers is significantly more cost-effective than acquiring new ones.  
This project aims to predict customer churn using machine learning by analyzing customer demographics, service usage patterns, contract details, and billing information.

The project implements a complete end-to-end machine learning pipeline, from data preprocessing and exploratory analysis to model deployment using a Flask web application.

---

## 🎯 Project Objectives
- Understand customer behavior through data analysis
- Identify factors influencing customer churn
- Apply feature engineering and selection techniques
- Train and evaluate multiple machine learning models
- Select the best-performing model
- Deploy the model using a Flask-based web application

---

## 🗂 Dataset Information
- **Dataset:** Telco Customer Churn Dataset
- **Target Variable:** `Churn` (Yes / No)
- **Problem Type:** Binary Classification
- **Feature Categories:**
  - Demographic information
  - Service usage details
  - Contract and billing information

---

## 📈 Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset, identify trends, detect anomalies, and uncover relationships between features and customer churn.

### 🔍 Tools Used
- **Pandas:** Data manipulation and preprocessing
- **NumPy:** Numerical computations
- **Matplotlib & Seaborn:** Data visualization

### 📊 Visualizations Created
- Bar charts
- Pie charts
- Histograms
- Count plots
- Box plots

### 🔑 Key Insights from EDA
- **Churn Distribution:**  
  Approximately 26.5% of customers churned, while 73.5% retained their services.
- **Gender-wise Churn:**  
  Churn rate is almost equally distributed between male and female customers.
- **Tenure vs Churn:**  
  Customers with shorter tenure show significantly higher churn rates, while long-term customers are more loyal.
- **Service Usage:**  
  Most churned customers used phone and internet services.
- **Value-Added Services:**  
  Customers subscribed to multiple services tend to churn less.
- **Payment and Billing Patterns:**  
  Customers using electronic payment methods show higher churn tendencies.

EDA insights guided feature engineering and model selection decisions.

---

## 🛠 Feature Engineering

### 🔹 Handling Missing Values
Missing values were identified mainly in the `TotalCharges` column due to incorrect data formatting.  
The following imputation techniques were evaluated:

- Mean Imputer
- Median Imputer
- **Constant Imputer (Selected)**
- KNN Imputer

The **Constant Imputer** was selected as it preserved dataset size and delivered stable model performance.

---

### 🔹 Data Separation
- Numerical and categorical features were separated using `select_dtypes()`
- This ensured appropriate preprocessing techniques for each feature type

---

### 🔹 Outlier Detection and Handling
- Outliers were detected using the **IQR (Interquartile Range) method**
- Applied techniques:
  - **Clipping (Winsorization)** to limit extreme values
  - **Outlier Removal (Trimming)** when necessary
- Box plots were used to validate outlier handling

---

## 🔍 Feature Selection
Feature selection was applied to remove irrelevant and redundant features, improving model performance and interpretability.

### Methods Used:
- Constant Feature Removal
- Quasi-Constant Feature Removal
- Chi-Square Test
- Pearson Correlation Test

---

## 🔠 Categorical Encoding
Categorical variables were converted into numerical format using the following encoding techniques:

- One-Hot Encoding
- Ordinal Encoding
- **Target Encoding (Selected)**
- Binary Encoding
- Hashing Encoder

Target Encoding was preferred for handling high-cardinality categorical features efficiently.

---

## ⚖ Data Balancing
The dataset showed class imbalance between churned and non-churned customers.

- **SMOTE (Synthetic Minority Over-sampling Technique)** was applied
- SMOTE generates synthetic samples instead of duplicating existing ones
- Improved minority class prediction and model generalization

---

## 📏 Feature Scaling
Feature scaling was applied to ensure all features contribute equally to model training.

### Scaling Techniques Evaluated:
- StandardScaler
- **Min-Max Scaler (Selected)**
- Robust Scaler
- MaxAbs Scaler

Min-Max Scaling provided consistent performance across multiple models.

---

## 🤖 Model Training
The churn prediction task was treated as a **binary classification problem**.

### Models Trained:
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- Support Vector Machine (SVM)

### Evaluation Metrics:
- Accuracy
- ROC Curve
- AUC Score

---

## 🏆 Best Model – Logistic Regression

Logistic Regression achieved the best overall performance.

| Model | Test Accuracy | ROC-AUC |
|------|--------------|--------|
| Logistic Regression | **0.8211** | **0.8614** |

### Why Logistic Regression?
- Highest ROC-AUC score
- Simple and interpretable
- Less prone to overfitting
- Fast training and prediction
- Provides probabilistic outputs for decision-making

---

## ⚙ Hyperparameter Tuning
Grid Search was applied to optimize Logistic Regression parameters:

- **C:** 10
- **Penalty:** l2
- **Solver:** lbfgs
- **Max Iterations:** 500

### Final Performance:
- Cross-validated ROC-AUC: 0.8402
- Test ROC-AUC: 0.8614
- Test Accuracy: 0.8211

---

## 🌐 Model Deployment
- Developed a **Flask-based web application**
- Users can input customer details through a web form
- The model predicts churn in real time
- The trained model is serialized using **pickle**

---

## ✅ Results
- Accurate and reliable churn prediction
- Strong generalization performance
- Fully deployable end-to-end ML pipeline

---

## 🧾 Conclusion
This project demonstrates an effective machine learning approach to predicting customer churn in the telecom industry.  
By combining exploratory data analysis, robust feature engineering, feature selection, class balancing, and model optimization, Logistic Regression emerged as a reliable and interpretable solution.  
The deployed web application enables proactive customer retention strategies and bridges the gap between data science and business decision-making.

---

## 🚀 Future Enhancements
- Real-time customer data integration
- Additional behavioral and engagement features
- Advanced ensemble and deep learning models
- Automated hyperparameter optimization
- Customer segment-wise churn analysis
- CRM system integration

---

## 📌 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Flask
- Pickle
# AI-Powered-Customer-Retention-Prediction-System-
