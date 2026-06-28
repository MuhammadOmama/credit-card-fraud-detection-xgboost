# Credit Card Fraud Detection: High-Precision Anomaly Modeling

An industry-standard, production-ready machine learning pipeline engineered to intercept fraudulent credit card transactions in real-time. This project uses the legendary **ULB Credit Card Fraud dataset** to solve an extreme class imbalance scenario ($0.172\%$ fraud) using advanced gradient boosting (**XGBoost**) and strategic decision threshold tuning.

## 📊 Dataset Source
The dataset used for this project is publicly available on Kaggle:
🔗 [Kaggle ULB Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 🎯 Project Performance Milestones (Hidden Test Set)
* **Baseline Target Rescue:** Upgraded an unweighted Logistic Regression baseline that missed 36% of all frauds to a highly sensitive, weighted **XGBoost Classifier**, boosting the operational **AUPRC score from 0.7433 to 0.8033**.
* **Zero Catch Rate Decay:** Successfully maintained an **86% Fraud Recall rate** across iterations, capturing 84 out of 98 hidden fraud cases cleanly.
* **Customer Protection Optimization:** By shifting the default classification threshold from `0.50` to **`0.80`**, fraud precision surged to **60%**, eliminating unnecessary false alarms and **saving 124 innocent customers from catastrophic card freezes**.

## 🛠️ Data Pipeline Architecture
1. **Feature Engineering & Leakage Prevention:** Isolated a 20% stratified test set prior to transformation. Scaled the heavily skewed financial `Amount` distribution using a `StandardScaler` to match the pre-computed PCA feature dimensions (`V1-V28`).
2. **Imbalance Mitigation:** Avoided destructive synthetic data oversampling (SMOTE) to eliminate downstream noise. Instead, calculated a native cost-sensitive scale weight (`scale_pos_weight = 577.29`) to penalize minority class omissions heavily.
3. **Boundary Tuning:** Implemented a custom probability threshold function at `0.80` to filter out uncertain false-positive anomalies, maximizing user experience without dropping true positives.

## 🔍 Structural Fraud Drivers (Feature Importance)
The model relies heavily on key underlying variance components extracted from transaction signatures. Below is the relative distribution weight of the top 10 fraud indicators:

 <img width="989" height="590" alt="download" src="https://github.com/user-attachments/assets/a4c203cf-ecf7-42db-8687-566536925869" />


## 🚀 Core Technologies Used
* Python 3.12
* Scikit-Learn (Preprocessing, Metrics, Stratified Splitting)
* XGBoost (Gradient Boosted Decision Trees)
* Matplotlib & Seaborn (Analytical Data Visualization)
* Pandas & NumPy (High-Performance Numerical Wrangling)
