# Customer Churn Prediction

A machine learning pipeline that predicts whether a telecom customer will churn based on their service usage, contract type, and billing information. Built with a full preprocessing pipeline (imputation + encoding), trained on 7,000+ customers, and evaluated across 3 classifiers with GridSearchCV tuning.

## How It Works

```
Customer Data → Preprocessing Pipeline → One-Hot Encoding → Model Training → Churn Prediction
```

## Dataset

- **Size:** 7,043 customers
- **Features:** 21 attributes (demographics, services, billing)
- **Target:** Churn (Yes/No)
- **Class Distribution:** 73.5% No, 26.5% Yes — imbalanced
- **Missing Values:** 11 rows in `TotalCharges` (handled via median imputation)

## Features

| Demographics | Services | Billing |
|-------------|----------|---------|
| Gender | Phone service | Contract type |
| Senior citizen | Internet service (DSL/Fiber) | Paperless billing |
| Partner | Online security | Payment method |
| Dependents | Online backup | Monthly charges |
| Tenure | Device protection | Total charges |
| | Tech support | |
| | Streaming TV/Movies | |

## Key Preprocessing Steps

- **ColumnTransformer pipeline** with separate handling for numerical and categorical features
- **Numerical:** Median imputation for missing values (11 null TotalCharges)
- **Categorical:** Frequent value imputation + One-Hot Encoding
- **Label encoding** for target variable (Churn: Yes/No → 1/0)
- **Stratified train/test split** (80/20) to preserve class balance

## Models & Results

| Model | Train Accuracy | Test Accuracy | AUROC |
|-------|---------------|---------------|-------|
| **Logistic Regression** | **80.60%** | **80.48%** | **0.8411** |
| Random Forest | 85.46% | 80.34% | 0.8409 |
| K-Nearest Neighbors | 79.87% | 78.99% | 0.8005 |

All models tuned with **10-fold cross-validation GridSearchCV** using ROC-AUC as scoring metric.

## Key Insights from EDA

- **Monthly charges:** Customers paying higher fees are more likely to churn
- **Tenure:** Shorter tenure = higher churn risk
- **Contract type:** Month-to-month contracts churn the most (1,655 vs 166 for one-year and 48 for two-year)
- **Internet service:** Fiber optic users churn at 2.8× the rate of DSL users
- **Payment method:** Electronic check users churn significantly more than other payment methods

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| ML Models | Scikit-learn (RandomForest, KNN, LogisticRegression) |
| Preprocessing | feature-engine (MeanMedianImputer, CategoricalImputer, OneHotEncoder) |
| Pipeline | Scikit-learn Pipeline + ColumnTransformer |
| Visualization | Matplotlib, Seaborn |
| Data Processing | Pandas, NumPy |

## Project Structure

```
├── Customer_Churn_Prediction.ipynb    # Full pipeline notebook
├── customer_churn_data.csv            # Dataset (7,043 customers)
└── README.md
```

## Key Findings

- Logistic Regression and Random Forest achieved nearly identical AUROC (~0.841), but LR generalizes better (smaller train-test gap)
- Random Forest overfits slightly (85.5% train vs 80.3% test)
- KNN underperforms on this dataset due to high-dimensional one-hot encoded features
- Contract type is the strongest business-actionable predictor — offering longer contracts could reduce churn
- The ~80% accuracy ceiling suggests additional features (e.g., customer support interactions, usage patterns) could improve predictions

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn feature-engine
jupyter notebook Customer_Churn_Prediction.ipynb
```

## License

Apache 2.0
