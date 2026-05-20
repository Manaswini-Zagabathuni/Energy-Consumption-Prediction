#  Energy Consumption Prediction

A machine learning project that predicts residential and commercial **energy consumption** using regression techniques, regularization strategies, and polynomial feature expansion.
 
---

##  Project Overview

This project builds and evaluates several linear regression models to predict energy usage based on building characteristics and environmental factors. It covers the full ML pipeline — from exploratory data analysis to final model evaluation on unseen test data.

**Best Model:** Polynomial (Degree 2) Lasso Regression  
**Key Metric:** Root Mean Squared Error (RMSE) via 4-Fold Cross-Validation

---

##  Project Structure

├── code.ipynb              # Main Jupyter Notebook (full pipeline)

├── train_energy_data.csv   # Training dataset

├── test_energy_data.csv    # Test dataset

└── README.md

---

##  Features Used

| Feature | Type |
|---|---|
| Square Footage | Numerical |
| Number of Occupants | Numerical |
| Appliances Used | Numerical |
| Average Temperature | Numerical |
| Building Type | Categorical |
| Day of Week | Categorical |

**Target:** `Energy Consumption`

---

##  Methodology

### 1. Exploratory Data Analysis (EDA)
- Statistical summary of training data
- Distribution plots (histograms + KDE) for continuous features
- Count plots for categorical features
- Pearson Correlation Coefficient (PCC) heatmap

### 2. Preprocessing
- `StandardScaler` for numerical features
- `OneHotEncoder` for categorical features
- `ColumnTransformer` pipeline for clean integration

### 3. Model Evaluation
- **4-Fold Cross-Validation** for all models
- Models compared: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`
- Regularization strengths tested: `α ∈ {0.0001, 1.0, 1000.0}`
- Optimizer: Stochastic Gradient Descent (SGD)

### 4. Learning Curve Analysis
- Train vs. validation RMSE tracked per epoch
- Best learning rate: `η₀ = 1e-3`

### 5. Polynomial Feature Expansion
- Degree-2 polynomial features on numerical inputs
- Interaction and squared terms to capture non-linear patterns
- Final model: Lasso with `α = 0.1` on expanded features

---

##  How to Run

### Prerequisites

pip install pandas numpy matplotlib seaborn scikit-learn jupyter

### Run the Notebook

jupyter notebook code.ipynb

Make sure train_energy_data.csv and test_energy_data.csv are in the same directory as the notebook.

---

##  Results Summary

|     Model            | Regularization |   Alpha  |  Mean CV RMSE  |
|----------------------|----------------|----------|----------------|
|   Linear Regression  |     None       |    —     |    baseline    |
|       Ridge          |      L2        |   0.0001 |       —        |
|       Lasso          |      L1        |   0.0001 |  best (linear) |
|     ElasticNet       |    L1 + L2     |   0.0001 |       —        |
|**Poly Lasso (Deg 2)**|    **L1**      | **0.1**  |**best overall**|

---

##  Tech Stack
 
- **Python 3**
- **pandas**, **NumPy** — data manipulation
- **matplotlib**, **seaborn** — visualization
- **scikit-learn** — preprocessing, modeling, evaluation
