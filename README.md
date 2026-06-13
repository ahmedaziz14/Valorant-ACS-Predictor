# Valorant ACS Predictor - Machine Learning Pipeline

A robust, end-to-end Multiple Linear Regression pipeline built from scratch using Python and `scikit-learn` to predict a player's **Average Combat Score (ACS)** in Valorant matches. 

The project evaluates the mathematical impact of in-game performance metrics (ADR, Headshot %, and Utility usage) while demonstrating industrial best practices like data standardization and K-Fold cross-validation.

---

## 📌 Project Overview & Objectives

In competitive tactical shooters like Valorant, evaluating a player's performance goes beyond raw kills. This project models the hidden formula behind the **Average Combat Score (ACS)** using a supervised learning approach.

The core objective is to reverse-engineer the relationship between a player's raw performance metrics and their final match impact score, creating an **interpretable AI model** where every coefficient tells a business/gameplay story.

### Dataset Features (The $X$ Matrix)
* **ADR (Average Damage per Round):** Continuous variable representing the average damage dealt.
* **HS_Percent:** Continuous variable tracking the percentage of headshots.
* **Flashes_Reussies:** Discrete variable counting successful enemy blinds using utility (e.g., Phoenix's Curveball).

### The Target ($y$)
* **ACS_Cible:** The final continuous score predicting the overall match impact.

---

## 🛠️ Software Architecture & Pipeline

The project is strictly modularized into 5 engineering sub-parts to ensure scannability and professional code maintenance:

1. **Data Acquisition & EDA:** Simulating 500 competitive matches using realistic statistical distributions (Gaussian distributions for continuous metrics, Poisson distribution for discrete utility counts).
2. **Preprocessing (Feature Scaling):** Implementing `StandardScaler` to normalize feature magnitudes, preventing high-scale features (ADR) from dominating the gradient optimization over low-scale features (Flashes).
3. **Data Splitting:** Enforcing a strict 80/20 Train/Test split to secure an untouched "vault" of validation data.
4. **Model Training:** Fitting a `LinearRegression` model using the Ordinary Least Squares (OLS) analytical approach.
5. **Model Evaluation & Interpretation:** Extracting the learned weights ($\theta$ coefficients) and evaluating performance on unseen data.

---

## 📊 Key Results & Metrics

The model achieves exceptional prediction accuracy on unseen testing data:

* **R² Score (Coefficient of Determination):** `0.894` — The model successfully explains **89.4%** of the variance in the ACS.
* **RMSE (Root Mean Squared Error):** `15.19 points` — On average, the model's predictions deviate by only ~15 ACS points from the ground truth.

### Feature Importance (Standardized Coefficients)
Thanks to the `StandardScaler`, the learned weights directly show the mathematical impact of each standard deviation increase:
* **ADR Impact:** `+44.68` ACS points per standard deviation.
* **HS % Impact:** `+12.36` ACS points per standard deviation.
* **Flashes Impact:** `+3.24` ACS points per standard deviation.
* **Model Bias (Intercept):** `270.23` (Represents the baseline ACS for a perfectly average performance).

---

## 🚀 How to Run the Code

### Prerequisites
Make sure you have Python 3 installed along with the required libraries:
```bash
pip install pandas numpy scikit-learn