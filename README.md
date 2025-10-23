# End-to-End Customer Churn Prediction 📊

This project is a complete, end-to-end data science workflow for predicting customer churn using the [Telco Customer Churn dataset from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

The project is broken into three modular notebooks, demonstrating a clean, professional workflow that separates data exploration, preprocessing, and modeling.

---

## 🚀 Project Goal

The primary goal is to build a high-performing, interpretable machine learning model that accurately identifies customers at a high risk of churning. The project prioritizes the **Recall** metric to minimize the number of "False Negatives" (i.e., failing to identify a customer who then churns), as this is the most costly error for the business.

---

## 📁 Project Structure

The project is divided into three distinct phases, each in its own notebook:

### 1. `01_Data_Exploration_and_EDA.ipynb`
* **Purpose:** To perform a thorough Exploratory Data Analysis (EDA) of the raw data.
* **Key Actions:**
    * Loaded the dataset and identified data quality issues (e.g., `TotalCharges` as an object).
    * Visualized the distributions of numerical features (`tenure`, `MonthlyCharges`).
    * Analyzed categorical features to find key insights, identifying `Contract` as a major churn predictor.
    * Uncovered the significant **class imbalance** in the `Churn` target variable.

### 2. `02_Preprocessing_and_Feature_Engineering.ipynb`
* **Purpose:** To build a robust ETL (Extract, Transform, Load) pipeline.
* **Key Actions:**
    * Cleaned and imputed data (handled missing `TotalCharges`).
    * Used `train_test_split` with stratification to prevent data leakage.
    * Constructed an advanced `ColumnTransformer` and `Pipeline` to:
        * **StandardScale** numerical features.
        * **OneHotEncode** categorical features.
    * Saved the final processed data and the `preprocessor` object for the modeling notebook.

### 3. `03_Modeling_and_Evaluation.ipynb`
* **Purpose:** To perform data mining, train multiple models, and select the best one.
* **Key Actions:**
    * **Baseline Model:** Trained a `LogisticRegression` model to establish initial performance.
    * **Imbalance Handling:** Diagnosed poor recall in the baseline and applied the **class weighting** technique (`scale_pos_weight`) in XGBoost to address the dataset's imbalance.
    * **Hyperparameter Tuning:** Used `RandomizedSearchCV` to find the optimal parameters for the XGBoost model, dramatically improving recall from 56% to **81%**.
    * **Model Explainability:** Used **SHAP (SHapley Additive exPlanations)** to interpret the final model, identifying the top features that drive churn predictions.

---

## 🛠️ How to Run

To reproduce this project, follow these steps:

1.  **Clone the Repository:**
    ```sh
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```
2.  **Add Data:**
    * Download the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
    * Place the `.csv` file inside the `data/` folder.
3.  **Run Notebooks in Order:**
    * You can explore **`01_Data_Exploration_and_EDA.ipynb`** (no outputs are required).
    * Run **`02_Preprocessing_and_Feature_Engineering.ipynb`** top-to-bottom. This will generate the processed data files (e.g., `.npy` files) in its output directory, which are used by the next notebook.
    * Run **`03_Modeling_and_Evaluation.ipynb`** top-to-bottom to train the models and see the final results. *Note: You may need to adjust the file paths in Notebook 3 to point to the output of Notebook 2.*

---

## ✨ Key Techniques & Concepts

* **Data Warehousing:** Modular ETL pipeline (`ColumnTransformer`), data cleaning, and feature scaling.
* **Data Mining:** Classification, addressing class imbalance, model evaluation (Precision-Recall trade-off).
* **Machine Learning:** Logistic Regression, XGBoost, Hyperparameter Tuning (`RandomizedSearchCV`), and Class Weighting.
* **Model Explainability:** Feature importance and decision analysis using SHAP.

