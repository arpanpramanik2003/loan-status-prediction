# Loan Status Prediction Model

## Project Overview
This project focuses on predicting loan approval status using machine learning techniques. The dataset contains various features related to loan applicants, and the goal is to classify whether a loan will be approved or not.

## Dataset
- The dataset used is `Loan_status.csv`.
- It consists of 614 rows and 13 columns.
- Key features include applicant's income, loan amount, credit history, and property area.

## Data Preprocessing
1. Checked for null values and removed rows with missing data.
2. Converted categorical values into numerical values:
   - Loan_Status: 'Y' to 1, 'N' to 0
   - Dependents: '3+' to 4
3. Split the data into features (X) and target (Y).
4. Defined categorical and numerical columns for preprocessing.

## Model Training
1. Used `StandardScaler` for numerical feature scaling.
2. Applied `OneHotEncoder` for categorical feature encoding.
3. Built an SVM model with a linear kernel.
4. Created a pipeline for data preprocessing and model training.

## Model Evaluation
- **Training Accuracy:** 81.02%
- **Testing Accuracy:** 79.17%

## Model Deployment
- The trained model is saved as `loan_status_model.pkl` using the `pickle` module.

## Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pickle

## Usage
1. Load the trained model from `loan_status_model.pkl`.
2. Provide applicant data for prediction.
3. Get loan approval status (1 for approved, 0 for not approved).

## Conclusion
This loan status prediction model can help financial institutions automate loan approval decisions based on applicant details, improving efficiency and accuracy.
