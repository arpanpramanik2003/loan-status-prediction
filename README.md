# 💰 Loan Status Prediction with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.0-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project implements an end-to-end machine learning solution for predicting loan approval status based on various applicant features such as income, credit history, loan amount, property area, and more. The project includes comprehensive data analysis, model training with Support Vector Machine (SVM) algorithm, and an interactive web application built with Streamlit for real-time loan approval predictions.

**Key Highlights:**
- 📊 Analyzed **614 loan application records** from comprehensive loan dataset
- 🤖 Built and trained a **Support Vector Machine (SVM)** classifier model
- 🎨 Deployed an interactive **Streamlit** web application
- 📦 Implemented production-ready ML pipeline with preprocessing
- 🎯 Achieved **81.02% training accuracy** and **79.17% testing accuracy**

## ✨ Features

### Machine Learning Pipeline
- **Advanced Preprocessing**: Utilizes `ColumnTransformer` and `Pipeline` for efficient data transformation
- **Feature Engineering**: 
  - One-Hot Encoding for categorical variables (Gender, Married, Education, Self_Employed, Property_Area, Dependents)
  - Standard Scaling for numerical features (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term)
  - Credit History processing (binary encoding)
- **SVM Model**: Trained Support Vector Machine classifier with linear kernel for accurate and reliable predictions
- **Model Persistence**: Serialized model using `pickle` for quick deployment and inference

### Web Application
- **Interactive UI**: User-friendly Streamlit interface for inputting loan applicant details
- **Real-time Predictions**: Instant loan approval estimation based on user inputs
- **Input Validation**: Smart input controls with appropriate ranges and options
- **Dynamic Features**: 
  - Support for multiple demographic categories
  - Multiple property area types (Urban, Semiurban, Rural)
  - Credit history tracking
  - Dependent count management
  - Education and employment status options

### Data Analysis
- **Exploratory Data Analysis (EDA)**: Comprehensive analysis in Jupyter notebooks
- **Data Visualization**: Plots and charts for understanding data distribution
- **Missing Value Handling**: Clean dataset preparation
- **Statistical Analysis**: Descriptive statistics and correlation analysis

## 📊 Dataset

**Source**: Loan_Status.csv

**Dataset Statistics:**
- **Total Records**: 614 loan applications
- **Features**: 13 columns (12 features + 1 target variable)

**Features Description:**
| Feature | Type | Description |
|---------|------|-------------|
| `Loan_ID` | String | Unique identifier for each loan application |
| `Gender` | Categorical | Applicant's gender (Male, Female) |
| `Married` | Categorical | Marital status (Yes, No) |
| `Dependents` | Categorical | Number of dependents (0, 1, 2, 3+) |
| `Education` | Categorical | Education level (Graduate, Not Graduate) |
| `Self_Employed` | Categorical | Self-employment status (Yes, No) |
| `ApplicantIncome` | Numerical | Applicant's income in currency units |
| `CoapplicantIncome` | Numerical | Co-applicant's income in currency units |
| `LoanAmount` | Numerical | Loan amount in thousands |
| `Loan_Amount_Term` | Numerical | Loan repayment term in days |
| `Credit_History` | Binary | Credit history meets guidelines (1: Yes, 0: No) |
| `Property_Area` | Categorical | Property location (Urban, Semiurban, Rural) |
| `Loan_Status` | Binary | Loan approval status - **Target Variable** (Y: Approved, N: Not Approved) |

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Jupyter Notebook**: For data analysis and model development
- **Streamlit 1.41.1**: Web application framework

### Machine Learning & Data Science
- **scikit-learn 1.6.0**: Machine learning algorithms and preprocessing
- **pandas 2.2.3**: Data manipulation and analysis
- **numpy 2.2.0**: Numerical computing
- **scipy 1.14.1**: Scientific computing

### Model Components
- **SVM (Support Vector Machine)**: Linear kernel classifier
- **ColumnTransformer**: Feature preprocessing
- **Pipeline**: ML workflow automation
- **OneHotEncoder**: Categorical feature encoding
- **StandardScaler**: Feature scaling

### Deployment & Utilities
- **pickle**: Model serialization
- **joblib 1.4.2**: Efficient model persistence

## 📁 Project Structure

```
loan-status-prediction/
│
├── 📓 Loan_Status_Prediction_Streamlit.ipynb  # Model development and pipeline notebook
├── 🐍 Loan_Status_Streamlit.py                # Streamlit web application
├── 📊 Loan_Status.csv                         # Dataset file
├── 🤖 loan_status_model.pkl                   # Trained model pipeline (serialized)
├── 📋 requirements.txt                        # Python dependencies
├── 📄 README.md                               # Project documentation
├── 📜 LICENSE                                 # Apache 2.0 License
└── 🚫 .gitignore                              # Git ignore rules
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository
```bash
git clone https://github.com/arpanpramanik2003/loan-status-prediction.git
cd loan-status-prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: The `requirements.txt` file includes all necessary packages:
- streamlit==1.41.1
- pandas==2.2.3
- numpy==2.2.0
- scikit-learn==1.6.0
- And other supporting libraries

## 💻 Usage

### Running the Streamlit Web Application

1. **Start the Streamlit server**:
```bash
streamlit run Loan_Status_Streamlit.py
```

2. **Access the application**:
   - Open your web browser
   - Navigate to `http://localhost:8501`

3. **Make Predictions**:
   - Select applicant's gender from dropdown
   - Choose marital status
   - Select education level
   - Choose self-employment status
   - Enter applicant income
   - Enter co-applicant income (if applicable)
   - Enter loan amount (in thousands)
   - Enter loan amount term (in days)
   - Select credit history status
   - Choose property area
   - Select number of dependents
   - Click "Predict" button
   - View the loan approval prediction result

### Working with Jupyter Notebooks

#### Model Training and Pipeline Development
```bash
jupyter notebook Loan_Status_Prediction_Streamlit.ipynb
```
This notebook covers:
- Data loading and exploration
- Exploratory Data Analysis (EDA)
- Data preprocessing and cleaning
- Feature engineering and encoding
- Model training with SVM (linear kernel)
- Model evaluation on training and test data
- Pipeline creation with ColumnTransformer
- Model serialization with pickle

## 🏗️ Model Architecture

### Preprocessing Pipeline

```python
ColumnTransformer:
  ├── OneHotEncoder (drop='first')
  │   └── Features: ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']
  └── StandardScaler
      └── Features: ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
```

### Model Configuration

- **Algorithm**: Support Vector Machine (SVM)
- **Kernel**: Linear
- **Target Variable**: Loan_Status (Approved/Not Approved)
- **Preprocessing**: ColumnTransformer with OneHotEncoder and StandardScaler

### Training Process

1. **Data Loading**: Import Loan_Status.csv dataset
2. **Data Cleaning**: 
   - Handle missing values
   - Convert categorical values (Loan_Status: 'Y' to 1, 'N' to 0)
   - Process Dependents column ('3+' to 4)
3. **Data Splitting**: Train-test split for model validation
4. **Pipeline Training**: 
   - Categorical encoding with OneHotEncoder
   - Numerical scaling with StandardScaler
   - Model training with SVM linear kernel
5. **Model Serialization**: Save pipeline as `loan_status_model.pkl`

### Model Workflow

```
User Input → Preprocessing Pipeline → SVM Classifier → Loan Approval Prediction
              (Encoding + Scaling)     (Linear Kernel)    (Approved/Not Approved)
```

## 📈 Results

The Support Vector Machine (SVM) model was evaluated using accuracy metrics:

- **Training Accuracy**: **81.02%** - Strong performance on training data
- **Testing Accuracy**: **79.17%** - Consistent generalization on unseen data
- **Model Reliability**: Linear SVM provides interpretable decision boundaries

**Key Achievements**:
- ✅ Successfully handles multiple categorical and numerical features
- ✅ Provides real-time predictions through web interface
- ✅ Production-ready ML pipeline with proper preprocessing
- ✅ Balanced accuracy between training and testing datasets
- ✅ Automated loan approval decision support system

## 🌐 Deployment

### Local Deployment
The application runs locally using Streamlit. Follow the [Usage](#usage) section to start the server.

### Cloud Deployment Options

#### Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click

#### Other Platforms
- **Heroku**: Deploy with Procfile configuration
- **AWS EC2**: Deploy on cloud server
- **Google Cloud Platform**: Use App Engine or Cloud Run
- **Azure**: Deploy as Web App

**Important Files for Deployment**:
- `requirements.txt`: Python dependencies
- `Loan_Status_Streamlit.py`: Main application file
- `loan_status_model.pkl`: Pre-trained model
- `Loan_Status.csv`: Dataset (if needed for retraining)

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. **Make your changes**
4. **Commit your changes**:
   ```bash
   git commit -m "Add some feature"
   ```
5. **Push to the branch**:
   ```bash
   git push origin feature/YourFeatureName
   ```
6. **Open a Pull Request**

### Areas for Contribution
- 🐛 Bug fixes and issue resolution
- ✨ New features (e.g., more ML models, ensemble methods, visualizations)
- 📝 Documentation improvements
- 🎨 UI/UX enhancements
- 🧪 Additional test cases
- 📊 More comprehensive data analysis
- 🔍 Feature importance analysis and model explainability

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

```
Copyright 2024 Arpan Pramanik

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

## 👤 Contact

**Arpan Pramanik**

- 🐱 GitHub: [@arpanpramanik2003](https://github.com/arpanpramanik2003)
- 📧 Email: [Contact via GitHub](https://github.com/arpanpramanik2003)
- 💼 Project Link: [https://github.com/arpanpramanik2003/loan-status-prediction](https://github.com/arpanpramanik2003/loan-status-prediction)

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Made with ❤️ by Arpan Pramanik**

</div>
