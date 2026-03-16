# 💳 Credit Risk Prediction using Machine Learning

## 📌 Project Overview

Credit risk analysis is an important task for financial institutions to determine whether a borrower is likely to **repay a loan or default**.

This project builds a **machine learning model that predicts the credit risk of loan applicants** based on financial and demographic attributes. The objective is to assist lenders in making **data-driven decisions** and reduce the probability of loan defaults.

---

## 📂 Project Structure

```
Credit-risk-model
│
├── dataset.csv          # Dataset used for training the model
├── model.pkl            # Trained machine learning model
├── app.py               # Flask web application
├── train_model.py       # Script for training the ML model
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Flask
* Joblib

These tools are commonly used for **data preprocessing, model training, and deployment in machine learning projects**.

---

## 📊 Machine Learning Workflow

The project follows a typical machine learning pipeline:

1. Data collection
2. Data preprocessing
3. Feature selection
4. Train-test split
5. Model training
6. Model evaluation
7. Model serialization using Joblib
8. Deployment through a Flask web application

---

## 📈 Model Objective

The model predicts whether a borrower is:

* **Low Risk (0)** – likely to repay the loan
* **High Risk (1)** – likely to default

Credit risk models help financial institutions **reduce financial losses and improve loan approval decisions**.

---

## 🚀 How to Run the Project

### Clone the repository

```bash
git clone https://github.com/RakeshChandra2127/Credit-risk-model.git
```

### Navigate to the project directory

```bash
cd Credit-risk-model
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

Then open the following URL in your browser:

```
http://127.0.0.1:5000
```

---

## 📊 Future Improvements

Possible improvements for this project:

* Hyperparameter tuning
* Testing additional models (Random Forest, XGBoost, Gradient Boosting)
* Adding model explainability tools (SHAP / LIME)
* Deploying the model on cloud platforms

---

## 👨‍💻 Author

**Rakesh Chandra Behera**

GitHub:
https://github.com/RakeshChandra2127

---

⭐ If you found this project useful, consider giving it a **star** on GitHub.
```
