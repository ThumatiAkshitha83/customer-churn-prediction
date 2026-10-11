# Customer Churn Prediction System

A machine learning project that predicts whether a bank customer is likely to leave the bank, helping businesses understand customer churn and improve customer retention.

## Project Overview

Customer churn is a major challenge for businesses because losing customers can affect revenue and long-term growth. This project uses machine learning to identify customers who are more likely to leave based on their account and demographic information.

The project includes data preprocessing, exploratory data analysis, model training, evaluation, and a web application for making predictions.

## Objectives

- Predict whether a customer is likely to churn.
- Identify important factors associated with customer churn.
- Evaluate model performance using classification metrics.
- Provide an interactive interface for generating predictions.

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Web Application:** Streamlit
- **Model Persistence:** Pickle

## Project Workflow

1. **Data Collection:** Load the customer dataset.
2. **Data Preprocessing:** Inspect data types and missing values, remove irrelevant columns, and encode categorical variables.
3. **Exploratory Data Analysis:** Examine customer characteristics and patterns associated with churn.
4. **Model Training:** Train a Random Forest classification model.
5. **Model Evaluation:** Evaluate performance using accuracy, precision, recall, and F1-score.
6. **Prediction:** Use the saved model and preprocessing pipeline to generate predictions through the web application.

## Model Performance

The project achieved the following results during model evaluation:

| Metric | Result |
|---|---:|
| Baseline model accuracy | 85.2% |
| Accuracy with class weighting | 86.0% |
| Churn-class precision | 72% |
| Churn-class recall | 44% |
| Churn-class F1-score | 0.55 |

Class weighting was explored to give greater importance to customers who churn. Although the accuracy improved, the churn-class recall remained 44%, indicating that further improvement is needed to identify more customers who actually leave.

*Note: These are the recorded evaluation results from this project. Performance may vary with changes to the data, preprocessing, or model configuration.*

## Key Features

- Predict customer churn using machine learning.
- Process customer information before making predictions.
- Explore customer data to understand churn-related patterns.
- View predictions through an interactive web interface.
- Identify important features associated with model predictions.

## Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── train_model.py
├── churn_model.pkl
├── churn_pipeline.pkl
├── Churn_Modelling.csv
├── Customer Churning Prediction.ipynb
└── .gitignore
```

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/ThumatiAkshitha83/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

Install the Python packages used by the project:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

## Future Improvements

- Improve recall for customers who churn.
- Compare additional classification algorithms.
- Perform hyperparameter tuning and cross-validation.
- Add more interactive visualizations and customer retention insights.
- Deploy the application for public access.

## Author

**Akshitha Thumati**

B.Tech in Computer Science and Engineering (Data Science)  
Presidency University, Bengaluru

- GitHub: [ThumatiAkshitha83](https://github.com/ThumatiAkshitha83)
- LinkedIn: [Akshitha Thumati](https://www.linkedin.com/in/akshitha-thumati/)

---

*Built as a machine learning project to explore customer churn prediction and data-driven customer retention.*
