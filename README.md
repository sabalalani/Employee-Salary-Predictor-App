💼 Employee Salary Predictor

This is a Streamlit web application designed to predict an employee's Salary based on their Age and Years of Experience.

The model is trained automatically upon startup using the provided Salary_Data.csv file, eliminating the need for manual training steps.

Data and Model

Data Source: Salary_Data.csv (Must be present in the project root directory).

Features Used: Age and Years of Experience.

Model: Scikit-learn Ridge Regression (A linear model suitable for estimating salaries).

Performance: The model's $R^2$ score (accuracy) is displayed in the sidebar after training.

Setup and Installation

1. Prerequisites

Ensure you have Python (3.7+) and the Salary_Data.csv file in the same directory as the Python script.

2. Install Dependencies

Install all necessary libraries using the provided requirements.txt file:

pip install -r requirements.txt


3. Run the Application

Execute the Streamlit script from your terminal:

streamlit run salary_predictor_app.py
