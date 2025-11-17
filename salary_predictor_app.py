import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
from io import StringIO

# --- Configuration ---
MODEL_PATH = Path(__file__).parent / "salary_predictor_model_real_v4.joblib"
DATA_FILE_NAME = "Salary_Data.csv"
DATA_PATH = Path(__file__).parent / DATA_FILE_NAME
RANDOM_STATE = 42


# ==============================================================================
# 1. DATA LOADING AND MODEL TRAINING
# ==============================================================================

@st.cache_data
def read_csv_content(data_path):
    """Loads CSV content from a file path."""
    if not os.path.exists(data_path):
        st.error(f"Error: Required file '{DATA_FILE_NAME}' not found.")
        st.warning(f"Please ensure '{DATA_FILE_NAME}' is placed next to this script.")
        return None

    with open(data_path, 'r', encoding='utf-8') as f:
        return f.read()


@st.cache_data
def load_and_prepare_data(csv_content_string):
    """Loads and cleans the data from the CSV string."""
    try:
        # Read the CSV content string into a DataFrame
        df = pd.read_csv(StringIO(csv_content_string))

        # Define features and target
        FEATURES = ['Age', 'Years of Experience']
        TARGET = 'Salary'

        if not all(col in df.columns for col in FEATURES + [TARGET]):
            st.error(f"Data error: Required columns missing. Need {FEATURES} and {TARGET}.")
            return None

        # Ensure correct data types and drop NaN rows
        df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors='coerce')
        df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')

        df = df.dropna(subset=FEATURES + [TARGET])

        if df.empty or len(df) < 5:
            st.error("Training data must contain at least 5 complete, valid rows.")
            return None

        return df

    except Exception as e:
        st.error(f"Error processing CSV data: {e}")
        return None


@st.cache_resource
def train_and_save_model(df):
    """Trains the Ridge Regressor model and saves the pipeline."""

    # Features used: Age and Years of Experience
    X = df[['Age', 'Years of Experience']]
    y = df['Salary']

    # Define the pipeline: Scaling + Ridge Regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0, random_state=RANDOM_STATE))
    ])

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    pipeline.fit(X_train, y_train)

    # Save model (optional, for persistence across sessions)
    joblib.dump(pipeline, MODEL_PATH)

    # R^2 score for performance indication
    r2_score = pipeline.score(X_test, y_test)

    return pipeline, r2_score


def predict_salary(model, input_data, df):
    """Makes a prediction for a single employee."""
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)

    # Clip the prediction to the realistic range seen in the training data
    min_salary = df['Salary'].min()
    max_salary = df['Salary'].max()

    return np.clip(prediction[0], min_salary, max_salary)


# ==============================================================================
# 2. STREAMLIT APP LAYOUT
# ==============================================================================

def main():
    st.set_page_config(page_title="Employee Salary Predictor", layout="wide")
    st.title("💼 Employee Salary Predictor (Trained on External CSV)")
    st.markdown(
        f"Model automatically trained on the data from **`{DATA_FILE_NAME}`** using Age and Years of Experience.")

    # --- Load Data and Train Model Automatically ---
    csv_content = read_csv_content(DATA_PATH)

    if csv_content is None:
        st.stop()

    data_df = load_and_prepare_data(csv_content)

    if data_df is None:
        st.stop()

    # Train model (uses st.cache_resource, so it only trains once)
    model, r2_score = train_and_save_model(data_df)

    st.sidebar.header("Model Status")
    st.sidebar.success(f"Trained on {len(data_df)} samples.")
    st.sidebar.info(f"Model R² Score: {r2_score:.2f}")

    st.markdown("---")

    col_input, col_results = st.columns([1, 1])

    # --- INPUT WIDGETS ---
    with col_input:
        st.header("1. Input Employee Profile")

        # Determine realistic input ranges from the data
        min_age = int(data_df['Age'].min())
        max_age = int(data_df['Age'].max())
        min_exp = float(data_df['Years of Experience'].min())
        max_exp = float(data_df['Years of Experience'].max())

        # Sliders for user input
        age = st.slider(
            "Age",
            min_value=min_age, max_value=max_age, value=int(data_df['Age'].mean()), step=1,
            help=f"Range based on your data: {min_age} to {max_age}."
        )

        years_exp = st.slider(
            "Years of Experience",
            min_value=min_exp, max_value=max_exp, value=float(data_df['Years of Experience'].mean().round(1)), step=0.5,
            help=f"Range based on your data: {min_exp} to {max_exp} years."
        )

    # --- PREDICTION AND RESULTS ---
    with col_results:
        st.header("2. Predicted Salary")

        input_data = {
            'Age': age,
            'Years of Experience': years_exp,
        }

        # Run prediction whenever inputs change
        predicted_salary = predict_salary(model, input_data, data_df)

        # Format the output nicely
        formatted_salary = f"${predicted_salary:,.0f}"

        st.metric(
            label="Predicted Annual Salary (USD)",
            value=formatted_salary,
            delta=f"R² Score: {r2_score:.2f}",
            delta_color="off"
        )

        # Dynamic feedback based on max salary in the dataset
        max_df_salary = data_df['Salary'].max()
        if predicted_salary < 60000:
            st.warning("Prediction suggests a starting or entry-level salary.")
        elif predicted_salary > max_df_salary * 0.7:
            st.success("Prediction suggests a senior or executive-level salary.")
        else:
            st.info("Predicted salary is competitive for the mid-career level.")

    st.markdown("---")
    st.subheader("3. Historical Training Data Preview")
    st.caption(f"Showing the first 10 rows of {len(data_df)} records.")
    st.dataframe(data_df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()