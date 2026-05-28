import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Medical Insurance Cost Prediction", layout="wide")

# Title
st.title("🏥 Medical Insurance Cost Prediction")
st.markdown("Predict medical insurance costs in **Indian Rupees (₹)** based on personal health information")

# USD to INR conversion rate
USD_TO_INR = 83.0

@st.cache_resource
def load_and_train_model():
    """Load data, train XGBoost model, and return necessary components"""
    try:
        df = pd.read_excel('Medical_Insurance_cost _prediction.xlsm')
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return None
    
    # Create a copy for processing
    df_processed = df.copy()
    
    # Drop duplicates
    df_processed.drop_duplicates(inplace=True)
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    df_processed['sex'] = le_sex.fit_transform(df_processed['sex'])
    
    le_smoker = LabelEncoder()
    df_processed['smoker'] = le_smoker.fit_transform(df_processed['smoker'])
    
    # One-Hot Encoding for region
    df_processed = pd.get_dummies(df_processed, columns=['region'], drop_first=True)
    
    # Prepare features and target
    X = df_processed.drop('charges', axis=1)
    y = df_processed['charges']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Calculate model performance
    y_pred = model.predict(X_test)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return {
        'model': model,
        'le_sex': le_sex,
        'le_smoker': le_smoker,
        'feature_names': list(X.columns),
        'train_score': train_score,
        'test_score': test_score,
        'mae': mae,
        'rmse': rmse,
        'X_test': X_test,
        'y_test': y_test
    }

# Main app layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Enter Your Information")
    
    age = st.slider("Age", 18, 100, 35)
    bmi = st.slider("BMI (Body Mass Index)", 10.0, 55.0, 25.0, step=0.1)
    children = st.slider("Number of Children", 0, 5, 0)
    sex = st.selectbox("Sex", ["Male", "Female"])
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

with col2:
    st.subheader("📊 Prediction Result")
    
    # Get model
    model_data = load_and_train_model()
    
    if model_data is not None:
        model = model_data['model']
        le_sex = model_data['le_sex']
        le_smoker = model_data['le_smoker']
        feature_names = model_data['feature_names']
        
        # Make prediction
        if st.button("🔮 Predict Cost", key="predict"):
            try:
                # Encode categorical variables
                sex_encoded = le_sex.transform([sex.lower()])[0]
                smoker_encoded = le_smoker.transform(['yes' if smoker == 'Yes' else 'no'])[0]
                
                # Create region one-hot encoding
                region_lower = region.lower()
                region_encoded = {
                    'region_northeast': 1 if region_lower == 'northeast' else 0,
                    'region_northwest': 1 if region_lower == 'northwest' else 0,
                    'region_southeast': 1 if region_lower == 'southeast' else 0,
                    'region_southwest': 1 if region_lower == 'southwest' else 0,
                }
                
                # Create input dataframe
                input_data = pd.DataFrame({
                    'age': [age],
                    'bmi': [bmi],
                    'children': [children],
                    'sex': [sex_encoded],
                    'region_northeast': [region_encoded['region_northeast']],
                    'region_northwest': [region_encoded['region_northwest']],
                    'region_southeast': [region_encoded['region_southeast']],
                    'region_southwest': [region_encoded['region_southwest']],
                    'smoker': [smoker_encoded]
                })
                
                # Ensure columns match training data
                input_data = input_data[feature_names]
                
                # Make prediction in USD
                prediction_usd = model.predict(input_data)[0]
                
                # Convert to INR
                prediction_inr = prediction_usd * USD_TO_INR
                
                # Display result
                st.success(f"### Predicted Insurance Cost")
                st.metric("Cost in Indian Rupees (₹)", f"₹ {prediction_inr:,.2f}")
                st.metric("Cost in US Dollars ($)", f"$ {prediction_usd:,.2f}")
                
                st.info(f"""
                **Input Summary:**
                - Age: {age} years
                - BMI: {bmi}
                - Children: {children}
                - Sex: {sex}
                - Smoker: {smoker}
                - Region: {region}
                """)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                import traceback
                st.write(traceback.format_exc())

# Additional Information
st.markdown("---")
st.subheader("ℹ️ About This Model")
st.info("""
This model uses **XGBoost Regressor** trained on medical insurance data to predict costs based on:
- **Age**: Your current age
- **BMI**: Body Mass Index
- **Children**: Number of dependents
- **Sex**: Gender
- **Smoker Status**: Whether you smoke or not
- **Region**: Geographic location (Northeast, Northwest, Southeast, Southwest)

**Output**: Medical insurance charges in **Indian Rupees (₹)**
""")

# Model Performance
if st.checkbox("📈 Show Model Training Information"):
    if model_data is not None:
        st.subheader("Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Train R² Score", f"{model_data['train_score']:.4f}")
        col2.metric("Test R² Score", f"{model_data['test_score']:.4f}")
        col3.metric("MAE (USD)", f"${model_data['mae']:,.2f}")
        col4.metric("RMSE (USD)", f"${model_data['rmse']:,.2f}")
        
        st.success("✅ Model successfully trained using XGBoost!")
