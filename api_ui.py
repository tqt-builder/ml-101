import streamlit as st
import requests

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")

st.title("🏥 Insurance Premium Predictor")
st.markdown("Enter your details below to get an estimated insurance cost.")

# 1. Create the input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.5, format="%.1f")
    
    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Smoker?", ["yes", "no"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    submit = st.form_submit_button("Calculate Estimated Charges")

# 2. Handle the prediction
if submit:
    # Prepare the data for our FastAPI backend
    payload = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }
    
    try:
        # Send POST request to your FastAPI server
        # Ensure your FastAPI server is running on http://127.0.0.1:8000
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            charges = result["estimated_charges"]
            
            st.success("### Calculation Successful!")
            st.metric(label="Estimated Insurance Charges", value=f"${charges:,.2f}")
            
            # Optional: Add some visual context
            if smoker == "yes":
                st.warning("Note: Smoking significantly increases your insurance premiums.")
        else:
            st.error(f"Error from API: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend server. Make sure FastAPI is running (uvicorn api.main:app).")
