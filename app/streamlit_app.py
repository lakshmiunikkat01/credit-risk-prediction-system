import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/credit_model.pkl")

st.title(" Credit Risk Prediction System")

st.subheader("Enter Applicant Details")

Gender = st.selectbox("Gender", ["Female", "Male"])
Married = st.selectbox("Married", ["No", "Yes"])
Dependents = st.slider("Dependents", 0, 3, 0)
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["No", "Yes"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Term", min_value=0)
Credit_History = st.selectbox("Credit History", ["Bad", "Good"])
Property_Area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

if st.button("Predict Risk"):

    Gender_val = 1 if Gender == "Male" else 0
    Married_val = 1 if Married == "Yes" else 0
    Education_val = 1 if Education == "Not Graduate" else 0
    Self_Employed_val = 1 if Self_Employed == "Yes" else 0
    Credit_History_val = 1 if Credit_History == "Good" else 0

    area_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    Property_Area_val = area_map[Property_Area]

    input_data = pd.DataFrame([[0,
                                Gender_val,
                                Married_val,
                                Dependents,
                                Education_val,
                                Self_Employed_val,
                                ApplicantIncome,
                                CoapplicantIncome,
                                LoanAmount,
                                Loan_Amount_Term,
                                Credit_History_val,
                                Property_Area_val]],
                              columns=[
                                  "Loan_ID",
                                  "Gender",
                                  "Married",
                                  "Dependents",
                                  "Education",
                                  "Self_Employed",
                                  "ApplicantIncome",
                                  "CoapplicantIncome",
                                  "LoanAmount",
                                  "Loan_Amount_Term",
                                  "Credit_History",
                                  "Property_Area"
                              ])

    prediction_prob = model.predict_proba(input_data)[0][1]

    if prediction_prob >= 0.7:
        risk = "High Risk"
        st.error("High Credit Risk")
    elif prediction_prob >= 0.4:
        risk = "Medium Risk"
        st.warning("Medium Credit Risk")
    else:
        risk = "Low Risk"
        st.success("Low Credit Risk")

    st.write("Risk Score:", round(prediction_prob, 3))
    st.progress(float(prediction_prob))