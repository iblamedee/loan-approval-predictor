import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("loan_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
columns = pickle.load(open("columns.pkl","rb"))

st.title("Loan Approval Prediction")

income = st.number_input("Applicant Income")
loan_amount = st.number_input("Loan Amount")
credit_score = st.number_input("Credit Score")
dti = st.number_input("DTI Ratio")
savings = st.number_input("Savings")
age = st.number_input("Age")

if st.button("Predict"):

    data = {
        "Applicant_Income": income,
        "Loan_Amount": loan_amount,
        "Credit_Score": credit_score,
        "DTI_Ratio": dti,
        "Savings": savings,
        "Age": age
    }

    input_df = pd.DataFrame([data])

    # create engineered features
    input_df["Credit_Score_sq"] = input_df["Credit_Score"]**2
    input_df["DTI_Ratio_sq"] = input_df["DTI_Ratio"]**2
    input_df["Applicant_Income_log"] = np.log1p(input_df["Applicant_Income"])

    # align columns with training
    input_df = input_df.reindex(columns=columns, fill_value=0)
    # scale
    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0][1]


    st.write("Approval probability:", prob)

    if prob >= 0.70:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")