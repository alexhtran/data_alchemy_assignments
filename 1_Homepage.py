import streamlit as st

st.set_page_config(
    page_title="Loan Prediction Dashboard",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.header("📊 Loan Status Prediction")
st.subheader("📋 Overview")
st.markdown("This is a dashboard using a dataset obtained from [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e10/).\n\n"
            "Use the sidebar to view the exploratory data analysis (EDA) and/or models.\n\n"
            "This dataset will be used to answer the following problem:")

st.subheader("🔍 Problem Statement")
st.markdown("**Can we predict if a loan is likely to be paid back, and if so, what factors influence the likelihood of being paid back?**\n\n"
    "In other words, can we predict if a loan is going to be defaulted, and what factors influence if a loan is prone to defaulting")

st.subheader("✏️ About the Dataset")
st.write("The dataset gives us information about a person's demographics and description of their loan application.")

st.subheader("💻 Variables")
st.markdown("- **id:** id of each entry\n\n"
                "- **person_age:** age of person\n\n"
                "- **person_income:** income of person (in $)\n\n"
                "- **person_home_ownership:** home ownership status of person\n\n"
                "- **person_emp_length:** employment length of person\n\n"
                "- **loan_intent:** reason for applying for a loan\n\n"
                "- **loan_grade:** grade of loan\n\n"
                "- **loan_amnt:** amount of loan (in $)\n\n"
                "- **loan_int_rate:** interest rate of loan\n\n"
                "- **loan_percent_income:** how much of the loan as part of their income\n\n"
                "- **history_of_default:** whether the person has defaulted before\n\n"
                "- **cred_hist_length:** how long this person has had a credit history\n\n"
                "- :red[ **loan_status:** (*the variable we are predicting*) whether the loan is likely to default or be paid back]\n\n")
                # "- **z_emp_length:** standardized score of employee length\n\n"
                # "- **z_age:** standardized score of age\n\n"
                # "- **z_income:** standardized score of income\n\n")