# Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import datasets

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split,cross_val_score

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

# Evaluation metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# For combining pipelines after encoding
from sklearn.compose import make_column_selector as selector

sns.set(style="whitegrid")

st.set_page_config(
    page_title="Loan Prediction Dashboard",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🖥️ Exploratory Data Analysis")
st.write("This page is dedicated to exploring the data with descriptive statistics and visualizations.")

# ----------------------
# Datasets
# ----------------------

def load_dataset(option):
    if option == "Cleaned Dataset":
        return pd.read_csv('loan_data_cleaned.csv')
    elif option == "Log Transformed Dataset":
        return pd.read_csv('loan_data_log.csv')
    else:
        return pd.DataFrame()

# ----------------------
# Functions for Data
# ----------------------

@st.cache_data
def get_scatter_plot(x_axis,y_axis,color):
    fig = px.scatter(df, x=x_axis, y=y_axis, color=color,
                            title=f"{x_axis} vs {y_axis}",
                            template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def get_hist(color_col,hist_column):
    fig_hist = px.histogram(df, x=hist_column, color=color_col, marginal="box",
                            title=f"Distribution of {hist_column}",
                            template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)

@st.cache_data
def get_box_plot(box_column):
    fig_box = px.box(df, x=box_column, title="Box Plot by Category")
    st.plotly_chart(fig_box, use_container_width=True)

# ----------------------
# Interactive EDA Section
# ----------------------

with st.expander("Why are there two datasets?"):
    st.markdown("Two datasets are included to showcase my thought process for EDA!")

cleaned = pd.read_csv('loan_data_cleaned.csv')
log = pd.read_csv('loan_data_log.csv')

clean_col, log_col = st.columns(2)
with clean_col:
    st.subheader("Cleaned Dataset")

    # Dataset Preview
    st.write(cleaned.head())

    # Dimensions
    st.write("Dataset Dimensions:", cleaned.shape)

    # Summary Statistics
    st.write(cleaned.describe())

    # Interpretation
    with st.expander("What does the cleaned data tell us?"):
        st.markdown("At a glance, there are some unordinary outliers in variables person_age and person_emp_length with values of 123.\n\n"
                    "Overall, this gives us a general idea of the demographics for people applying for a loan.\n\n"
                    "- More than half of the people applying for loans rent their houses.\n\n"
                    "- Most people also apply for a loan due to education.\n\n"
                    "- Judging from how small the age range is (at a mean of about 26-27 years old), this could likely be loans for higher education/to pay tuition.\n\n"
                    "- Most people applying for loans aren't employeed at their job for long with a mean of 4 years.\n\n"
                    "- The interest rates for loans have a mean of about 10%.\n\n"
                    "- The percentage of the loan amount as part of a person's income is fairly low at a mean of 14-16%, with an average loan amount of $9000.\n\n"
                    "- cred_hist_length and person_emp_length are very similar in numbers, so we can look into these variables to see if they measure the same thing.")
            
with log_col:
    st.subheader("Log Transformed Dataset")

    # Dataset Preview
    st.write(log.head())

    # Dimensions
    st.write("Dataset Dimensions", log.shape)

    # Summary Statistics
    st.write(log.describe())

    # Interpretation
    with st.expander("What does the log transformed dataset tell us?"):
        st.markdown("The data is significantly smaller when the data is transformed to a log.\n\n"
                    "This is because when the all the numerical data was transformed using logarithm functions, there were a lot of NaN values. For our EDA and models to work, NaN values need to be dropped.\n\n"
                    "The data was converted using logarithm functions because when looking at the distribution of the cleaned dataset using histograms, it was *not* normally distributed. Our data needs to be normally distributed for logistic regression (to determine whether a loan will default or not).\n\n"
                    "Check out the interactive visualizations to view the data!")

st.subheader("Interactive Visualizations")
dataset_option = st.selectbox("Choose a dataset to explore:", ["Cleaned Dataset", "Log Transformed Dataset"])
df = load_dataset(dataset_option)

# Get features
cat_features = df.select_dtypes(include=['object']).columns
num_features = df.select_dtypes(include=np.number).columns.tolist()

# Mapping for variables
column_map = {
    "person_income": "log_person_income",
    "person_age": "log_age",
    "person_emp_length": "log_emp",
    "loan_amnt": "log_loan_amnt",
    "loan_int_rate": "log_loan_int_rate",
    "loan_percent_income": "log_loan_percent_income",
    "cred_hist_length": "log_cred_hist_length"
}

# Interactive Histogram
st.subheader("Interactive Histogram")
hist_column = st.selectbox("Select column for histogram (numerical variables)", num_features, index=0)
color_col = st.selectbox("Select hist grouping color (categorical variables)", cat_features, index=0)
get_hist(color_col,hist_column)

clean_hist, log_hist = st.columns(2)
with clean_hist:
    with st.expander("What does the cleaned data tell us?"):
        st.markdown("The cleaned data isn't normally distributed. For the type of variable we're predicting (a classification variable), we need to make sure the data *is* normally distributed.\n\n"
                    "Otherwise, our model will have biases when finding a classification for given inputs.")

with log_hist:
    with st.expander("What does the log transformed data tell us?"):
        st.markdown("A logarithm function was applied to all the predictor variables to allow our values to be symmetric - essentially following a normal distribution. However, the histograms were still not normally distributed. Therefore, **Z-scores** were also used to identify and drop outliers with z-scores greater than 3.00."
                    "We also used a standard scale to make the variables compatible with logistic regression.")

# Scatter Plot with Plotly
st.subheader("Interactive Scatter Plot")
x_axis = st.selectbox("Select X-axis (numerical variables)", num_features, index=0)
y_axis = st.selectbox("Select Y-axis (numerical variables)", num_features, index=1)
color = st.selectbox("Select color grouping (categorical variables)", cat_features, index=0)
get_scatter_plot(x_axis,y_axis,color)

clean_scatt, log_scatt = st.columns(2)
with clean_scatt:
    with st.expander("What does the cleaned data tell us?"):
        st.markdown("This generally shows us the relationship between the chosen X and Y variables. A categorical variable for grouping visually shows more insights.")

with log_scatt:
    with st.expander("What does the log transformed data tell us?"):
        st.markdown("This generally shows us the relationship between the chosen X and Y variables.\n\n"
                    "For certain numerical variables like loan_status, it shows vertical lines because loan_status is measure on two numbers - likely to be paid back (approximately 0.38) and likely to default (approximately 2.60).")

# Interactive Box Plot
st.subheader("Interactive Boxplot")
box_column = st.selectbox("Select column for box plot (numerical variables)", num_features, index=0)
get_box_plot(box_column)

clean_box, log_box = st.columns(2)
with clean_box:
    with st.expander("What does the cleaned data tell us?"):
        st.markdown("This generally shows us the quartiles and outliers for a variable, easier than a histogram.")

with log_box:
    with st.expander("What does the log transformed data tell us?"):
        st.markdown("This generally shows us the quartiles and outliers for a variable, easier than a histogram.\n\n"
                    "There are not as many outliers in the log transformed dataset due to them being removed during the EDA process.")

# Interactive Corr Matrix
st.subheader("Interactive Correlation Matrix")
fig = px.imshow(
df.select_dtypes(include=['number']).corr(),text_auto=True,aspect="auto",
color_continuous_scale="RdBu_r",  # Red-blue color scale for positive/negative correlations
origin='lower',title="Correlation Matrix",width = 700, height=700)
st.plotly_chart(fig, use_container_width=True)

clean_corr, log_corr = st.columns(2)
with clean_corr:
    with st.expander("What does the cleaned data tell us?"):
        st.markdown("Any variables with high correlations (either dark red or blue) indicates multicollinearity.\n\n"
                    "- There's a high correlation between person_age and cred_hist_length, so one of these variables will be dropped in our dataset.\n\n"
                    "- There's a high correlation between loan_amnt and loan_percent_income, most likely because loan_percent_income represents the loan amount as part of the applicant's income.\n\n"
                    "- For our target variable, loan_status, the variables with the highest correlations are loan_percent_income, and loan_int_rate at approximately 0.30-0.40.")
with log_corr:
    with st.expander("What does the log transformed data tell us?"):
        st.markdown("A lot the variables were removed as they weren't important to our target variable.\n\n"
                    "- The variables that influence loan_status the most are log_loan_int_rate and log_loan_percent_income at a positive correlation. This esseentially means that the higher the interest rate or loan as part of the applicant's income, " \
                    "the more likely the loan *will default* (be a value of 1). This could be because a high interest rate for a loan is more likely to be a high-risk loan to default.\n\n"
                    "- With log_income and log_emp, these are negative correlations. The higher the income or employment length, the likelihood that loan_status will be 0.")