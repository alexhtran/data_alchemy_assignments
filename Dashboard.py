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
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the App mode",
                            ["Overview", "Interactive EDA", "Interactive Models"])

# ----------------------
# Charts for EDA
# ----------------------

@st.cache_data
def get_data():
    # Dataset """before""" converting categorical to numerical 
    loan_dataset1 = pd.read_csv('loan_data1.csv', index_col=0)
    df1 = loan_dataset1
    return df1

def get_conv_data():
    # Dataset """after""" converting categorical to numerical
    loan_dataset2 = pd.read_csv('loan_data2.csv', index_col=0)
    df2 = loan_dataset2
    return df2

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

def get_box_plot(box_column):
    fig_box = px.box(df1, x=box_column, title="Box Plot by Category")
    st.plotly_chart(fig_box, use_container_width=True)

# ----------------------
# Overview Section
# ----------------------

if app_mode == "Overview":
    st.header("About the Dataset")

    # About

    st.write("This dataset was obtained from [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e10/).\n\n"
            "The dataset gives us information about a person's demographics and description of their loan application.")

    st.subheader("Variables")
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
                "- :red[ **loan_status:** (*the variable we are predicting*) whether the loan is likely to default or be paid back]\n\n"
                "- **z_emp_length:** standardized score of employee length\n\n"
                "- **z_age:** standardized score of age\n\n"
                "- **z_income:** standardized score of income\n\n")

    st.subheader("Problem Statement")
    st.markdown("**Can we predict if a loan is likely to be paid back, and if so, what factors influence the likelihood of being paid back?**\n\n"
    "In other words, can we predict if a loan is going to be defaulted, and what factors influence if a loan is prone to defaulting")

    st.subheader("Brief Overview of Data")
    df = get_data()
    st.write(df.head())

# ----------------------
# Interactive EDA Section
# ----------------------

if app_mode == "Interactive EDA":
    st.header("Exploratory Data Analysis")
    
    # Load dataset
    
    df1 = get_data()
    df2 = get_conv_data()
    
    st.subheader("Dataset Preview")
    st.write(df1.head())
    
    # Show dataset dimensions
    st.write("Dataset Dimensions:", df1.shape)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(df1.describe(include="all"))
    
    #get columns
    numeric_columns = df1.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df1.select_dtypes(include="object").columns.tolist()

    # scatter plot with Plotly
    st.subheader("Interactive Scatter Plot")
    x_axis = st.selectbox("Select X-axis", numeric_columns, index=0)
    y_axis = st.selectbox("Select Y-axis", numeric_columns, index=1)
    color = st.selectbox("Select color grouping", categorical_columns, index=0)
    get_scatter_plot(x_axis,y_axis,color)

    # Interactive histogram
    st.subheader("Interactive Histogram")
    hist_column = st.selectbox("Select column for histogram", numeric_columns, index=0)
    color_col = st.selectbox("Select hist grouping color", categorical_columns, index=0)
    get_hist(color_col,hist_column)

    # Interactive Box Plot
    st.subheader("Interactive Boxplot")
    box_column = st.selectbox("Select column for box plot", numeric_columns, index=0)
    get_box_plot(box_column)

    # Interactive Corr Matrix
    st.subheader("Interactive Correlation Matrix")
    st.markdown("This correlation matrix uses a dataframe that converts categorical variables into numerical variables.")
    fig = px.imshow(
    df2.select_dtypes(include=['number']).corr(),text_auto=True,aspect="auto",
    color_continuous_scale="RdBu_r",  # Red-blue color scale for positive/negative correlations
    origin='lower',title="Correlation Matrix",width = 600, height=710
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------
# Interactive Models Section
# ----------------------

def clean_data(df):
    return df.drop(['person_age', 'loan_grade', 'z_age', 'person_income', 'person_emp_length'], axis=1)

if app_mode == "Interactive Models":
    st.header("Interactive Models")

    # Logistic Regression
    # Load & clean data
    df1 = get_data() # before converting categorical to numerical
    df2 = get_conv_data() # after converting categorical to numerical
    df3 = clean_data(df2)  # using all of the numerical data

    numeric_columns = df3.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df3.select_dtypes(include="object").columns.tolist()

    # Define features and target
    target_col = ['loan_status']
    features = [col for col in df3.columns if col not in target_col]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numeric_columns)
        ],
        sparse_threshold=0
    )

    X_processed = preprocessor.fit_transform(df3[features])
    df_transformed = pd.DataFrame(X_processed, columns=numeric_columns)

    # Split features and targets
    X_processed = df_transformed
    y_class = df2['loan_status'].astype(int) # Ensure binary target is integer (0/1)

    # Split into training and test sets (70/30 split)
    # We are using the same training set but distinct targets for the classification and regression models
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_class, test_size=0.3, random_state=4, stratify=y_class) # Prevents class imbalance

    # Logistic Regression for Classification
    log_reg = LogisticRegression(max_iter=1000)

    log_reg.fit(X_train, y_train)
    y_pred_log1 = log_reg.predict(X_test)

    acc_log1 = accuracy_score(y_test, y_pred_log1)
    print("Logistic Regression Accuracy:", acc_log1)

    cm1 =  confusion_matrix(y_test, y_pred_log1)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    precision1 = precision_score(y_test, y_pred_log1)
    recall1 = recall_score(y_test, y_pred_log1)
    f11 = f1_score(y_test, y_pred_log1)

    print("Precision:", precision1)
    print("Recall:", recall1)
    print("F1-Score:", f11)