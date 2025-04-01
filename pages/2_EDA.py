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
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🖥️ Exploratory Data Analysis")
st.write("This page is dedicated to exploring the data with descriptive statistics.")

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

dataset_option = st.selectbox("Choose a dataset:", ["Cleaned Dataset", "Log Transformed Dataset"])
df = load_dataset(dataset_option)

# Dataset Preview
st.subheader("Dataset Preview")
st.write(df.head())

# Show dataset dimensions
st.write("Dataset Dimensions:", df.shape)

# Summary statistics
st.subheader("Summary Statistics")
st.write(df.describe(include="all"))

summ_button = st.button("What does this tell us?", key="summary")
if summ_button:
    st.markdown("At a glance, we have some unordinary outliers in variables person_age and person_emp_length with values of 123.\n\n"
                "Overall, this gives us a general idea of the demographics for people applying for a loan.\n\n"
                "- More than half of the people applying for loans rent their houses.\n\n"
                "- Most people also apply for a loan due to education.\n\n"
                "- Judging from how small the age range is (at a mean of about 26-27 years old), this could likely be loans for higher education/to pay tuition.\n\n"
                "- Most people applying for loans aren't employeed at their job for long with a mean of 4 years.\n\n"
                "- The interest rates for loans have a mean of about 10%.\n\n"
                "- The percentage of the loan amount as part of a person's income is fairly low at a mean of 14-16%, with an average loan amount of $9000.\n\n"
                "- cred_hist_length and person_emp_length are very similar in numbers, so we can look into these variables to see if they measure the same thing.")

# Get features
cat_features = df.select_dtypes(include=['object']).columns
num_features = df.select_dtypes(include=np.number).columns.tolist()

# Interactive Histogram
st.subheader("Interactive Histogram")
hist_column = st.selectbox("Select column for histogram (numerical variables)", num_features, index=0)
color_col = st.selectbox("Select hist grouping color (categorical variables)", cat_features, index=0)
get_hist(color_col,hist_column)

hist_button = st.button("What does this tell us?", key="hist")
if hist_button:
    st.markdown("The cleaned data isn't normally distributed. For the type of variable we're predicting (a classification variable), we need to make sure the data *is* normally distributed.\n\n"
                "Otherwise, our model will have biases when finding a classification for given inputs.")

log_button = st.button("What's different with the log data?", key="log")
if log_button:
    st.markdown("A logarithm function was applied to all the predictor variables to allow our values to be symmetric - essentially following a normal distribution. However, the histograms were still not normally distributed. Therefore, Z-scores were also used to identify and drop outliers with z-scores greater than 3.00."
    "We also used a standard scale to make the variables compatible with logistic regression.")

# Scatter Plot with Plotly
st.subheader("Interactive Scatter Plot")
x_axis = st.selectbox("Select X-axis (numerical variables)", num_features, index=0)
y_axis = st.selectbox("Select Y-axis (numerical variables)", num_features, index=1)
color = st.selectbox("Select color grouping (categorical variables)", cat_features, index=0)
get_scatter_plot(x_axis,y_axis,color)

scattpl_button = st.button("What does this tell us?", key="scatter")
if scattpl_button:
    st.markdown("This generally shows us the relationship between the chosen X and Y variables. A categorical variable for grouping visually shows more insights.")

# Interactive Box Plot
st.subheader("Interactive Boxplot")
box_column = st.selectbox("Select column for box plot (numerical variables)", num_features, index=0)
get_box_plot(box_column)

box_button = st.button("What does this tell us?", key="box")
if box_button:
    st.markdown("This generally shows us the quartiles and outliers for a variable.")

# Interactive Corr Matrix
st.subheader("Interactive Correlation Matrix")
fig = px.imshow(
df.select_dtypes(include=['number']).corr(),text_auto=True,aspect="auto",
color_continuous_scale="RdBu_r",  # Red-blue color scale for positive/negative correlations
origin='lower',title="Correlation Matrix",width = 700, height=700)
st.plotly_chart(fig, use_container_width=True)

corr_button = st.button("What does this tell us?", "corr")
if corr_button:
    st.markdown("This correlation matrix uses a dataframe that converts categorical variables into numerical variables."
                "Any variables with high correlations (either dark red or blue) indicates multicollinearity.")