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

@st.cache_data
def loan_data_cleaned():
    # Cleaned dataset before log
    loan_cleaned = pd.read_csv('loan_data_cleaned.csv', index_col=0)
    df = loan_cleaned
    return df

@st.cache_data
def loan_data_log():
    # Cleaned dataset using log
    log_data = pd.read_csv('loan_data_log.csv', index_col=0)
    df = log_data
    return df

# ----------------------
# Functions for Log Data
# ----------------------

@st.cache_data
def get_scatter_plot_clean(x_axis,y_axis,color):
    fig = px.scatter(clean_data, x=x_axis, y=y_axis, color=color,
                            title=f"{x_axis} vs {y_axis}",
                            template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def get_hist_log_clean(color_col,hist_column):
    fig_hist = px.histogram(clean_data, x=hist_column, color=color_col, marginal="box",
                            title=f"Distribution of {hist_column}",
                            template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)

@st.cache_data
def get_box_plot_log_clean(box_column):
    fig_box = px.box(clean_data, x=box_column, title="Box Plot by Category")
    st.plotly_chart(fig_box, use_container_width=True)

# ----------------------
# Functions for Log Data
# ----------------------

@st.cache_data
def get_scatter_plot_log(x_axis,y_axis,color):
    fig = px.scatter(log_data, x=x_axis, y=y_axis, color=color,
                            title=f"{x_axis} vs {y_axis}",
                            template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def get_hist_log(color_col,hist_column):
    fig_hist = px.histogram(log_data, x=hist_column, color=color_col, marginal="box",
                            title=f"Distribution of {hist_column}",
                            template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)

@st.cache_data
def get_box_plot_log(box_column):
    fig_box = px.box(log_data, x=box_column, title="Box Plot by Category")
    st.plotly_chart(fig_box, use_container_width=True)

# ----------------------
# Interactive EDA Section
# ----------------------

# Load dataset
#outliers_dropped = loan_data_outliers_dropped()
#names_changes = loan_data_names_change()
#all_nums = loan_data_all_num()

clean_data = loan_data_cleaned()
log_data = loan_data_log()

# DATASET PREVIEW
st.subheader("Dataset Preview")
st.write(clean_data.head())

# Show dataset dimensions
st.write("Dataset Dimensions:", clean_data.shape)

# Summary statistics
st.subheader("Summary Statistics")
st.write(clean_data.describe(include="all"))

summ_button = st.button("What does this tell us?", key="summary")
if summ_button:
    st.markdown("At a glance, we have some unordinary outliers in variables person_age and person_emp_length with values of 123.\n\n")

clean_data = clean_data.drop(['id'], axis = 1)

# Get features
cd_cat_features = clean_data.select_dtypes(include=['object']).columns
cd_num_features = clean_data.select_dtypes(include=np.number).columns.tolist()

# Interactive Histogram
st.subheader("Interactive Histogram")
hist_column = st.selectbox("Select column for histogram (numerical variables)", cd_num_features, index=0)
color_col = st.selectbox("Select hist grouping color (categorical variables)", cd_cat_features, index=0)
get_hist_log_clean(color_col,hist_column)

hist_button = st.button("What does this tell us?", key="hist")
if hist_button:
    st.markdown("Hello hello")

st.subheader("Transforming Our Data")
st.write(log_data.head())

# Scatter Plot with Plotly
st.subheader("Interactive Scatter Plot")
x_axis = st.selectbox("Select X-axis (numerical variables)", cd_num_features, index=0)
y_axis = st.selectbox("Select Y-axis (numerical variables)", cd_num_features, index=1)
color = st.selectbox("Select color grouping (categorical variables)", cd_cat_features, index=0)
get_scatter_plot_clean(x_axis,y_axis,color)

scattpl_button = st.button("What does this tell us?", key="scatter")
if scattpl_button:
    st.markdown("Hello hello")

# Interactive Box Plot
st.subheader("Interactive Boxplot")
box_column = st.selectbox("Select column for box plot (numerical variables)", cd_num_features, index=0)
get_box_plot(box_column)

box_button = st.button("What does this tell us?", key="box")
if box_button:
    st.markdown("Hello hello")

# Interactive Corr Matrix
st.subheader("Interactive Correlation Matrix")
st.markdown("This correlation matrix uses a dataframe that converts categorical variables into numerical variables.")
fig = px.imshow(
clean_data.select_dtypes(include=['number']).corr(),text_auto=True,aspect="auto",
color_continuous_scale="RdBu_r",  # Red-blue color scale for positive/negative correlations
origin='lower',title="Correlation Matrix",width = 600, height=710)
st.plotly_chart(fig, use_container_width=True)

corr_button = st.button("What does this tell us?", "corr")
if corr_button:
    st.markdown("Hello hello")