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

st.title("📉 Modeling")
st.write("This page will cover different models to predict loan defaults.")

# ----------------------
# Datasets
# ----------------------

@st.cache_data
def loan_data_outliers_dropped():
    # Dataset with just outliers dropped
    outliers_dropped = pd.read_csv('loan_data_outliers_dropped.csv', index_col=0)
    df = outliers_dropped
    return df

@st.cache_data
def loan_data_names_change():
    # Dataset with name changes and adds Z-scores
    names_changes = pd.read_csv('loan_data_name_changes.csv', index_col=0)
    df = names_changes
    return df

@st.cache_data
def loan_data_all_num():
    # Dataset converted to all numerical values
    all_nums = pd.read_csv('loan_data_all_num.csv', index_col=0)
    df = all_nums
    return df

@st.cache_data
def get_hist(color_col,hist_column):
    fig_hist = px.histogram(all_nums, x=hist_column, color=color_col, marginal="box",
                            title=f"Distribution of {hist_column}",
                            template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)

# Load dataset
outliers_dropped = loan_data_outliers_dropped()
names_changes = loan_data_names_change()
all_nums = loan_data_all_num()

# Get columns
numeric_columns = names_changes.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = names_changes.select_dtypes(include="object").columns.tolist()

# Select models
# models = st.selectbox("Select a model to view", )
hist_column = st.selectbox("Select column for histogram", numeric_columns, index=0)
color_col = st.selectbox("Select hist grouping color", categorical_columns, index=0)
get_hist(color_col,hist_column)

# Interactive Histogram
st.subheader("Interactive Histogram")
hist_column = st.selectbox("Select column for histogram (numerical variables)", numeric_columns, index=0)
color_col = st.selectbox("Select hist grouping color (categorical variables)", categorical_columns, index=0)
get_hist(color_col,hist_column)

# Logistic Regression
# Define features and target
target_col = ['loan_status']
features = [col for col in all_nums.columns if col not in target_col]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
transformers=[
("num", RobustScaler(), numeric_columns)
],
sparse_threshold=0
)

X_processed = preprocessor.fit_transform(all_nums[features])
df_transformed = pd.DataFrame(X_processed, columns=numeric_columns)

# Split features and targets
X_processed = df_transformed
y_class = all_nums['loan_status'].astype(int) # Ensure binary target is integer (0/1)

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