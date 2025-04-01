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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
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

def load_dataset(option):
    if option == "Cleaned Dataset":
        return pd.read_csv('loan_data_cleaned.csv')
    elif option == "Log Transformed Dataset":
        return pd.read_csv('loan_data_log.csv')
    else:
        return pd.DataFrame()
    
dataset_option = st.selectbox("Choose a dataset:", ["Cleaned Dataset", "Log Transformed Dataset"])
df = load_dataset(dataset_option)

target_cols = ['loan_status']
features = [col for col in df.columns if col not in target_cols]

cat_features = df.select_dtypes(include=['object']).columns
num_features = [col for col in features if df[col].dtype != "object"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features), # Numeric features being processed; scales num features while reducing effect of outliers using median and IQR
        ("cat", OneHotEncoder(drop='first'), cat_features)
    ],
    sparse_threshold=0
)

X_processed = preprocessor.fit_transform(df[features])

cat_encoded = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features)
all_features = list(num_features) + list(cat_encoded)

df_transformed = pd.DataFrame(X_processed, columns=all_features)
st.write(df_transformed.head()) # All numerical features scaled to a consistent range

X = df.drop('loan_status', axis=1)
y = df['loan_status'].astype(int)
y = y.replace(2,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ----------------------
# Modeling
# ----------------------

# Logistic Regression for Classification
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000)

log_reg.fit(X_train_processed, y_train)
y_pred_log = log_reg.predict(X_test_processed)

acc_log = accuracy_score(y_test, y_pred_log)
print("Logistic Regression Accuracy:", acc_log)

cm1 =  confusion_matrix(y_test, y_pred_log)
st.write(cm1)

plt.figure(figsize=(8, 6))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

precision = precision_score(y_test, y_pred_log)
recall = recall_score(y_test, y_pred_log)
f1 = f1_score(y_test, y_pred_log)
roc_auc = roc_auc_score(y_test, y_pred_log)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC-AUC:", roc_auc)

@st.cache_data
def get_hist(color_col,hist_column):
    fig_hist = px.histogram(df, x=hist_column, color=color_col, marginal="box",
                            title=f"Distribution of {hist_column}",
                            template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)