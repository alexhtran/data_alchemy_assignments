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
    layout="wide",
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

cleaned = pd.read_csv('loan_data_cleaned.csv')
log = pd.read_csv('loan_data_log.csv')

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

st.subheader("Dataset Preview")
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
st.subheader("Logistic Regression")
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000)

log_reg.fit(X_train_processed, y_train)
y_pred_log = log_reg.predict(X_test_processed)

acc_log = accuracy_score(y_test, y_pred_log)
st.write("Logistic Regression Accuracy:", acc_log)

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

st.write("Precision:", precision)
st.write("Recall:", recall)
st.write("F1-Score:", f1)
st.write("ROC-AUC:", roc_auc)

clean_logi, log_logi = st.columns(2)
with clean_logi:
    with st.expander("What does the cleaned data tell us?"):
        st.markdown("The accuracy is at 0.84, precision at 0.46, recall at 0.82, F1 at 0.59, and ROC-AUC is 0.83.\n\n" \
        "For our data, we will focus on recall since the goal of our models is to see how many applicants are likely to default. Therefore, our recall is fairly good at 0.82.")

with log_logi:
    with st.expander("What does the log transformed data tell us?"):
        st.markdown("The accuracy is at 0.81, precision at 0.39, recall at 0.82, F1 at 0.53, and ROC-AUC is 0.81.\n\n" \
        "The accuracy and precision are slightly worse than the cleaned dataset, but the recall values are the same.")

# Decision Tree
st.subheader("Decision Tree")
dt_clf = DecisionTreeClassifier(random_state=50,max_depth=5)
dt_clf.fit(X_train_processed, y_train)
y_pred_dt_clf = dt_clf.predict(X_test_processed)
acc_dt_clf = accuracy_score(y_test, y_pred_dt_clf)
st.write("Decision Tree Classifier Accuracy:", acc_dt_clf)
st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt_clf))

pre_dt = precision_score(y_test, y_pred_dt_clf)
rec_dt = recall_score(y_test, y_pred_dt_clf)
f1_dt = f1_score(y_test, y_pred_dt_clf)
roc_auc_dt = roc_auc_score(y_test, y_pred_dt_clf)

st.write("Precision:", pre_dt)
st.write("Recall:", rec_dt)
st.write("F1-Score:", f1_dt)
st.write("ROC-AUC:", roc_auc_dt)

clean_dt, log_dt = st.columns(2)
with clean_dt:
    with st.expander("What does the cleaned data tell us?"):
        st.markdown("The accuracy is at 0.94, precision at 0.89, recall at 0.61, F1 at 0.73, and ROC-AUC is 0.80.\n\n"
                    "The recall is a bit poor, so decision tree might not be the best for determining defaulters. However, the accuracy and precision is much better than logistic regression.")

with log_dt:
    with st.expander("What does the log transformed data tell us?"):
        st.markdown("The accuracy is at 0.94, precision at 0.89, recall at 0.57, F1 at 0.70, and ROC-AUC is 0.78.\n\n"
                    "The recall and F1 are slightly worse than the cleaned dataset, but the accuracy and precision are the same.")

# Random Forest Classifier
st.subheader("Random Forest")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_processed, y_train)
y_pred_rf_clf = rf_clf.predict(X_test_processed)
acc_rf_clf = accuracy_score(y_test, y_pred_rf_clf)
st.write("Random Forest Classifier Accuracy:", acc_rf_clf)
st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf_clf))

pre_rf = precision_score(y_test, y_pred_rf_clf)
rec_rf = recall_score(y_test, y_pred_rf_clf)
f1_rf = f1_score(y_test, y_pred_rf_clf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf_clf)

st.write("Precision:", pre_rf)
st.write("Recall:", rec_rf)
st.write("F1-Score:", f1_rf)
st.write("ROC-AUC:", roc_auc_rf)

clean_rf, log_rf = st.columns(2)
with clean_rf:
    with st.expander("What does the cleaned data tell us?"):
        st.markdown("The accuracy is at 0.95, precision at 0.92, recall at 0.71, F1 at 0.80, and ROC-AUC is 0.85.\n\n"
        "The recall is okay, but much better than decision tree.")

with log_rf:
    with st.expander("What does the log transformed data tell us?"):
        st.markdown("The accuracy is at 0.95, precision at 0.90, recall at 0.68, F1 at 0.78, and ROC-AUC is 0.83.\n\n"
                    "The accuracy is the same but precision, recall, and F1 are slightly worse.")

# kNN
st.subheader("KNeighbors Classifiers")
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_processed, y_train)
knn_predict = knn.predict(X_test_processed)

knn.score(X_test_processed, y_test)

acc_knn = accuracy_score(y_test, knn_predict)

pre_knn = precision_score(y_test, knn_predict)
rec_knn = recall_score(y_test, knn_predict)
f1_knn = f1_score(y_test, knn_predict)
roc_auc_knn = roc_auc_score(y_test, knn_predict)

st.write("KNN Accuracy:", acc_knn)
st.write("Confusion Matrix:\n", confusion_matrix(y_test, knn_predict))

st.write("Precision:", pre_knn)
st.write("Recall:", rec_knn)
st.write("F1-Score:", f1_knn)
st.write("ROC-AUC:", roc_auc_knn)

clean_knn, log_knn = st.columns(2)
with clean_knn:
    with st.expander("What does the cleaned data tell us?"):
        st.markdown("The accuracy is at 0.93, precision at 0.80, recall at 0.62, F1 at 0.70, and ROC-AUC is 0.80.\n\n"
                    "The recall is a bit poor, so kNN might not be the best with a dataset that is not normally distributed.")

with log_knn:
    with st.expander("What does the log transformed data tell us?"):
        st.markdown("The accuracy is at 0.93, precision at 0.82, recall at 0.58, F1 at 0.78, and ROC-AUC is 0.78.\n\n"
                    "The accuracy is the same, and precision and F1 are better. However, recall is worse.")
        
st.subheader("💭 Insights")
st.markdown("Generally, accuracy, precision, and recall scores were lower with a log transformed dataset than the cleaned dataset.\n\n"
            "Recall is the best metric as we want to make sure all potential defaulters are identified. If they're misidentified, it can be costly to shareholders."
            "This could be because having a dataset transformed using log functions might not be the best with the models! The dataset could've been transformed using natural log, or not transformed at all - especially since the dataset was created from a deep learning model.\n\n"
            "The model also included categorical variables, some which could have been removed or adjusted to improve model performance since they may have not been meaningful to loan_status.\n\n"
            "Moreover, decision trees and random forests have higher scores with the cleaned dataset (not fully transformed). This is because these models are generally better with an uncleaned dataset since it relies on hierarchical splits.\n\n"
            "**Overall**, some next steps could be to adjusting the EDA to create different models and keep going over the life cycle.")