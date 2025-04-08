import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data():
    df = pd.read_csv("playground-series-s4e10/train.csv")
    df = df.rename(columns={
        'cb_person_default_on_file': 'history_of_default',
        'cb_person_cred_hist_length': 'cred_hist_length'
    })
    return df

def preprocess_data(df):
    # Separate features
    target_col = 'loan_status'
    features = df.drop(columns=[target_col])
    target = df[target_col].astype(int).replace(2, 1)

    cat_features = features.select_dtypes(include=['object']).columns
    num_features = features.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first"), cat_features)
        ],
        sparse_threshold=0
    )

    X_processed = preprocessor.fit_transform(features)
    return X_processed, target, preprocessor