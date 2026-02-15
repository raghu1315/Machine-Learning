import streamlit as st
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


st.set_page_config(
    page_title="Heart Disease ML App",
    layout="wide"
)

st.title("Heart Disease Classification – ML Model Comparison")

st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload TEST CSV file",
    type=["csv"]
)

model_name = st.sidebar.selectbox(
    "Select Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)


@st.cache_data
def load_data():
    """Load and cache the dataset"""
    url = "https://raw.githubusercontent.com/2025aa05526/2025aa05526/main/heart_disease_uci.csv"
    return pd.read_csv(url)


@st.cache_data
def preprocess_data(df):
    """Preprocess and cache the data preparation steps"""
    df["target"] = (df["num"] > 0).astype(int)
    df_processed = df.drop(["id", "num"], axis=1, errors="ignore")

    categorical_cols = df_processed.select_dtypes(include="object").columns.tolist()
    if "target" in categorical_cols:
        categorical_cols.remove("target")

    df_encoded = pd.get_dummies(
        df_processed,
        columns=categorical_cols,
        drop_first=True
    )

    X = df_encoded.drop("target", axis=1)
    y = df_encoded["target"]
    
    return X, y


@st.cache_data
def split_and_scale_data(X, y):
    """Split and scale data - cached to prevent recomputation"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


@st.cache_resource
def train_model(model_name, X_train, y_train):
    """Train and cache the model - this is the key optimization!"""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }
    
    model = models[model_name]
    model.fit(X_train, y_train)
    return model


# Load and preprocess data (cached)
df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Preprocess (cached)
X, y = preprocess_data(df)

# Split and scale (cached)
X_train, X_test, y_train, y_test = split_and_scale_data(X, y)

# Train model (cached) - only trains once per model type!
model = train_model(model_name, X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

# Display metrics
st.subheader("📊 Evaluation Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{acc:.3f}")
col1.metric("AUC", f"{auc:.3f}")

col2.metric("Precision", f"{precision:.3f}")
col2.metric("Recall", f"{recall:.3f}")

col3.metric("F1 Score", f"{f1:.3f}")
col3.metric("MCC", f"{mcc:.3f}")

# Confusion Matrix
st.subheader("📋 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.dataframe(
    pd.DataFrame(
        cm,
        columns=["Predicted 0", "Predicted 1"],
        index=["Actual 0", "Actual 1"]
    )
)

# Classification Report
st.subheader("📄 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())
