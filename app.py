
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="📈 Productivity Dashboard", layout="wide")
st.title("📈 Productivity App Market Analysis Dashboard")

DATA_URL = "https://raw.githubusercontent.com/username/repo/main/sample_data.csv"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
    except Exception:
        st.warning("Using bundled sample_data.csv")
        df = pd.read_csv("sample_data.csv")
    df.replace({"Yes":1,"No":0,"Maybe":np.nan}, inplace=True)
    df.fillna(0, inplace=True)
    return df

df = load_data()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Tabs
_, _, _, _, tab_reg = st.tabs(["📊 Data Viz","🤖 Classification","👥 Clustering","🔗 Assoc Rules","📈 Regression"])

# ---------------- Regression Tab -----------------
with tab_reg:
    st.header("📈 Regression (Metrics + Manual Prediction)")

    if not numeric_cols:
        st.error("No numeric columns found.")
    else:
        target_col = st.selectbox("Select numeric target:", numeric_cols)
        run_button = st.button("Train Regression Models")
        if run_button or "reg_trained" in st.session_state:
            # Train models (only once unless target changes)
            if run_button or st.session_state.get("reg_target") != target_col:
                X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
                y = df[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                models = {
                    "Linear": LinearRegression(),
                    "Ridge": Ridge(),
                    "Lasso": Lasso(),
                    "DecisionTree": DecisionTreeRegressor(random_state=42)
                }
                metrics = []
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics.append({
                        "Model": name,
                        "R2": r2_score(y_test, y_pred),
                        "MSE": mean_squared_error(y_test, y_pred),
                        "MAE": mean_absolute_error(y_test, y_pred)
                    })
                    models[name] = model  # update to fitted model
                st.session_state["reg_models"] = models
                st.session_state["reg_Xcols"] = X.columns.tolist()
                st.session_state["reg_target"] = target_col
                st.session_state["reg_metrics_df"] = pd.DataFrame(metrics).set_index("Model")
                st.session_state["reg_trained"] = True

            # Show metrics visually
            metrics_df = st.session_state["reg_metrics_df"].round(3)
            st.subheader("Model Performance")
            st.dataframe(metrics_df)
            bar_fig = px.bar(metrics_df, y="R2", title="R² Scores by Model", text="R2")
            st.plotly_chart(bar_fig, use_container_width=True)

            # Manual input form
            st.subheader("🎯 Manual Prediction")
            with st.form("manual_pred"):
                input_vals = {}
                for col in df.drop(columns=[target_col]).columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        input_vals[col] = st.number_input(col, value=float(df[col].mean()))
                    else:
                        input_vals[col] = st.selectbox(col, sorted(df[col].unique().tolist()))
                submitted = st.form_submit_button("Predict with all models")
            if submitted:
                input_df = pd.DataFrame([input_vals])
                input_enc = pd.get_dummies(input_df, drop_first=True)
                input_enc = input_enc.reindex(columns=st.session_state["reg_Xcols"], fill_value=0)
                preds = {name: model.predict(input_enc)[0] for name, model in st.session_state["reg_models"].items()}
                pred_df = pd.DataFrame.from_dict(preds, orient="index", columns=[f"Predicted {target_col}"]).round(2)
                st.table(pred_df)
                pred_bar = px.bar(pred_df, y=f"Predicted {target_col}", title="Predictions by Model", text=f"Predicted {target_col}")
                st.plotly_chart(pred_bar, use_container_width=True)
