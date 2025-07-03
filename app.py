
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             r2_score, mean_squared_error, mean_absolute_error)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="📈 Productivity App Dashboard", layout="wide")

st.title("📈 Productivity App Market Analysis Dashboard")

DATA_URL = "https://raw.githubusercontent.com/username/repo/main/sample_data.csv"  # change to real URL

@st.cache_data
def load_data(url=DATA_URL):
    try:
        df_read = pd.read_csv(url)
    except Exception:
        st.warning("Using bundled sample_data.csv")
        df_read = pd.read_csv('sample_data.csv')
    df_read.replace({"Yes":1,"No":0,"Maybe":np.nan}, inplace=True)
    df_read.fillna(0, inplace=True)
    return df_read

df = load_data()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

tab_vis, tab_clf, tab_clu, tab_arm, tab_reg = st.tabs(
    ["📊 Data Visualization","🤖 Classification","👥 Clustering","🔗 Assoc Rules","📈 Regression"])

with tab_vis:
    st.header("Quick Viz")
    if numeric_cols:
        col = st.selectbox("Histogram column:", numeric_cols)
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig, use_container_width=True)

with tab_clf:
    st.header("Classification (basic)")
    cat_targets = [c for c in df.columns if df[c].nunique()<=3 and c not in numeric_cols]
    if cat_targets:
        target_c = st.selectbox("Target:", cat_targets)
        if st.button("Train"):
            X = pd.get_dummies(df.drop(columns=[target_c]), drop_first=True)
            y = LabelEncoder().fit_transform(df[target_c])
            X_tr, X_te, y_tr, y_te = train_test_split(X,y, test_size=0.3, random_state=1, stratify=y)
            model = RandomForestClassifier(random_state=1).fit(X_tr,y_tr)
            st.write("Accuracy:", accuracy_score(y_te, model.predict(X_te)))

with tab_clu:
    st.header("KMeans quick")
    num_feats = st.multiselect("Features:", numeric_cols, default=numeric_cols[:2])
    if num_feats:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df[num_feats])
        k = st.slider("Clusters",2,5,3)
        if len(df)>=k:
            km = KMeans(n_clusters=k, n_init=10, random_state=1).fit(data_scaled)
            df['Cluster']=km.labels_
            st.write(df[['Cluster']+num_feats].head())

with tab_arm:
    st.header("Association Rules")
    bin_cols=[c for c in df.columns if df[c].isin([0,1]).all()]
    cols_choice=st.multiselect("Boolean columns:",bin_cols, default=bin_cols[:4])
    if cols_choice:
        freq=apriori(df[cols_choice].astype(bool), min_support=0.1, use_colnames=True)
        rules=association_rules(freq, metric="confidence", min_threshold=0.6)
        st.dataframe(rules.head())

# -------------- Regression with manual input ----------------
with tab_reg:
    st.header("📈 Regression with Manual Input")
    if not numeric_cols:
        st.warning("No numeric features.")
    else:
        target = st.selectbox("Select numeric target:", numeric_cols, key="reg_target")
        if st.button("Train models", key="train_reg"):
            X_orig = df.drop(columns=[target])
            X_enc = pd.get_dummies(X_orig, drop_first=True)
            y = df[target]
            X_tr, X_te, y_tr, y_te = train_test_split(X_enc, y, test_size=0.3, random_state=1)
            reg_models = {
                "Linear": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "DecisionTree": DecisionTreeRegressor(random_state=1)
            }
            for m in reg_models.values():
                m.fit(X_tr, y_tr)
            st.session_state["reg_models"] = reg_models
            st.session_state["reg_Xcols"] = X_enc.columns.tolist()
            st.success("Models trained. Scroll below to manual prediction.")

        if "reg_models" in st.session_state:
            st.subheader("Manual Input for Prediction")
            input_dict={}
            with st.form("manual_pred_form"):
                for col in df.drop(columns=[target]).columns:
                    if col in numeric_cols:
                        input_dict[col]=st.number_input(col, value=float(df[col].mean()))
                    else:
                        options=df[col].unique().tolist()
                        input_dict[col]=st.selectbox(col, options)
                submit=st.form_submit_button("Predict")
            if submit:
                inp_df=pd.DataFrame([input_dict])
                inp_enc=pd.get_dummies(inp_df, drop_first=True)
                inp_enc=inp_enc.reindex(columns=st.session_state["reg_Xcols"], fill_value=0)
                model_name=st.selectbox("Choose model:", list(st.session_state["reg_models"].keys()), key="pred_model")
                pred=st.session_state["reg_models"][model_name].predict(inp_enc)[0]
                st.success(f"Predicted {target}: {pred:.2f}")
