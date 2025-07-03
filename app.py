
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

st.set_page_config(page_title="📈 Productivity App Market Dashboard", layout="wide")

st.title("📈 Productivity App Market Analysis Dashboard")

# ----------------------------------------------
# 1. Data Loading
# ----------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/username/repo/main/sample_data.csv"  # <-- replace with your raw GitHub URL

@st.cache_data
def load_data(url=DATA_URL):
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.warning("Could not fetch data from GitHub URL. Loading local 'sample_data.csv' instead.")
        df = pd.read_csv('sample_data.csv')
    # Basic cleaning
    df.replace({"Yes": 1, "No": 0, "Maybe": np.nan}, inplace=True)
    df.fillna(0, inplace=True)
    return df

df = load_data()
st.sidebar.success(f"Dataset loaded with shape: {df.shape}")

# ----------------------------------------------
# Tabs
# ----------------------------------------------
tab_vis, tab_clf, tab_clu, tab_arm, tab_reg = st.tabs(
    ["📊 Data Visualization", "🤖 Classification", "🧩 Clustering",
     "🔗 Association Rule Mining", "📈 Regression"]
)

# ==============================================
# Data Visualization Tab
# ==============================================
with tab_vis:
    st.header("📊 Exploratory Data Visualization")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    st.subheader("Histogram of a Numeric Column")
    num_choice = st.selectbox("Select numeric column for histogram:", numeric_cols, key="hist")
    bins = st.slider("Number of bins:", 5, 50, 20, key="bins")
    fig_hist = px.histogram(df, x=num_choice, nbins=bins, title=f"Distribution of {num_choice}")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Scatter Plot")
    scatter_x = st.selectbox("X-axis:", numeric_cols, index=0, key="scatter_x")
    scatter_y = st.selectbox("Y-axis:", numeric_cols, index=1, key="scatter_y")
    color_by = st.selectbox("Color by (categorical):", cat_cols, key="color")
    fig_scatter = px.scatter(df, x=scatter_x, y=scatter_y, color=color_by,
                             hover_data=df.columns,
                             title=f"{scatter_x} vs {scatter_y} colored by {color_by}")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df[numeric_cols].corr()
    fig_corr, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig_corr)

# ==============================================
# Classification Tab
# ==============================================
with tab_clf:
    st.header("🤖 Classification Models")
    cat_candidates = [c for c in df.columns if df[c].nunique() <= 3]  # likely binary/categorical
    if not cat_candidates:
        st.warning("No suitable binary/class target variables found.")
    else:
        target_col = st.selectbox("Select target variable (binary preferred):", cat_candidates, key="target")
        if st.button("Run Classification", key="run_clf"):
            # Prepare X, y
            X = df.drop(columns=[target_col])
            # One-hot encode categoricals
            X = pd.get_dummies(X, drop_first=True)
            y_raw = df[target_col]
            # Encode target if not numeric
            if y_raw.dtype == 'O':
                le = LabelEncoder()
                y = le.fit_transform(y_raw)
            else:
                y = y_raw.values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

            models = {
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }

            metrics_table = []
            preds_dict = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                preds_dict[name] = model.predict(X_test)

                metrics_table.append({
                    "Model": name,
                    "Train Accuracy": accuracy_score(y_train, y_pred_train),
                    "Test Accuracy": accuracy_score(y_test, y_pred_test),
                    "Precision": precision_score(y_test, y_pred_test, zero_division=0),
                    "Recall": recall_score(y_test, y_pred_test, zero_division=0),
                    "F1-Score": f1_score(y_test, y_pred_test, zero_division=0)
                })

            st.subheader("Performance Metrics")
            st.dataframe(pd.DataFrame(metrics_table).set_index("Model").round(3))

            # Confusion Matrix
            cm_model = st.selectbox("Select model for Confusion Matrix:", list(models.keys()), key="cm_model")
            cm = confusion_matrix(y_test, preds_dict[cm_model])
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"{cm_model} - Confusion Matrix")
            st.pyplot(fig_cm)

            # ROC Curves
            st.subheader("ROC Curve Comparison")
            fig_roc = go.Figure()
            for name, model in models.items():
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:,1]
                else:
                    y_prob = model.decision_function(X_test)
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_score = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr,
                                             mode='lines',
                                             name=f"{name} (AUC={auc_score:.2f})"))
            fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                              line=dict(dash='dash'))
            fig_roc.update_layout(xaxis_title='False Positive Rate',
                                  yaxis_title='True Positive Rate',
                                  legend_title='Models')
            st.plotly_chart(fig_roc, use_container_width=True)

            # Prediction on new data
            st.subheader("Predict on New Data")
            uploaded_file = st.file_uploader("Upload new data CSV (without target column):", type=["csv"], key="pred_upload")
            pred_model_name = st.selectbox("Select model for prediction:", list(models.keys()), key="pred_model")
            if uploaded_file is not None:
                new_df = pd.read_csv(uploaded_file)
                new_df_proc = pd.get_dummies(new_df, drop_first=True)
                # Align columns with training set
                new_df_proc = new_df_proc.reindex(columns=X.columns, fill_value=0)
                pred_model = models[pred_model_name]
                new_preds = pred_model.predict(new_df_proc)
                new_df["Prediction"] = new_preds
                st.write(new_df.head())
                csv_out = new_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", data=csv_out,
                                   file_name="predictions.csv", mime="text/csv")
# ==============================================
# Clustering Tab
# ==============================================
with tab_clu:
    st.header("🧩 K-Means Clustering")
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    cluster_features = st.multiselect("Select features for clustering:", all_numeric, default=all_numeric[:4], key="clus_feats")
    if cluster_features:
        k = st.slider("Select number of clusters (k):", 2, 10, 3, key="k_slider")
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df[cluster_features])
        # Elbow plot
        distortions = []
        K_range = range(1, 11)
        for ki in K_range:
            km = KMeans(n_clusters=ki, random_state=42, n_init='auto')
            km.fit(data_scaled)
            distortions.append(km.inertia_)
        fig_elbow, ax = plt.subplots()
        ax.plot(K_range, distortions, marker='o')
        ax.set_xlabel('k'); ax.set_ylabel('Inertia'); ax.set_title("Elbow Method")
        st.pyplot(fig_elbow)

        km_final = KMeans(n_clusters=k, random_state=42, n_init='auto')
        clusters = km_final.fit_predict(data_scaled)
        df["Cluster"] = clusters
        st.subheader("Data with Cluster Labels")
        st.dataframe(df.head())

        st.subheader("Cluster Personas")
        persona = df.groupby("Cluster")[cluster_features].mean().round(2)
        st.dataframe(persona)

        csv_cluster = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data", data=csv_cluster,
                           file_name="clustered_data.csv", mime="text/csv")
    else:
        st.info("Select at least one numeric feature for clustering.")

# ==============================================
# Association Rule Mining Tab
# ==============================================
with tab_arm:
    st.header("🔗 Association Rule Mining (Apriori)")
    boolean_cols = [c for c in df.columns if df[c].dropna().isin([0,1]).all()]
    ar_cols = st.multiselect("Select boolean columns for association rules:", boolean_cols, default=boolean_cols[:6], key="arcols")
    if ar_cols:
        min_sup = st.slider("Minimum Support:", 0.01, 1.0, 0.1, 0.01, key="sup")
        min_conf = st.slider("Minimum Confidence:", 0.01, 1.0, 0.5, 0.01, key="conf")
        min_lift = st.slider("Minimum Lift:", 1.0, 10.0, 1.0, 0.1, key="lift")

        df_bool = df[ar_cols].astype(bool)
        freq_items = apriori(df_bool, min_support=min_sup, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
        rules = rules[rules['lift'] >= min_lift]
        if not rules.empty:
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            top_rules = rules.sort_values(by='lift', ascending=False).head(10)
            st.dataframe(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(2))
        else:
            st.info("No association rules meet the criteria. Try lowering thresholds.")
    else:
        st.info("Select at least one boolean column.")

# ==============================================
# Regression Tab
# ==============================================
with tab_reg:
    st.header("📈 Regression Models")
    num_targets = [c for c in numeric_cols if c != "Cluster"]
    if not num_targets:
        st.warning("No numeric target variables available.")
    else:
        target_num = st.selectbox("Select numeric target variable:", num_targets, key="reg_target")
        if st.button("Run Regression", key="run_reg"):
            X_reg = df.drop(columns=[target_num])
            X_reg = pd.get_dummies(X_reg, drop_first=True)
            y_reg = df[target_num]
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

            reg_models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
            }

            reg_metrics = []
            preds_reg = {}

            for name, model in reg_models.items():
                model.fit(X_train_r, y_train_r)
                y_pred = model.predict(X_test_r)
                preds_reg[name] = y_pred
                reg_metrics.append({
                    "Model": name,
                    "R2": r2_score(y_test_r, y_pred),
                    "MSE": mean_squared_error(y_test_r, y_pred),
                    "MAE": mean_absolute_error(y_test_r, y_pred)
                })

            st.subheader("Regression Performance")
            st.dataframe(pd.DataFrame(reg_metrics).set_index("Model").round(3))

            # Scatter plot predicted vs actual
            reg_choice = st.selectbox("Select model for Actual vs Predicted plot:", list(reg_models.keys()), key="reg_choice")
            fig_reg = px.scatter(x=y_test_r, y=preds_reg[reg_choice],
                                 labels={'x': 'Actual', 'y': 'Predicted'},
                                 title=f"{reg_choice} - Actual vs Predicted")
            fig_reg.add_shape(type='line', x0=y_test_r.min(), y0=y_test_r.min(),
                              x1=y_test_r.max(), y1=y_test_r.max(),
                              line=dict(dash='dash'))
            st.plotly_chart(fig_reg, use_container_width=True)
