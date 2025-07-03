
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
        df_local = pd.read_csv(url)
    except Exception:
        st.warning("Could not fetch data from GitHub URL. Loading local 'sample_data.csv' instead.")
        df_local = pd.read_csv('sample_data.csv')
    # Basic cleaning
    df_local.replace({"Yes": 1, "No": 0, "Maybe": np.nan}, inplace=True)
    df_local.fillna(0, inplace=True)
    # Ensure numeric conversion where possible
    for col in df_local.columns:
        if df_local[col].dtype == object and df_local[col].str.replace('.', '', 1).str.isnumeric().all():
            df_local[col] = df_local[col].astype(float)
    return df_local

df = load_data()
st.sidebar.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# ----------------------------------------------
# Tabs
# ----------------------------------------------
tab_vis, tab_clf, tab_clu, tab_arm, tab_reg = st.tabs(
    ["📊 Data Visualization", "🤖 Classification", "👥 Clustering",
     "🔗 Association Rule Mining", "📈 Regression"]
)

# ==============================================
# Data Visualization Tab
# ==============================================
with tab_vis:
    st.header("📊 Exploratory Data Visualization")

    # ---- KPI Metrics ----
    kpi_cols = st.columns(3)
    if "Daily_Minutes_Spent" in df.columns:
        avg_minutes = df["Daily_Minutes_Spent"].mean()
        heavy_pct = (df["Daily_Minutes_Spent"] > 180).mean() * 100
        kpi_cols[0].metric("Avg Daily Minutes", f"{avg_minutes:.1f}")
        kpi_cols[1].metric("% Heavy Users (>180 min)", f"{heavy_pct:.1f}%")
    if "Monthly_Income" in df.columns:
        avg_income = df["Monthly_Income"].mean()
        kpi_cols[2].metric("Avg Monthly Income", f"${avg_income:,.0f}")

    st.markdown("---")

    # ---- Age Distribution & Minutes KDE ----
    if {"Age", "Daily_Minutes_Spent"}.issubset(df.columns):
        fig_kde, ax = plt.subplots()
        sns.kdeplot(data=df, x="Age", fill=True, ax=ax, label="Age")
        ax2 = ax.twinx()
        sns.kdeplot(data=df, x="Daily_Minutes_Spent", fill=True, color="orange", ax=ax2, label="Minutes Spent")
        ax.set_xlabel("Age / Minutes Spent")
        ax.set_title("Age Distribution & Minutes Spent (KDE)")
        st.pyplot(fig_kde)

    # ---- Pay Amount vs Subscription Intent ----
    if {"Pay_Amount", "Would_Subscribe"}.issubset(df.columns):
        fig_box = px.box(df, x="Would_Subscribe", y="Pay_Amount", title="Pay Amount vs Subscription Intent")
        st.plotly_chart(fig_box, use_container_width=True)

    # ---- Platform Usage Share ----
    platform_cols = [c for c in df.columns if c.lower().startswith("uses_")]
    if platform_cols:
        usage_counts = df[platform_cols].sum().sort_values(ascending=False)
        fig_bar = px.bar(usage_counts, x=usage_counts.index.str.replace("Uses_", "").str.title(),
                         y=usage_counts.values,
                         title="Platform Usage Share")
        fig_bar.update_layout(xaxis_title="Platform", yaxis_title="User Count")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---- Numeric Correlation Heatmap ----
    st.subheader("Numeric Correlation Heatmap")
    corr = df[numeric_cols].corr()
    fig_corr, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig_corr)

    # ---- Custom Scatter or Histogram ----
    st.markdown("### Custom Scatter / Histogram")
    chart_type = st.radio("Choose chart type:", ["Scatter", "Histogram"], horizontal=True, key="chart_type")
    if chart_type == "Scatter":
        scatter_x = st.selectbox("X-axis:", numeric_cols, index=0, key="scatter_x")
        scatter_y = st.selectbox("Y-axis:", numeric_cols, index=1, key="scatter_y")
        color_by = st.selectbox("Color by (categorical):", cat_cols if cat_cols else [None], key="color")
        fig_scatter = px.scatter(df, x=scatter_x, y=scatter_y, color=color_by,
                                 title=f"{scatter_x} vs {scatter_y}")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        num_choice = st.selectbox("Select numeric column for histogram:", numeric_cols, key="hist_col")
        bins = st.slider("Number of bins:", 5, 50, 20, key="bins_hist")
        fig_hist = px.histogram(df, x=num_choice, nbins=bins, title=f"Distribution of {num_choice}")
        st.plotly_chart(fig_hist, use_container_width=True)

# ==============================================
# Classification Tab
# ==============================================
with tab_clf:
    st.header("🤖 Classification Models")
    cat_candidates = [c for c in df.columns if df[c].nunique() <= 3 and df[c].dtype != float]
    if not cat_candidates:
        st.warning("No suitable binary/ternary target variables found.")
    else:
        target_col = st.selectbox("Select target variable:", cat_candidates, key="target")
        run_clf = st.button("Run Classification", key="run_clf")
        if run_clf:
            # Prepare X, y
            X = df.drop(columns=[target_col])
            X = pd.get_dummies(X, drop_first=True)
            y_raw = df[target_col]
            if y_raw.dtype == 'O':
                le = LabelEncoder()
                y = le.fit_transform(y_raw)
            else:
                y = y_raw.values.astype(int)

            if len(np.unique(y)) < 2:
                st.error("Target variable must have at least two classes.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y)

                models = {
                    "KNN": KNeighborsClassifier(),
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "Random Forest": RandomForestClassifier(random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
                }

                metrics_table, preds_dict = [], {}

                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    preds_dict[name] = y_pred_test
                    metrics_table.append({
                        "Model": name,
                        "Train Acc": accuracy_score(y_train, y_pred_train),
                        "Test Acc": accuracy_score(y_test, y_pred_test),
                        "Precision": precision_score(y_test, y_pred_test, zero_division=0, average='binary'),
                        "Recall": recall_score(y_test, y_pred_test, zero_division=0, average='binary'),
                        "F1": f1_score(y_test, y_pred_test, zero_division=0, average='binary')
                    })

                st.subheader("Performance Metrics")
                st.dataframe(pd.DataFrame(metrics_table).set_index("Model").round(3))

                cm_model = st.selectbox("Select model for Confusion Matrix:", list(models.keys()), key="cm_model")
                cm = confusion_matrix(y_test, preds_dict[cm_model])
                fig_cm, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                st.pyplot(fig_cm)

                # ROC Curves
                st.subheader("ROC Curve Comparison")
                fig_roc = go.Figure()
                for name, model in models.items():
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_test)[:, 1]
                    else:
                        # For models without predict_proba, use decision_function and min-max scale
                        y_scores = model.decision_function(X_test)
                        y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
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
                    # Align with training columns
                    new_df_proc = new_df_proc.reindex(columns=X.columns, fill_value=0)
                    preds = models[pred_model_name].predict(new_df_proc)
                    new_df["Prediction"] = preds
                    st.dataframe(new_df.head())
                    csv_out = new_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Predictions CSV", data=csv_out,
                                       file_name="predictions.csv", mime="text/csv")

# ==============================================
# Clustering Tab
# ==============================================
with tab_clu:
    st.header("👥 K-Means Clustering")
    available_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    cluster_features = st.multiselect("Select features for clustering:", available_numeric, default=available_numeric[:4], key="clus_feats")
    if cluster_features:
        max_k = min(10, max(2, len(df) - 1))
        if max_k < 2:
            st.error("Not enough samples to perform clustering.")
        else:
            k = st.slider("Select number of clusters (k):", 2, max_k, min(3, max_k), key="k_slider")

            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(df[cluster_features])

            # Elbow plot
            distortions = []
            K_range = range(1, max_k + 1)
            for ki in K_range:
                km = KMeans(n_clusters=ki, random_state=42, n_init=10)
                km.fit(data_scaled)
                distortions.append(km.inertia_)
            fig_elbow, ax = plt.subplots()
            ax.plot(K_range, distortions, marker='o')
            ax.set_xlabel('k'); ax.set_ylabel('Inertia'); ax.set_title("Elbow Method")
            st.pyplot(fig_elbow)

            # Final clustering
            km_final = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = km_final.fit_predict(data_scaled)
            df["Cluster"] = clusters
            st.subheader("Data with Cluster Labels")
            st.dataframe(df.head())

            st.subheader("Cluster Personas (mean stats)")
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
    bool_cols = [c for c in df.columns if df[c].dropna().isin([0,1]).all()]
    ar_cols = st.multiselect("Select boolean columns:", bool_cols, default=bool_cols[:6], key="ar_cols")
    if ar_cols:
        min_sup = st.slider("Minimum Support:", 0.01, 1.0, 0.1, 0.01, key="sup")
        min_conf = st.slider("Minimum Confidence:", 0.01, 1.0, 0.5, 0.01, key="conf")
        min_lift = st.slider("Minimum Lift:", 1.0, 10.0, 1.0, 0.1, key="lift")

        df_bool = df[ar_cols].astype(bool)
        freq_items = apriori(df_bool, min_support=min_sup, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
        rules = rules[rules['lift'] >= min_lift]
        if not rules.empty:
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
            top_rules = rules.sort_values(by='lift', ascending=False).head(10)
            st.dataframe(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(2))
        else:
            st.info("No association rules meet the parameters. Consider lowering thresholds.")
    else:
        st.info("Please select at least one boolean column.")

# ==============================================
# Regression Tab
# ==============================================
with tab_reg:
    st.header("📈 Regression Models")
    if not numeric_cols:
        st.warning("No numeric columns found.")
    else:
        target_num = st.selectbox("Select numeric target variable:", numeric_cols, key="reg_target")
        if st.button("Run Regression", key="run_reg"):
            X_reg = df.drop(columns=[target_num])
            X_reg = pd.get_dummies(X_reg, drop_first=True)
            y_reg = df[target_num]
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                X_reg, y_reg, test_size=0.3, random_state=42)

            reg_models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
            }

            reg_metrics, preds_reg = [], {}
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

            model_choice = st.selectbox("Select model for Actual vs Predicted:", list(reg_models.keys()), key="reg_choice")
            fig_reg = px.scatter(x=y_test_r, y=preds_reg[model_choice],
                                 labels={'x': 'Actual', 'y': 'Predicted'},
                                 title=f"{model_choice}: Actual vs Predicted")
            fig_reg.add_shape(type='line',
                              x0=y_test_r.min(), y0=y_test_r.min(),
                              x1=y_test_r.max(), y1=y_test_r.max(),
                              line=dict(dash='dash'))
            st.plotly_chart(fig_reg, use_container_width=True)
