# Productivity App Market Analysis Dashboard

This Streamlit dashboard helps evaluate whether there is analytical evidence to justify launching a new productivity-focused mobile application. It offers deep insights, machine‑learning models, clustering, association rule mining, and regression analysis in one interactive interface.

## Features
- **Data Visualization**: Histograms, scatter plots, correlation heatmaps, and more.
- **Classification**: Train and compare KNN, Decision Tree, Random Forest, and Gradient Boosting models.
- **Clustering**: K‑Means with an elbow chart and interactive cluster personas.
- **Association Rules**: Apriori algorithm with adjustable support, confidence, and lift.
- **Regression**: Evaluate Linear, Ridge, Lasso, and Decision Tree regression models.

## How to Run Locally
```bash
git clone https://github.com/yourusername/ProductivityAppDashboard.git
cd ProductivityAppDashboard
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push this repository to GitHub.
2. Sign into **https://streamlit.io/cloud** and click **New app**.
3. Connect your GitHub repo and select `app.py` as the entry‑point.
4. Deploy – Streamlit Cloud will install dependencies from `requirements.txt` and launch the app.

## Data
- The app expects a CSV accessible at the raw GitHub URL specified in `DATA_URL` inside `app.py`.
- A small `sample_data.csv` is included for quick testing. Replace or update the URL to point to your full dataset.

## Author
Generated automatically by ChatGPT.