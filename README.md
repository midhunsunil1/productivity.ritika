# Productivity App Market Analysis Dashboard (v6)

This Streamlit dashboard lets you explore survey data to assess demand for a productivity‑focused mobile app.
It features:

- **Data Visualization** – quick histograms and exploratory charts  
- **Classification** – basic Random Forest example  
- **Clustering** – K‑Means with adjustable k  
- **Association Rules** – Apriori mining on boolean columns  
- **Regression** – train 4 models (Linear, Ridge, Lasso, Decision Tree) and
  *manually enter feature values* to predict the chosen numeric target.

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud deployment
1. Push this folder to a GitHub repo.  
2. Go to <https://streamlit.io/cloud>, create a **New app**, and point it to
   `app.py` in that repo.  
   Streamlit Cloud installs dependencies from `requirements.txt` automatically.

## Using the Regression tab
1. Select the target numeric column (e.g., **Pay_Amount**).  
2. Click **Train Regression Models** – the app trains Linear, Ridge, Lasso, and Decision‑Tree regressors.  
3. Fill in independent variables in the **Manual Prediction** form.  
4. Hit **Predict with all models** – a table and bar‑chart show each model’s prediction.

### Interpretation tips
- Higher **R²** indicates better fit; compare models via the bar‑chart.  
- Use the predictions table to see how model choice affects pay‑amount expectations.  
- Retrain after switching the target variable or adding new survey data.

---

**Sample data** is bundled in `sample_data.csv`. Replace `DATA_URL` at the top
of `app.py` with a raw GitHub link to load your own dataset.