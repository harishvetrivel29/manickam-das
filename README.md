# AI-Powered Life Cycle Assessment (LCA) Platform

Production-ready Streamlit app + ML pipeline (ISO 14040/14044-aligned) with circularity-friendly scenario analysis.

## 📦 Features
- Streamlit UI for scenario building and A/B comparison
- Multi-output ML model predicts **Emissions (kgCO₂e/t)**, **Energy (MJ/t)**, **Waste (kg/t)**
- Alternative suggestions by swapping stage options
- Explainability via aggregated feature importances
- Export CSV and **PDF reports** (ReportLab)
- Synthetic dataset included (replace with real LCI for production)

## 🗂 Structure
```
lca_platform/
  data/
    synthetic_lca_dataset.csv
    dataset_sample_head.csv
  models/
    lca_model.pkl
    feature_columns.json
  app/
    streamlit_app.py
    utils.py
    explainer.py
  reports/
    training_report.json
    scenario_exports/
  requirements.txt
  README.md
```

## 🚀 Quickstart
```bash
# 1) Create & activate a virtual env (example using venv)
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Run the app
streamlit run app/streamlit_app.py
```

## 🔧 Retrain with real data
1. Replace `data/synthetic_lca_dataset.csv` with a real LCI dataset.
2. Ensure columns match or adjust `feature_columns.json` and training script as needed.
3. Retrain a model and overwrite `models/lca_model.pkl`.

## 📝 Notes
- PDF export uses **reportlab**. If missing, `pip install reportlab`.
- You can swap the estimator to **XGBoost/LightGBM** easily.
- Metrics from the included synthetic training are in `reports/training_report.json`.
