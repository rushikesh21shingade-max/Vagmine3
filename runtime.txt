# VAGMINÉ — Beauty DNA™ Survey Analytics Dashboard

An interactive Streamlit dashboard that analyses VAGMINÉ's synthetic customer survey data to
validate the luxury beauty personalization business model. Built for investor/stakeholder
presentations and Master's dissertation use.

## Features
- **Executive Overview** — KPIs, segment sizes, estimated NPS
- **Descriptive Analytics** — demographics, purchase intent, Beauty DNA™ interest, membership,
  pricing, sustainability, product preferences, and an interactive cross-tabulation builder
- **Diagnostic Analytics** — driver analysis, correlation heatmaps, interactive driver vs outcome explorer
- **Customer Segmentation** — K-Means behavioural clustering with radar/sunburst visuals and segment narratives
- **Predictive Modelling** — KNN, Decision Tree, Random Forest, and Gradient Boosting classifiers
  with train/test accuracy, precision, recall, F1, ROC/AUC, confusion matrices, and feature importance
- **Insights & Recommendations** — findings translated into product, pricing, segmentation and
  marketing recommendations

## Project Structure
```
vagmine_dashboard/
├── app.py                  # Everything: data processing, ML, and the Streamlit app
├── requirements.txt
├── runtime.txt              # pins Python 3.11 for Streamlit Cloud
├── data/
│   └── VAGMINE_Synthetic_Survey_Data.xlsx
└── .streamlit/
    └── config.toml         # Theme configuration
```

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Create a new **public GitHub repository** (e.g. `vagmine-dashboard`).
2. Unzip this folder and push all of its contents to the repo root (keep the folder structure,
   especially `data/`, `utils/`, and `.streamlit/`).
   ```bash
   git init
   git add .
   git commit -m "Initial commit: VAGMINÉ analytics dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/vagmine-dashboard.git
   git push -u origin main
   ```
3. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **New app**.
4. Select your repository, branch `main`, and set the main file path to `app.py`.
5. Click **Deploy**. The first build takes 1–3 minutes while dependencies install.

## Notes
- The dataset is synthetic and included in `data/`. Replace it with live survey data (same column
  names) to re-run the full pipeline on real customers — no code changes required.
- Models train on the fly (cached) on app load — first load may take a few seconds.
