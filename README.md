# 🏠 Real Estate Investment Advisor

Predicts whether an Indian property is a **Good Investment** (classification) and estimates its **price after 5 years** (regression), served through an interactive Streamlit app with MLflow experiment tracking.

**Live app:** _add Streamlit Cloud link_ · **Report:** [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)

---

## Results (held-out test set, 50,000 properties)

| Task | Model | Key metrics | Baseline |
|---|---|---|---|
| Good Investment | XGBoost (tuned) | Accuracy **94.7%** · F1 **0.938** · ROC-AUC **0.991** | Majority class: 58.2% |
| 5-year price | XGBoost | RMSE **₹19.4 L** · MAE ₹13.5 L · R² 0.991 | Fixed 8% growth: RMSE ₹39.1 L |

The regression RMSE is within 2% of the theoretical noise floor (≈ ₹19.1 L).

## Tech stack
Python · Pandas · Scikit-learn · XGBoost · MLflow · Streamlit · Plotly · SciPy

## Project structure
```
├── app/                      Streamlit app (OOP, modular)
│   ├── app.py                entry point
│   ├── settings.py           paths + constants
│   ├── core/                 loader, feature builder, advisor
│   └── ui/                   one class per tab + chart factory
├── data/
│   ├── raw/                  india_housing_prices.csv
│   └── processed/            cleaned CSV, app_data.parquet
├── models/                   exported models + app_metadata.json
├── notebooks/
│   ├── 01_data_understanding.ipynb   inspection + cleaning
│   ├── 02_eda.ipynb                  20 EDA questions + stats tests
│   ├── 03_feature_engineering.ipynb  features + target creation
│   └── 04_model_training.ipynb       training, tuning, MLflow, export
├── docs/                     report, MLflow guide, demo script, figures
└── requirements.txt
```

## Setup
```bash
git clone https://github.com/AdityaDev-17/Real-Estate-Investment-Advisor.git
cd Real-Estate-Investment-Advisor
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Run
**App** (from the project root):
```bash
streamlit run app/app.py
```
**Notebooks:** run `01 → 04` in order to rebuild the data, models and MLflow logs.

**MLflow UI:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Then open http://127.0.0.1:5000. See [docs/MLFLOW_GUIDE.md](docs/MLFLOW_GUIDE.md).

## App features
- **Predict:** enter property details → verdict with confidence, 5-year price, growth path, and a 6-criteria explanation.
- **Explore:** filter 250k properties by city, type, BHK, price and size; download results.
- **Insights:** City × BHK heatmap, growth by city, tier trends, feature importance, model metrics.

## Key findings
- The dataset is **synthetic**: features are uniformly distributed and independent, and after Bonferroni correction **no feature significantly affects price**.
- Because there are no historical sale prices, both targets are **rule-based** (documented in the report). The models learn and generalise those rules; they do not discover market patterns.

## Documentation
- [Project Report](docs/PROJECT_REPORT.md): methodology, findings, results, limitations
- [MLflow Guide](docs/MLFLOW_GUIDE.md): navigating experiments and the model registry
- [Demo Script](docs/DEMO_SCRIPT.md): 8–9 minute walkthrough

## Author
**Aditya Singh** · M.Tech AI & ML, BITS Pilani (Hyderabad) · [LinkedIn](https://www.linkedin.com/in/aditya-17-dev) · [GitHub](https://github.com/AdityaDev-17)