# MLflow Navigation Guide

How to open, explore and present the experiment tracking for this project.

---

## 1. Launch the UI

From the **project root** (the folder that contains `mlflow.db`), with the venv active:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open **http://127.0.0.1:5000**. Keep the terminal running; press `Ctrl + C` to stop.

> ⚠️ If you run plain `mlflow ui` without `--backend-store-uri`, MLflow opens an empty default store and the UI shows **no experiments**. Always pass the SQLite path.

---

## 2. What's tracked

| Experiment | Runs | Primary metric |
|---|---|---|
| `Good_Investment_Classification` | Baseline_MajorityClass, LogisticRegression, RandomForest, XGBoost, **XGBoost_Tuned** | `f1` |
| `Future_Price_Regression` | Baseline_Fixed8pct, LinearRegression, RandomForest, **XGBoost** | `rmse` |

| Registered model | Alias | Source run |
|---|---|---|
| `RealEstate_GoodInvestment_Classifier` | `@champion` | XGBoost_Tuned |
| `RealEstate_FuturePrice_Regressor` | `@champion` | XGBoost |

Each run logs:
- **Parameters:** every model hyperparameter, plus `n_features`
- **Metrics:** accuracy / precision / recall / f1 / roc_auc, or rmse / mae / r2, plus `train_time_s`
- **Tags:** `task`, `model_type`
- **Artifacts:** diagnostic plot (`<run>_diagnostic.png`) and the full sklearn pipeline (`model/`)

---

## 3. Navigate

### A. Experiments list
The left sidebar shows both experiments. Click **Good_Investment_Classification**.

### B. Runs table
- Each row is one model run. Use the **Columns** selector to show metrics like `f1`, `roc_auc` and `train_time_s`.
- Click a metric column header to **sort** (e.g. `f1` descending puts XGBoost_Tuned on top).
- Use the search bar to filter, e.g. `metrics.f1 > 0.9` or `tags.model_type = "XGBClassifier"`.

### C. Compare runs
1. Tick the checkboxes for LogisticRegression, RandomForest, XGBoost and XGBoost_Tuned.
2. Click **Compare**.
3. Useful views:
   - **Parallel coordinates:** how parameters relate to F1
   - **Scatter plot:** e.g. `train_time_s` vs `f1` (shows the speed/accuracy trade-off)
   - **Metrics table:** side-by-side numbers

### D. Inside a run
Click a run name (e.g. **XGBoost_Tuned**):
- **Overview:** parameters, metrics, tags, duration
- **Artifacts:**
  - `XGBoost_Tuned_diagnostic.png`: the confusion matrix
  - `model/`: `MLmodel` (flavour + signature), `requirements.txt`, `input_example.json`, and the pickled pipeline

### E. Model Registry
1. Open the **Models** page from the top navigation.
2. Click **RealEstate_GoodInvestment_Classifier**.
3. The versions list shows the `champion` alias on the active version. Click the version to see its source run and signature.
4. Repeat for **RealEstate_FuturePrice_Regressor**.

---

## 4. Load a registered model in code

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")          # run from project root

clf = mlflow.sklearn.load_model("models:/RealEstate_GoodInvestment_Classifier@champion")
reg = mlflow.sklearn.load_model("models:/RealEstate_FuturePrice_Regressor@champion")
```

The alias always resolves to the current best version, so no version number is hard-coded.

## 5. Export runs for the repo

`mlflow.db` and `mlruns/` are git-ignored, so export a summary that evaluators can read on GitHub:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(experiment_names=["Good_Investment_Classification", "Future_Price_Regression"])
cols = ["experiment_id", "tags.mlflow.runName"] + [c for c in runs.columns if c.startswith(("metrics.", "params."))]
runs[cols].to_csv("docs/mlflow_runs_summary.csv", index=False)
```

## 6. Screenshots for documentation

Save these to `docs/figures/mlflow/`:
1. Experiments list with both experiments
2. Classification runs table sorted by `f1`
3. Compare view (scatter of `train_time_s` vs `f1`)
4. XGBoost_Tuned run → confusion-matrix artifact
5. Models page showing both registered models with `@champion`

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| UI shows no experiments | Use `--backend-store-uri sqlite:///mlflow.db`, run from the project root |
| `Address already in use` | Another UI is running; use `--port 5001` or close the other terminal |
| Artifacts tab is empty | The `mlruns/` folder was moved or deleted; artifacts are stored there, not in the DB |
| Duplicate registered versions (v1, v2…) | Harmless; the `@champion` alias points to the correct one. Old versions can be deleted in the UI |