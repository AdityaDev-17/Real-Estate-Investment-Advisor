"""📊 Insights tab — market charts and model performance."""
import pandas as pd
import streamlit as st

from ui.base import BaseTab
from ui.charts import ChartFactory


class InsightsTab(BaseTab):
    label = "📊 Insights"

    def __init__(self, data: pd.DataFrame, meta: dict):
        self.data = data
        self.meta = meta

    def _market_charts(self):
        st.subheader("📊 Market insights")
        st.plotly_chart(ChartFactory.city_bhk_heatmap(self.data), use_container_width=True)

        c1, c2 = st.columns(2)
        c1.plotly_chart(ChartFactory.growth_by_city(self.data), use_container_width=True)
        c2.plotly_chart(ChartFactory.tier_trend(self.data), use_container_width=True)
        c2.plotly_chart(ChartFactory.feature_importance(self.meta["permutation_importance"]),
                        use_container_width=True)

    def _model_performance(self):
        st.subheader("🎯 Model performance (held-out test set, 50,000 properties)")
        mc = self.meta["metrics"]["classifier"]
        mr = self.meta["metrics"]["regressor"]
        k = st.columns(6)
        k[0].metric("Accuracy", f"{mc['accuracy']:.1%}")
        k[1].metric("F1-score", f"{mc['f1']:.3f}")
        k[2].metric("ROC-AUC", f"{mc['roc_auc']:.3f}")
        k[3].metric("RMSE", f"₹{mr['rmse']:.1f} L")
        k[4].metric("MAE", f"₹{mr['mae']:.1f} L")
        k[5].metric("R²", f"{mr['r2']:.3f}")
        st.caption(f"Regression RMSE is within ~2% of the irreducible noise floor "
                   f"(≈ ₹{self.meta['metrics']['regression_noise_floor']} L). Targets are rule-based "
                   "(the dataset has no historical sale prices) — see project documentation.")

    def render(self):
        self._market_charts()
        self._model_performance()