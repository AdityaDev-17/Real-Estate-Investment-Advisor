"""Runs both models and packages the result with a rule-based explanation."""
from dataclasses import dataclass

import pandas as pd

from core.features import FeatureBuilder
from settings import FORECAST_YEARS


@dataclass
class PredictionResult:
    """Everything the UI needs to display one prediction."""
    features: pd.Series
    probability: float
    current_price: float
    future_price: float
    criteria: dict

    @property
    def is_good(self) -> bool:
        return self.probability >= 0.5

    @property
    def confidence(self) -> float:
        """Confidence in the predicted class (not just P(good))."""
        return self.probability if self.is_good else 1 - self.probability

    @property
    def gain(self) -> float:
        return self.future_price - self.current_price

    @property
    def cagr(self) -> float:
        return (self.future_price / self.current_price) ** (1 / FORECAST_YEARS) - 1

    @property
    def criteria_met(self) -> int:
        return sum(self.criteria.values())


class InvestmentAdvisor:
    """Classifier → Good Investment probability · Regressor → 5-year price."""

    def __init__(self, clf, reg, builder: FeatureBuilder, data: pd.DataFrame):
        self.clf, self.reg, self.builder = clf, reg, builder
        # Market medians used by the explanation checklist (same as training criteria)
        self.median_rate = data["Appreciation_Rate"].median()
        self.median_infra = data["Infrastructure_Score"].median()

    def _criteria(self, f: pd.Series, cagr: float) -> dict:
        return {
            "Priced at/below city median (₹/sqft)": f["PPS_vs_City"] <= 1,
            f"Projected growth ≥ market median ({self.median_rate:.1%}/yr)": cagr >= self.median_rate,
            "Family size (BHK ≥ 3)": f["BHK"] >= 3,
            "Ready to move": f["Is_Ready_To_Move"] == 1,
            f"Infrastructure ≥ median ({self.median_infra:.1f}/10)": f["Infrastructure_Score"] >= self.median_infra,
            "Amenities ≥ 3": f["Amenity_Count"] >= 3,
        }

    def predict(self, inp: dict) -> PredictionResult:
        X = self.builder.build(inp)
        probability = float(self.clf.predict_proba(X)[0, 1])
        future_price = float(self.reg.predict(X)[0])
        current_price = inp["Price_in_Lakhs"]
        cagr = (future_price / current_price) ** (1 / FORECAST_YEARS) - 1

        features = X.iloc[0]
        return PredictionResult(
            features=features,
            probability=probability,
            current_price=current_price,
            future_price=future_price,
            criteria=self._criteria(features, cagr),
        )