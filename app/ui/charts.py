"""All Plotly figures in one place — tabs only decide WHERE to show them."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from settings import FORECAST_YEARS, REFERENCE_YEAR, TIER_COLORS


class ChartFactory:

    @staticmethod
    def confidence_gauge(probability: float) -> go.Figure:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(probability * 100, 99.9),            # never display a false "100%"
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Good-investment probability"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#2ca02c" if probability >= 0.5 else "#d62728"},
                   "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
                   "steps": [{"range": [0, 50], "color": "#fbe3e3"},
                             {"range": [50, 100], "color": "#e3f5e3"}]},
        ))
        fig.update_layout(height=260, margin=dict(t=50, b=10, l=30, r=30))
        return fig

    @staticmethod
    def value_path(price: float, cagr: float) -> go.Figure:
        years = np.arange(0, FORECAST_YEARS + 1)
        fig = px.line(x=REFERENCE_YEAR + years, y=price * (1 + cagr) ** years, markers=True,
                      labels={"x": "Year", "y": "Estimated value (₹ Lakhs)"},
                      title=f"Projected value path ({cagr:.2%} / yr)")
        fig.update_layout(height=300, margin=dict(t=50, b=10))
        return fig

    @staticmethod
    def city_bhk_heatmap(data: pd.DataFrame) -> go.Figure:
        tier = data.groupby("City", observed=True)["City_Tier"].first().sort_values()
        heat = (data.pivot_table(index="City", columns="BHK", values="Good_Investment",
                                 aggfunc="mean", observed=True) * 100).reindex(tier.index)
        heat.index = [f"{c} (T{tier[c]})" for c in heat.index]
        fig = px.imshow(heat, text_auto=".0f", aspect="auto", color_continuous_scale="RdYlGn",
                        labels={"color": "% good"}, title="Good-investment rate (%) by City × BHK")
        fig.update_layout(height=900)
        return fig

    @staticmethod
    def growth_by_city(data: pd.DataFrame) -> go.Figure:
        g = (data.groupby("City", observed=True)
             .agg(rate=("Appreciation_Rate", "mean"), tier=("City_Tier", "first"))
             .sort_values("rate").reset_index())
        g["rate"] *= 100
        g["tier"] = "Tier " + g["tier"].astype(str)
        fig = px.bar(g, x="rate", y="City", color="tier", orientation="h",
                     color_discrete_map=TIER_COLORS,
                     labels={"rate": "Avg projected growth (% / yr)"}, title="Projected growth by city")
        fig.update_layout(height=800)
        return fig

    @staticmethod
    def tier_trend(data: pd.DataFrame) -> go.Figure:
        tier_rate = data.groupby("City_Tier")["Appreciation_Rate"].mean()
        base = data["Price_in_Lakhs"].median()
        trend = pd.DataFrame([
            {"Year": REFERENCE_YEAR + y, "Tier": f"Tier {t}", "Value": base * (1 + r) ** y}
            for t, r in tier_rate.items() for y in range(FORECAST_YEARS + 1)
        ])
        return px.line(trend, x="Year", y="Value", color="Tier", markers=True,
                       color_discrete_map=TIER_COLORS, labels={"Value": "Value (₹ Lakhs)"},
                       title=f"5-year trend for a median property (₹{base:.0f} L)")

    @staticmethod
    def feature_importance(importance: dict) -> go.Figure:
        imp = pd.Series(importance).sort_values()
        return px.bar(x=imp.values, y=imp.index, orientation="h",
                      labels={"x": "F1 drop when shuffled", "y": ""},
                      title="What drives the investment verdict")