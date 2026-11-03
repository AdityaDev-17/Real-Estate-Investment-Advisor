"""🔍 Explore tab — filter the dataset by city, type, BHK, price and size."""
import pandas as pd
import streamlit as st

from settings import EXPLORE_COLS
from ui.base import BaseTab


class ExploreTab(BaseTab):
    label = "🔍 Explore"
    MAX_ROWS = 500

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def _filters(self) -> dict:
        st.subheader("🔍 Filter properties")
        c1, c2, c3 = st.columns(3)
        f = {
            "cities": c1.multiselect("City", sorted(self.data["City"].unique()), placeholder="All cities"),
            "types": c2.multiselect("Property type", sorted(self.data["Property_Type"].unique()),
                                    placeholder="All types"),
            "bhks": c3.multiselect("BHK", sorted(self.data["BHK"].unique()), placeholder="Any BHK"),
        }
        c4, c5, c6 = st.columns(3)
        f["price"] = c4.slider("Price (₹ Lakhs)", 10, 500, (10, 500))
        f["size"] = c5.slider("Size (sq ft)", 500, 5000, (500, 5000), step=100)
        f["good_only"] = c6.checkbox("Good investments only")
        return f

    def _apply(self, f: dict) -> pd.DataFrame:
        d = self.data
        mask = d["Price_in_Lakhs"].between(*f["price"]) & d["Size_in_SqFt"].between(*f["size"])
        if f["cities"]:
            mask &= d["City"].isin(f["cities"])
        if f["types"]:
            mask &= d["Property_Type"].isin(f["types"])
        if f["bhks"]:
            mask &= d["BHK"].isin(f["bhks"])
        if f["good_only"]:
            mask &= d["Good_Investment"] == 1
        return d[mask]

    @staticmethod
    def _kpis(sub: pd.DataFrame):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Properties", f"{len(sub):,}")
        m2.metric("Median price", f"₹{sub['Price_in_Lakhs'].median():.0f} L")
        m3.metric("Good-investment rate", f"{sub['Good_Investment'].mean():.1%}")
        m4.metric("Avg projected growth", f"{sub['Appreciation_Rate'].mean():.2%} / yr")

    def _table(self, sub: pd.DataFrame):
        show = sub.sort_values("Appreciation_Rate", ascending=False).head(self.MAX_ROWS)[EXPLORE_COLS].copy()
        show["Appreciation_Rate"] = (show["Appreciation_Rate"] * 100).round(2)
        show["Infrastructure_Score"] = show["Infrastructure_Score"].round(1)
        show["Good_Investment"] = show["Good_Investment"].map({1: "✅", 0: "—"})
        st.dataframe(show, use_container_width=True, hide_index=True, height=420)
        st.caption(f"Top {self.MAX_ROWS} by projected growth. Appreciation_Rate in % per year.")
        st.download_button("⬇️ Download filtered results (CSV)", sub.to_csv(index=False).encode(),
                           "filtered_properties.csv", "text/csv")

    def render(self):
        sub = self._apply(self._filters())
        if sub.empty:
            st.warning("No properties match these filters.")
            return
        self._kpis(sub)
        self._table(sub)