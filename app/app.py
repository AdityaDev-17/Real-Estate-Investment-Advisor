"""
Real Estate Investment Advisor — Streamlit entry point
Run from project root:  streamlit run app/app.py
"""
import streamlit as st

from core import ArtifactLoader, FeatureBuilder, InvestmentAdvisor
from settings import PAGE_TITLE
from ui import ExploreTab, InsightsTab, PredictTab


class RealEstateApp:
    """Wires loaders → core logic → tabs. No business logic lives here."""

    def __init__(self):
        st.set_page_config(page_title=PAGE_TITLE, page_icon="🏠", layout="wide")

        clf, reg = ArtifactLoader.load_models()
        meta = ArtifactLoader.load_metadata()
        data = ArtifactLoader.load_data()

        advisor = InvestmentAdvisor(clf, reg, FeatureBuilder(meta), data)
        self.tabs = [PredictTab(advisor, meta), ExploreTab(data), InsightsTab(data, meta)]

    def run(self):
        st.title(f"🏠 {PAGE_TITLE}")
        st.caption("Predicts whether a property is a good investment and estimates its value in 5 years.")

        for container, tab in zip(st.tabs([t.label for t in self.tabs]), self.tabs):
            with container:
                tab.render()


if __name__ == "__main__":
    RealEstateApp().run()