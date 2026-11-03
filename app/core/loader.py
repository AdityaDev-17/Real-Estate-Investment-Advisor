"""Loads models, metadata and data once, then serves them from Streamlit's cache."""
import json

import joblib
import pandas as pd
import streamlit as st

from settings import CLF_PATH, DATA_PATH, META_PATH, REG_PATH


class ArtifactLoader:
    """All file I/O lives here — the rest of the app never touches disk."""

    @staticmethod
    def _check_exists(*paths):
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            st.error("Missing required files:\n\n" + "\n".join(f"- `{m}`" for m in missing)
                     + "\n\nRun notebook `04_model_training.ipynb` (Cells 15 & 17) first.")
            st.stop()

    @staticmethod
    @st.cache_resource(show_spinner="Loading models…")
    def load_models():
        """cache_resource → one shared copy of the models for all users/reruns."""
        ArtifactLoader._check_exists(CLF_PATH, REG_PATH)
        return joblib.load(CLF_PATH), joblib.load(REG_PATH)

    @staticmethod
    @st.cache_resource
    def load_metadata() -> dict:
        ArtifactLoader._check_exists(META_PATH)
        with open(META_PATH) as f:
            return json.load(f)

    @staticmethod
    @st.cache_data(show_spinner="Loading property data…")
    def load_data() -> pd.DataFrame:
        """cache_data → returns a copy each call, so filtering can't mutate the cache."""
        ArtifactLoader._check_exists(DATA_PATH)
        return pd.read_parquet(DATA_PATH)