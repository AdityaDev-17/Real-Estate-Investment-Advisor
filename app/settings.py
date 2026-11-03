"""Central configuration: paths and domain constants shared by every module."""
from pathlib import Path

# ── Paths (resolved from this file → works locally and on Streamlit Cloud) ──
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "processed" / "app_data.parquet"

CLF_PATH = MODEL_DIR / "good_investment_clf.joblib"
REG_PATH = MODEL_DIR / "future_price_reg.joblib"
META_PATH = MODEL_DIR / "app_metadata.json"

# ── Domain constants (must match the training notebooks exactly) ──
REFERENCE_YEAR = 2025                      # Age_of_Property = 2025 − Year_Built
AMENITIES = ["Gym", "Pool", "Clubhouse", "Garden", "Playground"]
INFRA_WEIGHTS = {"school": 0.3, "hospital": 0.3, "transport": 0.4}
TRANSPORT_MAP = {"Low": 0, "Medium": 1, "High": 2}
FURNISH_MAP = {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2}
GI_THRESHOLD = 4                           # criteria needed for "Good Investment"
FORECAST_YEARS = 5

# ── UI constants ──
PAGE_TITLE = "Real Estate Investment Advisor"
TIER_COLORS = {"Tier 1": "#1f77b4", "Tier 2": "#2ca02c", "Tier 3": "#ff7f0e"}
EXPLORE_COLS = ["City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
                "Price_per_SqFt", "Age_of_Property", "Infrastructure_Score", "Appreciation_Rate",
                "Future_Price_5Y", "Good_Investment"]