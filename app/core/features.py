"""Builds the exact 29-feature row the models were trained on (prevents training–serving skew)."""
import pandas as pd

from settings import AMENITIES, FURNISH_MAP, INFRA_WEIGHTS, REFERENCE_YEAR, TRANSPORT_MAP


class FeatureBuilder:
    """Raw form inputs (dict) → one-row DataFrame in training column order."""

    def __init__(self, meta: dict):
        self.city_tier = meta["city_tier"]
        self.city_median_pps = meta["city_median_pps"]
        self.columns = meta["cat_features"] + meta["num_features"]

    # ── individual feature rules (same formulas as notebooks 01–03) ──
    @staticmethod
    def age(year_built: int) -> int:
        return max(REFERENCE_YEAR - year_built, 0)

    @staticmethod
    def fix_floor(floor_no: int, total_floors: int) -> tuple[int, int]:
        """Cap floor at building height; flag if it had to be capped."""
        return min(floor_no, total_floors), int(floor_no > total_floors)

    @staticmethod
    def infrastructure_score(schools: int, hospitals: int, transport: str) -> float:
        school_n = (schools - 1) / 9
        hospital_n = (hospitals - 1) / 9
        transport_n = TRANSPORT_MAP[transport] / 2
        return 10 * (INFRA_WEIGHTS["school"] * school_n
                     + INFRA_WEIGHTS["hospital"] * hospital_n
                     + INFRA_WEIGHTS["transport"] * transport_n)

    def pps_vs_city(self, price_lakhs: float, size_sqft: int, city: str) -> float:
        pps = price_lakhs * 1e5 / size_sqft
        return round(pps / self.city_median_pps[city], 3)

    # ── assemble ──
    def build(self, inp: dict) -> pd.DataFrame:
        city = inp["City"]
        age = self.age(inp["Year_Built"])
        floor_no, floor_flag = self.fix_floor(inp["Floor_No"], inp["Total_Floors"])

        row = {
            "City": city,
            "Property_Type": inp["Property_Type"],
            "Facing": inp["Facing"],
            "Owner_Type": inp["Owner_Type"],
            "BHK": inp["BHK"],
            "Size_in_SqFt": inp["Size_in_SqFt"],
            "Price_in_Lakhs": inp["Price_in_Lakhs"],
            "Floor_No": floor_no,
            "Total_Floors": inp["Total_Floors"],
            "Age_of_Property": age,
            "Nearby_Schools": inp["Nearby_Schools"],
            "Nearby_Hospitals": inp["Nearby_Hospitals"],
            "Parking_Space": int(inp["Parking"]),
            "Security": int(inp["Security"]),
            "Floor_Inconsistent": floor_flag,
            "Is_Ready_To_Move": int(inp["Ready_To_Move"]),
            **{f"Has_{a}": int(a in inp["Amenities"]) for a in AMENITIES},
            "Amenity_Count": len(inp["Amenities"]),
            "Transport_Score": TRANSPORT_MAP[inp["Transport"]],
            "Furnish_Score": FURNISH_MAP[inp["Furnished"]],
            "Infrastructure_Score": self.infrastructure_score(
                inp["Nearby_Schools"], inp["Nearby_Hospitals"], inp["Transport"]),
            "City_Tier": self.city_tier[city],
            "Floor_Ratio": round(floor_no / inp["Total_Floors"], 3),
            "Is_New": int(age <= 5),
            "PPS_vs_City": self.pps_vs_city(inp["Price_in_Lakhs"], inp["Size_in_SqFt"], city),
        }
        return pd.DataFrame([row])[self.columns]