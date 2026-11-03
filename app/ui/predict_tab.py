"""🔮 Predict tab — property form, verdict, forecast and explanation."""
import streamlit as st

from core.advisor import InvestmentAdvisor, PredictionResult
from settings import AMENITIES, FURNISH_MAP, GI_THRESHOLD, REFERENCE_YEAR, TRANSPORT_MAP
from ui.base import BaseTab
from ui.charts import ChartFactory


class PredictTab(BaseTab):
    label = "🔮 Predict"
    STATE_KEY = "prediction"

    def __init__(self, advisor: InvestmentAdvisor, meta: dict):
        self.advisor = advisor
        self.meta = meta
        self.opts = meta["options"]

    # ── input ──
    def _form(self) -> tuple[bool, dict]:
        with st.form("property_form"):
            st.subheader("🏠 Property details")
            c1, c2, c3 = st.columns(3)

            with c1:
                city = st.selectbox("City", self.opts["City"], index=self.opts["City"].index("Pune"))
                st.caption(f"State: **{self.meta['city_to_state'][city]}** · "
                           f"Tier **{self.meta['city_tier'][city]}**")
                ptype = st.selectbox("Property type", self.opts["Property_Type"])
                bhk = st.slider("BHK", 1, 5, 3)
                size = st.number_input("Size (sq ft)", 500, 5000, 1800, step=50)
                price = st.number_input("Current price (₹ Lakhs)", 10.0, 500.0, 150.0, step=5.0)

            with c2:
                year = st.slider("Year built", 1990, REFERENCE_YEAR, 2015)
                total_floors = st.slider("Total floors in building", 1, 30, 12)
                floor_no = st.slider("Floor number", 0, 30, 5)
                furnished = st.selectbox("Furnishing", list(FURNISH_MAP))
                facing = st.selectbox("Facing", self.opts["Facing"])
                owner = st.selectbox("Listed by", self.opts["Owner_Type"])

            with c3:
                schools = st.slider("Nearby schools", 1, 10, 6)
                hospitals = st.slider("Nearby hospitals", 1, 10, 5)
                transport = st.select_slider("Public transport access", list(TRANSPORT_MAP), value="Medium")
                amenities = st.multiselect("Amenities", AMENITIES, default=["Gym", "Garden", "Playground"])
                ready = st.radio("Availability", ["Ready to move", "Under construction"], horizontal=True)
                parking = st.checkbox("Parking available", value=True)
                security = st.checkbox("Security (gated / CCTV)", value=True)

            submitted = st.form_submit_button("🔮 Analyse investment", type="primary",
                                              use_container_width=True)

        if floor_no > total_floors:
            st.warning(f"Floor {floor_no} is above the building's {total_floors} floors — capped to "
                       f"{total_floors}, same rule used on the training data.")

        return submitted, {
            "City": city, "Property_Type": ptype, "BHK": bhk, "Size_in_SqFt": size,
            "Price_in_Lakhs": price, "Year_Built": year, "Floor_No": floor_no,
            "Total_Floors": total_floors, "Furnished": furnished, "Facing": facing,
            "Owner_Type": owner, "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
            "Transport": transport, "Amenities": amenities,
            "Ready_To_Move": ready == "Ready to move", "Parking": parking, "Security": security,
        }

    # ── output sections ──
    @staticmethod
    def _verdict(res: PredictionResult):
        conf = "≥ 99%" if res.confidence >= 0.99 else f"{res.confidence:.0%}"
        if res.is_good:
            st.success(f"### ✅ Good Investment — {conf} confidence")
        else:
            st.error(f"### ❌ Not a Good Investment — {conf} confidence")

    def _metrics(self, res: PredictionResult):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current price", f"₹{res.current_price:.1f} L")
        c2.metric("Estimated price in 5 years", f"₹{res.future_price:.1f} L", f"+₹{res.gain:.1f} L")
        c3.metric("Implied annual growth", f"{res.cagr:.2%}")
        c4.metric("Typical error (RMSE)", f"± ₹{self.meta['metrics']['regressor']['rmse']:.0f} L")

    @staticmethod
    def _charts(res: PredictionResult):
        g1, g2 = st.columns(2)
        g1.plotly_chart(ChartFactory.confidence_gauge(res.probability), use_container_width=True)
        g2.plotly_chart(ChartFactory.value_path(res.current_price, res.cagr), use_container_width=True)

    @staticmethod
    def _explanation(res: PredictionResult):
        st.subheader("🧾 Why? — Investment criteria check")
        for label, ok in res.criteria.items():
            st.markdown(f"{'✅' if ok else '⬜'} {label}")
        st.caption(f"{res.criteria_met}/6 criteria met (rule threshold: {GI_THRESHOLD}). The verdict comes "
                   "from the trained classifier; this checklist explains it. They can occasionally disagree "
                   "near the threshold because the model learned from noisy historical growth rates.")
        with st.expander("Engineered features sent to the model"):
            st.dataframe(res.features.astype(str).to_frame("value"), use_container_width=True)

    # ── entry point ──
    def render(self):
        submitted, inp = self._form()
        if submitted:
            st.session_state[self.STATE_KEY] = self.advisor.predict(inp)

        res = st.session_state.get(self.STATE_KEY)
        if res is None:
            st.info("Fill in the property details and click **Analyse investment**.")
            return

        st.divider()
        self._verdict(res)
        self._metrics(res)
        self._charts(res)
        self._explanation(res)