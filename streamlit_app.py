import json
from pathlib import Path
import math
import pandas as pd
import streamlit as st
import plotly.express as px
from utils import (
    load_resources, predict_impacts, suggest_alternatives,
    format_results, generate_pdf_report, to_percent_change,
    compute_efficiency_score, stage_contributions, pareto_front,
)
from explainer import top_feature_importances

# =========================
# Page Config and Styling
# =========================
st.set_page_config(page_title="LCA Decision Studio", page_icon="🌍", layout="wide")

# Subtle CSS polish
st.markdown("""
<style>
/* Tighter section spacing */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
/* KPI card numbers */
[data-testid="stMetricValue"] { font-weight: 700; }
/* Hide fullscreen button on charts (cleaner UI) */
button[title="View fullscreen"] { display: none; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def _load():
    return load_resources()

pipe, feature_meta, dataset_head, training_report = _load()

# =========================
# Sidebar — Scenario Wizard
# =========================
with st.sidebar:
    st.title("🌍 LCA Studio")
    st.caption("Build, compare and optimize scenarios with AI-powered insights.")
    st.markdown("---")

    st.subheader("⛏️ Mining")
    mining_method = st.selectbox("Method", feature_meta["categorical_choices"]["mining_method"])
    sorting_method = st.selectbox("Sorting", feature_meta["categorical_choices"]["sorting_method"])

    st.subheader("⚙️ Processing")
    concentration_method = st.selectbox("Concentration", feature_meta["categorical_choices"]["concentration_method"])
    smelting_method = st.selectbox("Smelting", feature_meta["categorical_choices"]["smelting_method"])

    st.subheader("⚡ Energy")
    energy_source = st.selectbox("Source", feature_meta["categorical_choices"]["energy_source"])

    st.subheader("🏭 Product")
    product_type = st.selectbox("Material", feature_meta["categorical_choices"]["product_type"])

    st.subheader("🚚 Logistics")
    ore_grade = st.number_input("Ore grade (%)", min_value=0.1, max_value=100.0, value=40.0, step=0.1)
    transport_km = st.number_input("Transport (km)", min_value=0.0, max_value=10000.0, value=500.0, step=10.0)

    st.subheader("♻️ Circularity")
    manufacturing_efficiency = st.slider("Manufacturing efficiency (%)", 50.0, 100.0, 90.0, 0.1)
    recycling_rate = st.slider("Recycling rate (%)", 0.0, 100.0, 30.0, 0.1)

    st.markdown("---")
    run_analysis = st.button("🔮 Predict & Analyze", type="primary")
    st.caption("Tip: After predicting, use the tabs to compare, rank, and explore Pareto insights.")

# Initialize history
if "history" not in st.session_state:
    st.session_state["history"] = []

def add_history(record):
    st.session_state["history"].append(record)

scenario = dict(
    mining_method=mining_method,
    sorting_method=sorting_method,
    concentration_method=concentration_method,
    smelting_method=smelting_method,
    energy_source=energy_source,
    product_type=product_type,
    ore_grade=float(ore_grade),
    transport_km=float(transport_km),
    manufacturing_efficiency=float(manufacturing_efficiency),
    recycling_rate=float(recycling_rate),
)

# =========================
# Header and Tabs
# =========================
st.title("✨ LCA Decision Studio — Interactive")
st.caption("Ordered, visual, and action-focused analysis for emissions, energy, and waste.")

tab_build, tab_compare, tab_history, tab_decision, tab_about = st.tabs(
    ["🧪 Build Scenario", "🆚 Compare", "🗂️ History", "🧭 Decision Studio", "ℹ️ About"]
)

# ---------------------------
# 🧪 Build Scenario
# ---------------------------
with tab_build:
    st.subheader("Scenario Summary")
    st.json(scenario)

    if run_analysis:
        preds = predict_impacts(pipe, feature_meta, scenario)
        record = {**scenario, **preds}
        score = compute_efficiency_score(preds)
        record["composite_efficiency_score"] = score
        add_history(record)

        # KPI strip
        st.markdown("### 📊 KPI Overview")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        cards = format_results(preds)
        with c1: st.metric("🌫️ Emissions", f"{preds['total_emissions_kgCO2e_per_t']:,.1f} kgCO₂e/t")
        with c2: st.metric("⚡ Energy", f"{preds['total_energy_MJ_per_t']:,.1f} MJ/t")
        with c3: st.metric("🗑️ Waste", f"{preds['total_waste_kg_per_t']:,.1f} kg/t")
        with c4: st.metric("🧮 Composite (↓ better)", f"{score:,.1f}")

        # Impact chart
        chart_df = pd.DataFrame({
            "Metric": ["Emissions (kgCO₂e/t)", "Energy (MJ/t)", "Waste (kg/t)"],
            "Value": [
                preds["total_emissions_kgCO2e_per_t"],
                preds["total_energy_MJ_per_t"],
                preds["total_waste_kg_per_t"],
            ],
        })
        fig = px.bar(chart_df, x="Metric", y="Value", title="Impact Breakdown 🧭")
        st.plotly_chart(fig, use_container_width=True)

        # Insights expanders
        with st.expander("💡 What-if Alternatives (quick wins)", expanded=True):
            alt_df = suggest_alternatives(pipe, feature_meta, scenario)
            st.dataframe(alt_df, use_container_width=True)

        with st.expander("🧠 Model Drivers (top features)", expanded=False):
            imp_df = top_feature_importances(pipe, feature_meta, scenario, top_k=8)
            st.dataframe(imp_df, use_container_width=True)

        with st.expander("🔍 Stage Contributions (sensitivity)", expanded=False):
            contrib_df = stage_contributions(pipe, feature_meta, scenario)
            st.dataframe(contrib_df, use_container_width=True)

        # Downloads
        st.markdown("### ⬇️ Export")
        csv_bytes = pd.DataFrame([record]).to_csv(index=False).encode()
        st.download_button("Download Scenario CSV", data=csv_bytes, file_name="scenario.csv", mime="text/csv")

        try:
            pdf_bytes = generate_pdf_report(scenario, preds, alt_df, training_report, extra_contrib=contrib_df)
            st.download_button("Download Scenario PDF", data=pdf_bytes, file_name="scenario_report.pdf", mime="application/pdf")
        except Exception:
            st.info("PDF export requires 'reportlab'. Try: pip install reportlab")

    else:
        st.info("Use the sidebar to set parameters, then click “🔮 Predict & Analyze”.")

# ---------------------------
# 🆚 Compare
# ---------------------------
with tab_compare:
    st.subheader("Interactive Scenario Comparison")
    st.caption("Select any historical scenario to compare against the most recent inputs.")

    if not st.session_state["history"]:
        st.warning("No scenarios yet. Build one in the 🧪 Build tab.")
    else:
        hist_df = pd.DataFrame(st.session_state["history"])
        colA, colB = st.columns([1, 2])
        with colA:
            choice_idx = st.selectbox(
                "Scenario B (from history)",
                options=list(range(len(hist_df))),
                format_func=lambda i: f"{i}: {hist_df.loc[i,'product_type']} | score {hist_df.loc[i,'composite_efficiency_score']:.1f}"
            )
        B = hist_df.loc[choice_idx].to_dict()
        A = st.session_state["history"][-1]

        A_pred = {k: A[k] for k in A if k.startswith("total_")}
        B_pred = {k: B[k] for k in B if k.startswith("total_")}

        cmp_mode = st.radio("Display mode", ["Absolute", "% change"], horizontal=True)

        df_cmp = pd.DataFrame([
            {"Metric": "Emissions (kgCO₂e/t)", "A": A_pred["total_emissions_kgCO2e_per_t"], "B": B_pred["total_emissions_kgCO2e_per_t"]},
            {"Metric": "Energy (MJ/t)", "A": A_pred["total_energy_MJ_per_t"], "B": B_pred["total_energy_MJ_per_t"]},
            {"Metric": "Waste (kg/t)", "A": A_pred["total_waste_kg_per_t"], "B": B_pred["total_waste_kg_per_t"]},
        ])
        if cmp_mode == "Absolute":
            df_cmp["Δ (B-A)"] = df_cmp["B"] - df_cmp["A"]
        else:
            df_cmp["% change (A→B)"] = df_cmp.apply(lambda r: to_percent_change(r["A"], r["B"]), axis=1)

        def winner_label(a, b):
            if b < a: return "B better"
            if b > a: return "A better"
            return "Equal"
        df_cmp["Winner"] = df_cmp.apply(lambda r: winner_label(r["A"], r["B"]), axis=1)

        st.dataframe(df_cmp, use_container_width=True)
        plot_df = df_cmp.melt(id_vars=["Metric"], value_vars=["A", "B"], var_name="Scenario", value_name="Value")
        fig = px.bar(plot_df, x="Metric", y="Value", color="Scenario", barmode="group", title="A vs B Comparison 🆚")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# 🗂️ History
# ---------------------------
with tab_history:
    st.subheader("Scenario History & Ranking")
    if not st.session_state["history"]:
        st.info("No scenarios stored yet. Run 'Predict & Analyze' first.")
    else:
        hist_df = pd.DataFrame(st.session_state["history"])
        st.markdown("#### 📈 Ranked (lower composite score is better)")
        st.dataframe(hist_df.sort_values("composite_efficiency_score").reset_index(drop=True), use_container_width=True)
        st.download_button("Download history CSV", data=hist_df.to_csv(index=False).encode(), file_name="lca_history.csv", mime="text/csv")
        best_idx = hist_df["composite_efficiency_score"].idxmin()
        st.success("🏆 Best scenario so far")
        st.json(hist_df.loc[best_idx].to_dict())

# ---------------------------
# 🧭 Decision Studio
# ---------------------------
with tab_decision:
    st.subheader("Pareto Front & Decision Insights")
    if not st.session_state["history"]:
        st.info("No scenarios in history to analyze.")
    else:
        hist_df = pd.DataFrame(st.session_state["history"])
        metrics = ["total_emissions_kgCO2e_per_t", "total_energy_MJ_per_t", "total_waste_kg_per_t"]
        pareto_idx = pareto_front(hist_df, metrics)
        pf = hist_df.loc[pareto_idx].reset_index(drop=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### 🎯 Pareto-optimal scenarios")
            st.dataframe(pf, use_container_width=True)
            st.download_button("Download Pareto CSV", data=pf.to_csv(index=False).encode(), file_name="pareto_scenarios.csv", mime="text/csv")
        with c2:
            st.markdown("#### 🧾 Quick Recommendations")
            if not pf.empty:
                rec = {cat: pf[cat].mode().iloc[0] for cat in feature_meta["categorical_cols"]}
                st.json(rec)
                st.caption("Choose the most frequent stage options in the Pareto set to reduce impacts.")

# ---------------------------
# ℹ️ About
# ---------------------------
with tab_about:
    st.subheader("About & Notes")
    st.write("""
- Scenarios are kept in memory for this session. For persistence, connect to a DB or local CSV.
- Composite weights: Emissions 0.5, Energy 0.3, Waste 0.2 (see utils.py).
- PDF export requires: pip install reportlab
    """)
    st.markdown("#### Training report")
    st.json(training_report)
