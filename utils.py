import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

def load_resources():
    """Load pipeline, feature metadata, sample dataset head, and training report.
    Expects project structure:
    - project_root/
        - models/lca_model.pkl
        - models/feature_columns.json
        - reports/training_report.json
        - app/
            - utils.py (this file)
            - data/dataset_sample_head.csv
    """
    app_dir = Path(__file__).resolve().parent
    root = app_dir.parent

    # Model
    pkl = root / "models" / "lca_model.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"Model not found at {pkl}. Retrain first or place model there.")
    pipe = joblib.load(pkl)

    # Feature metadata
    with open(root / "models" / "feature_columns.json", "r") as f:
        feature_meta = json.load(f)

    # Sample dataset head (optional)
    dataset_head_path = app_dir / "data" / "dataset_sample_head.csv"
    dataset_head = pd.read_csv(dataset_head_path) if dataset_head_path.exists() else pd.DataFrame()

    # Training report
    report_path = root / "reports" / "training_report.json"
    with open(report_path, "r") as f:
        training_report = json.load(f)

    return pipe, feature_meta, dataset_head, training_report

def _to_df_row(feature_meta, scenario: dict):
    cols = feature_meta["categorical_cols"] + feature_meta["numeric_cols"]
    row = {c: scenario.get(c, None) for c in cols}
    return pd.DataFrame([row])

def predict_impacts(pipe, feature_meta, scenario: dict):
    X = _to_df_row(feature_meta, scenario)
    pred = pipe.predict(X)[0]
    targets = feature_meta["target_cols"]
    return dict(zip(targets, map(float, pred)))

def format_results(preds: dict):
    return [
        ("Emissions", preds["total_emissions_kgCO2e_per_t"], "kgCO₂e / t"),
        ("Energy", preds["total_energy_MJ_per_t"], "MJ / t"),
        ("Waste", preds["total_waste_kg_per_t"], "kg / t"),
    ]

def suggest_alternatives(pipe, feature_meta, scenario: dict):
    cat_choices = feature_meta["categorical_choices"]
    base = predict_impacts(pipe, feature_meta, scenario)
    base_em = base["total_emissions_kgCO2e_per_t"]
    rows = []
    for feat, choices in cat_choices.items():
        if feat not in feature_meta["categorical_cols"]:
            continue
        for choice in choices:
            if choice == scenario.get(feat):
                continue
            cand = scenario.copy()
            cand[feat] = choice
            pred = predict_impacts(pipe, feature_meta, cand)
            delta = pred["total_emissions_kgCO2e_per_t"] - base_em
            rows.append({
                "Stage": feat, "Option": choice,
                "Emissions_if_applied (kgCO2e/t)": pred["total_emissions_kgCO2e_per_t"],
                "Δ vs base (kgCO2e/t)": delta
            })
    df = pd.DataFrame(rows).sort_values(by="Δ vs base (kgCO2e/t)").reset_index(drop=True)
    if df.empty:
        return df
    return df.groupby("Stage", as_index=False).first()

def to_percent_change(a, b):
    if a == 0:
        return "—"
    return f"{(b - a) / a * 100:.1f}%"

# Composite efficiency score (lower is better). Tweak weights here.
def compute_efficiency_score(preds, weights=(0.5, 0.3, 0.2)):
    w_em, w_en, w_w = weights
    em = preds["total_emissions_kgCO2e_per_t"]
    en = preds["total_energy_MJ_per_t"]
    wa = preds["total_waste_kg_per_t"]
    # Normalize by some rough scale to keep score numeric-stable
    score = w_em * em + w_en * (en / 10.0) + w_w * (wa * 2.0)
    return score

# Stage contributions via sensitivity analysis
def stage_contributions(pipe, feature_meta, scenario: dict, pct_change=0.1):
    base_preds = predict_impacts(pipe, feature_meta, scenario)
    metrics = ["total_emissions_kgCO2e_per_t", "total_energy_MJ_per_t", "total_waste_kg_per_t"]
    rows = []

    # Categorical: try each alternative and measure delta
    for feat in feature_meta["categorical_cols"]:
        base_val = scenario.get(feat)
        deltas = {m: 0.0 for m in metrics}
        for choice in feature_meta["categorical_choices"].get(feat, []):
            if choice == base_val:
                continue
            cand = scenario.copy()
            cand[feat] = choice
            pred = predict_impacts(pipe, feature_meta, cand)
            for m in metrics:
                deltas[m] += (pred[m] - base_preds[m])
        n_choices = max(1, len(feature_meta["categorical_choices"].get(feat, [])) - 1)
        avg_deltas = {m: deltas[m] / n_choices for m in metrics}
        rows.append({
            "Stage": feat,
            "Δ_emissions_avg(kgCO2e/t)": avg_deltas["total_emissions_kgCO2e_per_t"],
            "Δ_energy_avg(MJ/t)": avg_deltas["total_energy_MJ_per_t"],
            "Δ_waste_avg(kg/t)": avg_deltas["total_waste_kg_per_t"],
        })

    # Numeric features: perturb +/- pct_change and compute sensitivity (slope)
    for feat in feature_meta["numeric_cols"]:
        base_val = scenario.get(feat, 0)
        up = scenario.copy(); up[feat] = max(base_val * (1 + pct_change), base_val + 1e-6)
        down = scenario.copy(); down[feat] = max(base_val * (1 - pct_change), base_val - 1e-6)
        p_up = predict_impacts(pipe, feature_meta, up)
        p_down = predict_impacts(pipe, feature_meta, down)
        denom = (up[feat] - down[feat]) if (up[feat] - down[feat]) != 0 else 1.0
        deriv = {m: (p_up[m] - p_down[m]) / denom for m in metrics}
        rows.append({
            "Stage": feat,
            "Δ_emissions_per_unit": deriv["total_emissions_kgCO2e_per_t"],
            "Δ_energy_per_unit": deriv["total_energy_MJ_per_t"],
            "Δ_waste_per_unit": deriv["total_waste_kg_per_t"],
        })

    return pd.DataFrame(rows)

# Pareto front (non-dominated) detection
def pareto_front(df, metric_cols):
    # lower is better for all metrics
    points = df[metric_cols].values
    n = points.shape[0]
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                dominated[i] = True
                break
    return [i for i, d in enumerate(dominated) if not d]

# PDF export (enhanced with contributions)
def generate_pdf_report(inputs: dict, preds: dict, alternatives_df, training_report: dict, extra_contrib=None) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    x, y = 2*cm, height - 2*cm

    def line(text, dy=0.6*cm, size=10):
        nonlocal y
        c.setFont("Helvetica", size)
        c.drawString(x, y, str(text))
        y -= dy
        if y < 2*cm:
            c.showPage()
            y = height - 2*cm

    c.setTitle("LCA Scenario Report")
    line("LCA Scenario Report", size=16, dy=1*cm)
    line(f"Created: {datetime.utcnow().isoformat()}Z", size=9)

    line("Inputs:", dy=0.8*cm)
    for k, v in inputs.items():
        line(f" - {k}: {v}", size=9)
    line("")

    line("Predicted Impacts:", dy=0.8*cm)
    line(f" - Emissions: {preds['total_emissions_kgCO2e_per_t']:.1f} kgCO2e/t")
    line(f" - Energy: {preds['total_energy_MJ_per_t']:.1f} MJ/t")
    line(f" - Waste: {preds['total_waste_kg_per_t']:.1f} kg/t")
    line("")

    if isinstance(alternatives_df, pd.DataFrame) and not alternatives_df.empty:
        line("Top Alternatives:", dy=0.8*cm)
        for _, r in alternatives_df.iterrows():
            line(f" - {r['Stage']}: {r['Option']} → {r['Emissions_if_applied (kgCO2e/t)']:.1f} kgCO2e/t (Δ {r['Δ vs base (kgCO2e/t)']:.1f})", size=9)

    if extra_contrib is not None and isinstance(extra_contrib, pd.DataFrame) and not extra_contrib.empty:
        line("", dy=0.6*cm)
        line("Stage Contributions (summary):", dy=0.8*cm)
        for _, r in extra_contrib.head(10).iterrows():
            emis_cols = [c for c in extra_contrib.columns if 'emissions' in c]
            if emis_cols:
                val = r[emis_cols[0]]
                line(f" - {r['Stage']}: emissions Δ ~ {val:.2f}", size=9)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()
