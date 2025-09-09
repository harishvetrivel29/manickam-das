# train_model_compat.py (project root)

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

base = Path(__file__).resolve().parent
csv_path = base / "app" / "data" / "synthetic_lca_dataset.csv"

# Load
data = pd.read_csv(csv_path)

target_cols = [
    "total_emissions_kgCO2e_per_t",
    "total_energy_MJ_per_t",
    "total_waste_kg_per_t",
]
categorical_cols = [
    "mining_method",
    "sorting_method",
    "concentration_method",
    "smelting_method",
    "energy_source",
    "product_type",
]
numeric_cols = [
    "ore_grade",
    "transport_km",
    "manufacturing_efficiency",
    "recycling_rate",
]

def has_expected_schema(df: pd.DataFrame) -> bool:
    need = set(categorical_cols + numeric_cols + target_cols)
    return need.issubset(set(df.columns))

def build_expected_from_legacy(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["product_type"] = df["material"].str.lower().map(
        {"steel": "steel", "aluminium": "aluminum", "aluminum": "aluminum", "copper": "copper", "plastic": "steel"}
    ).fillna("steel")
    proc = df["process"].str.lower().fillna("casting")
    out["mining_method"] = np.where(proc.isin(["rolling", "extrusion"]), "open_pit", "underground")
    out["sorting_method"] = np.where(proc.isin(["rolling", "injection molding"]), "sensor_based", "manual")
    out["concentration_method"] = np.where(proc.isin(["casting", "rolling"]), "magnetic", "flotation")
    out["smelting_method"] = np.where(proc.isin(["extrusion", "rolling"]), "electric_arc", "blast_furnace")

    eol = df["end_of_life"].str.lower().fillna("landfill")
    out["energy_source"] = np.select(
        [eol.eq("recycled"), eol.eq("incinerated"), eol.eq("landfill")],
        ["renewable", "gas", "grid_mix"],
        default="coal",
    )

    energy = df["energy_use"].astype(float)
    water = df["water_use"].astype(float)
    out["ore_grade"] = np.clip((1000 / (1 + energy / 50 + water / 50)), 0.1, 100.0)
    out["transport_km"] = df["transport_km"].astype(float)

    eff_raw = 100 - 0.05 * (energy + 0.5 * water)
    out["manufacturing_efficiency"] = np.clip(eff_raw, 50.0, 100.0)

    impact = df.get("impact_score", pd.Series(50.0, index=df.index)).astype(float)
    base_rr = np.where(eol.eq("recycled"), 60.0, np.where(eol.eq("incinerated"), 20.0, 10.0))
    out["recycling_rate"] = np.clip(base_rr + (50 - impact) * 0.3, 0.0, 100.0)

    emissions = df["emissions"].astype(float)
    out["total_energy_MJ_per_t"] = (energy * 30 + water * 5 + 1000).astype(float)
    out["total_emissions_kgCO2e_per_t"] = (emissions * 10 + energy * 5 + out["transport_km"] * 0.5 + 200).astype(float)
    waste = df["waste"].astype(float)
    out["total_waste_kg_per_t"] = (waste * 10 + water * 2 + 50).astype(float)
    return out[categorical_cols + numeric_cols + target_cols]

if not has_expected_schema(data):
    data = build_expected_from_legacy(data)

X = data.drop(columns=target_cols)
y = data[target_cols]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
pipe = Pipeline([("prep", preprocess), ("model", rf)])
pipe.fit(X, y)

out = base / "models"
out.mkdir(exist_ok=True)
joblib.dump(pipe, out / "lca_model.pkl")

feature_columns = {
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols,
    "target_cols": target_cols,
    "categorical_choices": {
        "mining_method": ["open_pit", "underground"],
        "sorting_method": ["manual", "sensor_based"],
        "concentration_method": ["flotation", "magnetic", "gravity"],
        "smelting_method": ["blast_furnace", "electric_arc", "hydromet"],
        "energy_source": ["coal", "gas", "renewable", "grid_mix"],
        "product_type": ["steel", "aluminum", "copper"],
    }
}
with open(out / "feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)

app_data_dir = base / "app" / "data"
app_data_dir.mkdir(parents=True, exist_ok=True)
data.head().to_csv(app_data_dir / "dataset_sample_head.csv", index=False)

report = {
    "created_at": pd.Timestamp.utcnow().isoformat() + "Z",
    "n_rows": int(len(data)),
    "features": {"categorical": categorical_cols, "numeric": numeric_cols},
    "targets": target_cols,
    "estimator": "RandomForestRegressor (multi-output) inside Pipeline",
    "notes": "Derived from legacy CSV schema by compatible trainer.",
}
rep_out = base / "reports"
rep_out.mkdir(exist_ok=True)
with open(rep_out / "training_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("✅ Model retrained (compatible schema), artifacts saved.")
