import numpy as np
import pandas as pd

def top_feature_importances(pipe, feature_meta, scenario=None, top_k=5):
    """
    Returns aggregated feature importances grouped by original feature names.
    Works whether the pipeline step name is 'prep' or 'preprocess'.
    """
    # find preprocess step
    named = getattr(pipe, "named_steps", {})
    if "prep" in named:
        prep = named["prep"]
    elif "preprocess" in named:
        prep = named["preprocess"]
    else:
        # assume first step
        prep = list(named.values()) if named else None

    model = pipe.named_steps.get("model", None) or pipe.steps[-1][1]

    # Get categorical names via OneHotEncoder if available
    cat_cols = feature_meta["categorical_cols"]
    num_cols = feature_meta["numeric_cols"]

    try:
        ohe = prep.named_transformers_["cat"]
        cat_names = ohe.get_feature_names_out(cat_cols)
    except Exception:
        # fallback: combine cat col names with levels if possible
        cat_names = []
        if "categorical_choices" in feature_meta:
            for c in cat_cols:
                levels = feature_meta["categorical_choices"].get(c, [])
                cat_names.extend([f"{c}_{lv}" for lv in levels])

    all_names = list(cat_names) + list(num_cols)

    # model feature importances (works for tree-based)
    try:
        importances = model.feature_importances_
        # assemble dataframe
        df = pd.DataFrame({"feature": all_names, "importance": importances})

        # group back to original feature names
        def base_name(s):
            for col in cat_cols:
                if s.startswith(col + "_"):
                    return col
            return s
        df["group"] = df["feature"].apply(base_name)
        agg = df.groupby("group", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
        return agg.head(top_k)
    except Exception:
        # if model doesn't have importances, return empty
        return pd.DataFrame(columns=["group", "importance"])
