# ruff: noqa: E402

import sys
from pathlib import Path

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

import mygene
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
shap_df = pd.read_csv(SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv")
loo_df = pd.read_csv(SCRIPT_DIR / "lgbm_iter_final_loo_summary.csv")

bases = [f.split(".")[0] for f in shap_df["feature"]]
mg = mygene.MyGeneInfo()
hits = mg.querymany(
    bases, scopes="ensembl.gene", fields="symbol", species="human", verbose=False
)
sym_map = {h["query"]: h.get("symbol", h["query"]) for h in hits}

loo_map = dict(zip(loo_df["feature"], loo_df["delta_auc"]))

print(f"{'Rank':<5} {'ENSG':<25} {'Symbol':<18} {'SHAP':>8}  {'LOO Δ':>8}")
print("-" * 70)
for _, row in shap_df.head(50).iterrows():
    base = row["feature"].split(".")[0]
    sym = sym_map.get(base, "?")
    delta = loo_map.get(row["feature"], float("nan"))
    print(
        f"{int(row['rank']):<5} {row['feature']:<25} {sym:<18} {row['mean_abs_shap']:>8.4f}  {delta:>+8.4f}"
    )

print("\n--- Original MLP panel genes ---")
panel = {
    "ENSG00000162654": "GBP4",
    "ENSG00000146070": "PLA2G7",
    "ENSG00000157426": "NADK2",
}
for base, sym in panel.items():
    match = shap_df[shap_df["feature"].str.startswith(base)]
    if not match.empty:
        r = match.iloc[0]
        d = loo_map.get(r["feature"], float("nan"))
        print(
            f"  {sym}: SHAP rank {int(r['rank'])}, SHAP={r['mean_abs_shap']:.4f}, LOO Δ={d:+.4f}"
        )
    else:
        print(f"  {sym}: NOT in LightGBM SFM feature set")
