"""Adaptive panel external validation.

For each external cohort, missing panel genes are substituted with the best
non-artifact replacement present in that cohort (by symbol matching).

Cohorts:
  GSE76220  — lumbar spinal cord LCM, n=20, symbol-keyed RPKM
  GSE122649 — motor cortex postmortem, n=38, symbol-keyed raw counts

For each cohort two conditions are compared:
  (A) Subset panel  : panel genes natively matched in the cohort
  (B) Adapted panel : (A) + clean substitutes for each missing panel gene

Clean pool: gene_replacement_results.csv with 41 confirmed artifacts excluded
(per shap500_artifact_audit.csv; threshold: max GTEx TPM < 1.0).

Outputs
-------
  adaptive_panel_validation.png
  adaptive_panel_validation_statistics.txt
"""

from __future__ import annotations

import gzip
import json
import re
import sys
import tarfile
import time
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

import matplotlib  # noqa: E402
from utils import load_dataset  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SCRIPT_DIR = Path(__file__).parent
ALS_DIR_RES = ALS_DIR / "resources"

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_REPL_CSV = SCRIPT_DIR / "gene_replacement_results.csv"
_AUDIT_CSV = SCRIPT_DIR / "shap500_artifact_audit.csv"

RANDOM_STATE = 42
N_BOOTSTRAP = 2_000

_GSE122649_RAW_TAR = ALS_DIR_RES / "GSE122649" / "GSE122649_RAW.tar"
_SOFT_URL_122649 = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122649/soft/"
    "GSE122649_family.soft.gz"
)
_RAW_URL_122649 = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122649/suppl/"
    "GSE122649_RAW.tar"
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _bootstrap_auc(
    y: np.ndarray,
    scores: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    arr = np.array(aucs)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _loo_predict(
    X_log: np.ndarray,
    y: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Within-cohort LOO — new StandardScaler per fold."""
    loo_params = {**params, "min_child_samples": 1, "verbose": -1, "n_jobs": 1}
    scores = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_log[tr]).astype(np.float32)
        X_te = sc.transform(X_log[[i]]).astype(np.float32)
        clf = LGBMClassifier(**loo_params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(X_tr, y[tr])
        scores[i] = clf.predict_proba(X_te)[0, 1]
    return scores


# ---------------------------------------------------------------------------
# Replacement lookup
# ---------------------------------------------------------------------------


# Canonical artifact gene families (aligned with cross_cohort_substitution.py
# and druggable_replacement_targets.py). Symbol-prefix patterns for
# non-CNS / quantification-artifact families that must never serve as
# biological surrogates. The GTEx TPM<1 audit alone misses these (e.g. the
# keratin-associated protein KRTAP6-2 has skin expression > 1 TPM).
_ARTIFACT_FAMILY_RE = [
    re.compile(p)
    for p in (
        r"^OR\d",
        r"^TAS",
        r"^PATE",
        r"^DEFA",
        r"^DEFB",
        r"^KRTAP",
        r"^KRT\d",
        r"^LCE",
        r"^SPRR",
        r"^VN1R",
    )
]


def _is_family_artifact(symbol: str) -> bool:
    """True if a gene symbol matches a known artifact gene family prefix."""
    return any(p.match(symbol) for p in _ARTIFACT_FAMILY_RE)


def _build_replacement_lookup(
    repl_csv: Path,
    audit_csv: Path,
) -> dict[str, list[tuple[float, float, str]]]:
    """Return {panel_symbol → [(abs_r, auc, cand_ensg_base), ...] descending |r|, auc}.

    Sort priority: |pearson_r| descending (co-expression criterion first),
    then replacement_auc descending as tiebreaker. Candidates with no Pearson r
    (SHAP-only source) receive abs_r = -1.0 and are ranked after correlation
    candidates regardless of AUC. Excludes confirmed artifacts (GTEx TPM<1)
    and known artifact gene families by symbol (olfactory/taste/keratin/...).
    """
    repl = pd.read_csv(repl_csv)
    audit = pd.read_csv(audit_csv)
    artifact_bases = set(
        audit.loc[audit["classification"] == "confirmed_artifact", "ensg_base"]
    )
    # Also exclude artifact gene families by symbol (e.g. keratin KRTAP6-2),
    # which the GTEx TPM<1 audit criterion does not flag.
    sym_map = dict(zip(audit["ensg_base"].astype(str), audit["symbol"].astype(str)))
    artifact_bases |= {b for b, s in sym_map.items() if _is_family_artifact(str(s))}
    repl["candidate_base"] = repl["candidate_ensg"].str.split(".").str[0]
    clean = repl[
        repl["is_replacement"] & ~repl["candidate_base"].isin(artifact_bases)
    ].copy()
    # abs_r: correlation candidates get |r|; SHAP-only get -1 (rank last)
    clean["abs_r"] = clean["pearson_r"].abs().fillna(-1.0)

    lookup: dict[str, list[tuple[float, float, str]]] = {}
    for sym, grp in clean.groupby("panel_symbol"):
        grp_s = grp.sort_values(["abs_r", "replacement_auc"], ascending=False)
        lookup[str(sym)] = list(
            zip(
                grp_s["abs_r"].tolist(),
                grp_s["replacement_auc"].tolist(),
                grp_s["candidate_base"].tolist(),
            )
        )
    return lookup


def _build_ensg_sym_map(audit_csv: Path) -> dict[str, str]:
    """ENSG base → HGNC symbol from the artifact audit CSV."""
    audit = pd.read_csv(audit_csv)
    result: dict[str, str] = {}
    for _, row in audit.iterrows():
        sym = str(row.get("symbol", "") or "").strip()
        if sym and sym != "nan":
            result[str(row["ensg_base"])] = sym
    return result


def _resolve_mygeneinfo(ensg_bases: list[str]) -> dict[str, str]:
    """Batch-query MyGeneInfo for ENSG base → symbol."""
    if not ensg_bases:
        return {}
    print(f"  MyGeneInfo lookup for {len(ensg_bases)} unresolved ENSGs ...")
    url = "https://mygene.info/v3/gene"
    result: dict[str, str] = {}
    for i in range(0, len(ensg_bases), 500):
        batch = ensg_bases[i : i + 500]
        try:
            r = requests.post(
                url,
                json={"ids": batch, "fields": "symbol", "species": "human"},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            r.raise_for_status()
            for item in r.json():
                sym = item.get("symbol", "")
                qid = item.get("query", "")
                if sym and qid:
                    result[qid] = sym
        except Exception as exc:
            print(f"  WARNING: MyGeneInfo batch failed: {exc}")
        time.sleep(0.25)
    return result


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_train() -> tuple[np.ndarray, list[str], np.ndarray]:
    """GPL24676 log1p RSEM — (X_log1p, ensg_col_names, y)."""
    (ds,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=ALS_DIR_RES)
    X = np.log1p(ds.X.values.astype(np.float32))
    return X, list(ds.X.columns), ds.y.values.astype(int)


def _load_gse76220() -> tuple[np.ndarray, list[str], np.ndarray]:
    """GSE76220 log1p RPKM — (X_log1p, gene_symbols, y)."""
    (ds,) = load_dataset("GSE76220", resources_dir=ALS_DIR_RES)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        X = np.log1p(ds.X.values.astype(np.float32))
    return X, list(ds.X.columns), ds.y.values.astype(int)


def _download_if_needed(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name} ...")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    print(f"  Saved {dest.stat().st_size / 1e6:.0f} MB → {dest}")


def _parse_122649_soft() -> dict[str, str]:
    r = requests.get(_SOFT_URL_122649, timeout=60, stream=True)
    content = gzip.decompress(r.content).decode("utf-8", errors="ignore")
    samples = re.split(r"\^SAMPLE", content)[1:]
    meta: dict[str, str] = {}
    for s in samples:
        acc_m = re.search(r"!Sample_geo_accession = (GSM\d+)", s)
        diag_m = re.search(r"!Sample_characteristics_ch1 = diagnosis: (.+)", s)
        if acc_m and diag_m:
            meta[acc_m.group(1)] = diag_m.group(1).strip()
    return meta


def _load_gse122649() -> tuple[np.ndarray, list[str], np.ndarray]:
    """GSE122649 log1p raw counts — (X_log1p, gene_symbols, y)."""
    _download_if_needed(_RAW_URL_122649, _GSE122649_RAW_TAR)
    gsm_to_diag = _parse_122649_soft()

    gene_symbols: list[str] = []
    sample_counts: dict[str, np.ndarray] = {}

    with tarfile.open(_GSE122649_RAW_TAR, "r") as tf:
        for member in tf.getmembers():
            fname = member.name
            gsm_m = re.search(r"(GSM\d+)", fname)
            if not gsm_m or gsm_m.group(1) not in gsm_to_diag:
                continue
            gsm = gsm_m.group(1)
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            raw = fobj.read()
            if fname.endswith(".gz"):
                raw = gzip.decompress(raw)
            lines = raw.decode("utf-8", errors="ignore").splitlines()
            rows = []
            for line in lines[1:]:
                line = line.strip()
                if not line or line.startswith("__"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                try:
                    count = float(parts[1])
                except ValueError:
                    continue
                rows.append((parts[0].strip('"').strip("'"), count))
            if not rows:
                continue
            syms = [r[0] for r in rows]
            counts = np.array([r[1] for r in rows], dtype=np.float32)
            if not gene_symbols:
                gene_symbols = syms
            elif len(syms) != len(gene_symbols):
                continue
            sample_counts[gsm] = counts

    gsm_ids = list(sample_counts.keys())
    X_raw = np.stack([sample_counts[g] for g in gsm_ids], axis=0)
    diag_map = {"sALS": 1, "sALS/FTD": 1, "Non-neurological control": 0}
    y = np.array([diag_map.get(gsm_to_diag[g], -1) for g in gsm_ids])
    valid = y >= 0
    X_log = np.log1p(X_raw[valid].astype(np.float32))
    return X_log, gene_symbols, y[valid]


# ---------------------------------------------------------------------------
# Match + adapt
# ---------------------------------------------------------------------------


def _match_and_adapt(
    panel_df: pd.DataFrame,
    cohort_syms: list[str],
    replacement_lookup: dict[str, list[tuple[float, str]]],
    ensg_sym_map: dict[str, str],
    train_ensg: list[str],
    cohort_name: str,
) -> dict:
    """Identify matched panel genes and best clean substitutes for missing ones.

    Returns a dict with keys:
      matched     : list of entry-dicts with tr_col + te_col
      missing     : list of missing panel gene symbols
      substitutes : list of entry-dicts for substitutes
      no_sub      : list of symbols with no substitute found
    """
    cohort_sym_idx: dict[str, int] = {s: i for i, s in enumerate(cohort_syms)}
    train_ensg_base_idx: dict[str, int] = {
        e.split(".")[0]: i for i, e in enumerate(train_ensg)
    }
    train_ensg_full_idx: dict[str, int] = {e: i for i, e in enumerate(train_ensg)}

    matched: list[dict] = []
    missing_syms: list[str] = []

    for _, row in panel_df.iterrows():
        sym = str(row["symbol"])
        feat = str(row["feature"])
        feat_base = feat.split(".")[0]
        tr_col = train_ensg_full_idx.get(feat, train_ensg_base_idx.get(feat_base))
        if tr_col is None:
            continue
        te_col = cohort_sym_idx.get(sym)
        if te_col is not None:
            matched.append({"symbol": sym, "tr_col": tr_col, "te_col": te_col})
        else:
            missing_syms.append(sym)

    print(f"\n  {cohort_name}: {len(matched)}/25 panel genes matched natively")
    print(f"  Matched  : {', '.join(m['symbol'] for m in matched)}")
    if missing_syms:
        print(f"  Missing  : {', '.join(missing_syms)}")

    substitutes: list[dict] = []
    no_sub: list[str] = []
    used_bases: set[str] = set()  # de-duplication: each replacement used at most once

    for sym in missing_syms:
        if sym not in replacement_lookup:
            no_sub.append(sym)
            continue
        found = False
        for abs_r, auc_repl, cand_base in replacement_lookup[sym]:
            if cand_base in used_bases:
                continue  # skip already-assigned replacement
            cand_sym = ensg_sym_map.get(cand_base)
            if not cand_sym:
                continue
            te_col = cohort_sym_idx.get(cand_sym)
            if te_col is None:
                continue
            tr_col = train_ensg_base_idx.get(cand_base)
            if tr_col is None:
                continue
            substitutes.append(
                {
                    "replaces": sym,
                    "symbol": cand_sym,
                    "ensg_base": cand_base,
                    "tr_col": tr_col,
                    "te_col": te_col,
                    "replacement_auc": auc_repl,
                    "pearson_r_abs": abs_r if abs_r >= 0.0 else None,
                }
            )
            used_bases.add(cand_base)
            r_str = f"  |r|={abs_r:.3f}" if abs_r >= 0.0 else ""
            print(
                f"    Sub {sym} → {cand_sym} ({cand_base})  "
                f"replacement_AUC={auc_repl:.4f}{r_str}"
            )
            found = True
            break
        if not found:
            no_sub.append(sym)

    if no_sub:
        print(f"  No substitute: {', '.join(no_sub)}")

    return {
        "matched": matched,
        "missing": missing_syms,
        "substitutes": substitutes,
        "no_sub": no_sub,
    }


# ---------------------------------------------------------------------------
# Run one cohort
# ---------------------------------------------------------------------------


def _run_cohort(
    X_train_log: np.ndarray,
    y_train: np.ndarray,
    X_cohort_log: np.ndarray,
    y_cohort: np.ndarray,
    match_info: dict,
    params: dict,
    rng: np.random.Generator,
    cohort_name: str,
) -> dict:
    """Zero-shot + within-cohort LOO for subset and adapted panels."""
    matched = match_info["matched"]
    substitutes = match_info["substitutes"]
    results: dict = {"cohort": cohort_name, "y": y_cohort}

    conditions: list[tuple[str, list[dict]]] = [
        ("subset", matched),
        ("adapted", matched + substitutes),
    ]

    for label, entries in conditions:
        if not entries:
            results[label] = {
                "auc": float("nan"),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "loo_auc": float("nan"),
                "n_genes": 0,
                "scores": np.array([]),
                "loo_scores": np.array([]),
            }
            continue

        X_tr_log = np.column_stack([X_train_log[:, e["tr_col"]] for e in entries])
        X_te_log = np.column_stack([X_cohort_log[:, e["te_col"]] for e in entries])
        n_genes = X_tr_log.shape[1]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_log).astype(np.float32)
        X_te_sc = scaler.transform(X_te_log).astype(np.float32)

        # Zero-shot: train on GPL24676, predict on cohort
        clf = LGBMClassifier(**{**params, "verbose": -1})
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(X_tr_sc, y_train)
        scores = clf.predict_proba(X_te_sc)[:, 1]
        auc = float(roc_auc_score(y_cohort, scores))
        ci_lo, ci_hi = _bootstrap_auc(y_cohort, scores, N_BOOTSTRAP, rng)

        # Within-cohort LOO (each sample held out once; cohort-only scaler)
        loo_scores = _loo_predict(X_te_log, y_cohort, params)
        loo_auc = float(roc_auc_score(y_cohort, loo_scores))

        results[label] = {
            "auc": auc,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "loo_auc": loo_auc,
            "n_genes": n_genes,
            "scores": scores,
            "loo_scores": loo_scores,
        }
        print(
            f"    [{label:7}] {n_genes:2d} genes  "
            f"zero-shot={auc:.4f} [{ci_lo:.4f},{ci_hi:.4f}]  "
            f"LOO={loo_auc:.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _make_figure(results76220: dict, results122649: dict) -> None:
    cohort_info = [
        (results76220, "GSE76220 — Lumbar SC LCM\n(n=20, ALS=12, Ctrl=8)"),
        (results122649, "GSE122649 — Motor Cortex\n(n=38, ALS=26, Ctrl=12)"),
    ]
    colours = {"subset": "#888888", "adapted": "#1A6FBF"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for row, (res, cohort_title) in enumerate(cohort_info):
        y = res["y"]
        for col, cond in enumerate(["subset", "adapted"]):
            ax = axes[row][col]
            r = res[cond]
            if np.isnan(r["auc"]) or len(r["scores"]) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            colour = colours[cond]
            fpr, tpr, _ = roc_curve(y, r["scores"])
            ax.plot(
                fpr,
                tpr,
                lw=2,
                color=colour,
                ls="-",
                label=(
                    f"Zero-shot AUC={r['auc']:.4f}\n"
                    f"95% CI [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]"
                ),
            )
            fpr_l, tpr_l, _ = roc_curve(y, r["loo_scores"])
            ax.plot(
                fpr_l,
                tpr_l,
                lw=2,
                color=colour,
                ls="--",
                label=f"Within-cohort LOO={r['loo_auc']:.4f}",
            )
            ax.plot([0, 1], [0, 1], "k--", lw=0.7)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            cond_label = "Subset panel" if cond == "subset" else "Adapted panel"
            ax.set_title(
                f"{cohort_title}\n{cond_label} ({r['n_genes']} genes)",
                fontsize=9,
            )
            ax.legend(fontsize=7.5, loc="lower right")
            ax.grid(alpha=0.25)

    fig.suptitle(
        "Adaptive Panel External Validation\n"
        "Solid = zero-shot (train GPL24676 n=874)  ·  Dashed = within-cohort LOO",
        fontsize=10,
    )
    plt.tight_layout()
    out = SCRIPT_DIR / "adaptive_panel_validation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {out.name}")


# ---------------------------------------------------------------------------
# Statistics file
# ---------------------------------------------------------------------------


def _write_stats(
    results76220: dict,
    results122649: dict,
    match76220: dict,
    match122649: dict,
) -> None:
    lines = [
        "Adaptive Panel External Validation",
        "=" * 72,
        "",
        "Strategy: missing panel genes replaced with best non-artifact",
        "replacement from gene_replacement_results.csv (clean pool).",
        "Artifacts excluded: 41 confirmed (max GTEx TPM < 1.0).",
        "",
    ]

    for res, minfo, title in [
        (results76220, match76220, "GSE76220 — Lumbar SC LCM (n=20, ALS=12, Ctrl=8)"),
        (
            results122649,
            match122649,
            "GSE122649 — Motor Cortex (n=38, ALS=26, Ctrl=12)",
        ),
    ]:
        matched_syms = [m["symbol"] for m in minfo["matched"]]
        lines += [
            title,
            "-" * 72,
            f"Panel genes matched natively: {len(matched_syms)}/25",
            f"  {', '.join(matched_syms)}",
            f"Missing panel genes: {len(minfo['missing'])}",
            f"  {', '.join(minfo['missing']) if minfo['missing'] else 'none'}",
            f"Substitutes found: {len(minfo['substitutes'])}",
        ]
        for s in minfo["substitutes"]:
            r_str = (
                f"  |r|={s['pearson_r_abs']:.3f}"
                if s.get("pearson_r_abs") is not None
                else ""
            )
            lines.append(
                f"  {s['replaces']:<18} → {s['symbol']:<14} "
                f"({s['ensg_base']})  repl_AUC={s['replacement_auc']:.4f}{r_str}"
            )
        lines.append(
            f"No substitute available: "
            f"{', '.join(minfo['no_sub']) if minfo['no_sub'] else 'none'}"
        )
        lines.append("")

        for cond in ["subset", "adapted"]:
            r = res[cond]
            n = r["n_genes"]
            cond_label = "Subset panel" if cond == "subset" else "Adapted panel"
            if np.isnan(r["auc"]):
                lines.append(f"  {cond_label}: insufficient data")
                continue
            lines += [
                f"  {cond_label} ({n} genes):",
                f"    Zero-shot AUC    : {r['auc']:.4f}",
                f"    Bootstrap 95% CI : [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
                f"  (n={N_BOOTSTRAP})",
                f"    Within-cohort LOO: {r['loo_auc']:.4f}",
            ]

        if not np.isnan(res["subset"]["auc"]) and not np.isnan(res["adapted"]["auc"]):
            delta_zs = res["adapted"]["auc"] - res["subset"]["auc"]
            delta_loo = res["adapted"]["loo_auc"] - res["subset"]["loo_auc"]
            lines.append(
                f"\n  Delta (adapted − subset): "
                f"zero-shot {delta_zs:+.4f}  LOO {delta_loo:+.4f}"
            )
        lines.append("")

    out = SCRIPT_DIR / "adaptive_panel_validation_statistics.txt"
    out.write_text("\n".join(lines))
    print(f"Statistics saved: {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("Adaptive Panel External Validation")
    print("=" * 72)

    panel_df = pd.read_csv(_PANEL_CSV)
    params = json.loads(_PARAMS_PATH.read_text())
    params["colsample_bytree"] = 1.0
    rng = np.random.default_rng(RANDOM_STATE)

    # Build clean replacement lookup
    print("\n[1] Building clean replacement lookup ...")
    replacement_lookup = _build_replacement_lookup(_REPL_CSV, _AUDIT_CSV)
    n_entries = sum(len(v) for v in replacement_lookup.values())
    print(
        f"  {len(replacement_lookup)} panel genes, {n_entries} clean candidate entries"
    )

    # Build ENSG → symbol map
    print("\n[2] Building ENSG→symbol map ...")
    ensg_sym_map = _build_ensg_sym_map(_AUDIT_CSV)
    print(f"  {len(ensg_sym_map)} symbols from audit CSV")

    # Resolve any candidates not in the audit CSV
    all_cand_bases = {
        base for cands in replacement_lookup.values() for _, _, base in cands
    }
    unresolved = [b for b in all_cand_bases if b not in ensg_sym_map]
    if unresolved:
        extra = _resolve_mygeneinfo(unresolved)
        ensg_sym_map.update(extra)
        print(f"  +{len(extra)} symbols from MyGeneInfo")
    resolved_total = sum(1 for b in all_cand_bases if b in ensg_sym_map)
    print(
        f"  {resolved_total}/{len(all_cand_bases)} replacement candidates have a symbol"
    )

    # Load GPL24676 training data
    print("\n[3] Loading GPL24676 training data ...")
    X_train_log, train_ensg, y_train = _load_train()
    print(
        f"  n={len(y_train)}  ALS={int(y_train.sum())}  Ctrl={int((y_train == 0).sum())}"
    )

    # GSE76220
    print("\n[4] Loading GSE76220 ...")
    X_76220, syms_76220, y_76220 = _load_gse76220()
    print(
        f"  n={len(y_76220)}  ALS={int(y_76220.sum())}  Ctrl={int((y_76220 == 0).sum())}"
    )

    match76220 = _match_and_adapt(
        panel_df,
        syms_76220,
        replacement_lookup,
        ensg_sym_map,
        train_ensg,
        "GSE76220",
    )

    print("\n  Running GSE76220 validation ...")
    results76220 = _run_cohort(
        X_train_log,
        y_train,
        X_76220,
        y_76220,
        match76220,
        params,
        rng,
        "GSE76220",
    )

    # GSE122649
    print("\n[5] Loading GSE122649 ...")
    X_122649, syms_122649, y_122649 = _load_gse122649()
    print(
        f"  n={len(y_122649)}  ALS={int(y_122649.sum())}  "
        f"Ctrl={int((y_122649 == 0).sum())}"
    )

    match122649 = _match_and_adapt(
        panel_df,
        syms_122649,
        replacement_lookup,
        ensg_sym_map,
        train_ensg,
        "GSE122649",
    )

    print("\n  Running GSE122649 validation ...")
    results122649 = _run_cohort(
        X_train_log,
        y_train,
        X_122649,
        y_122649,
        match122649,
        params,
        rng,
        "GSE122649",
    )

    # Figure and statistics
    print("\n[6] Generating figure ...")
    _make_figure(results76220, results122649)
    _write_stats(results76220, results122649, match76220, match122649)

    print("\nDone.")


if __name__ == "__main__":
    main()
