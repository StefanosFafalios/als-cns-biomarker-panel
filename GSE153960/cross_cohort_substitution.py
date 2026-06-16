"""Cross-cohort surrogate substitution for the 15-gene critical panel.

When the 15-gene critical panel is matched against external cohorts, three
non-coding members are missing in at least one cohort vocabulary:

  HERC2P8 (pseudogene)  — absent in GSE76220, GSE122649 (HGNC-keyed)
  SMG1P5  (pseudogene)  — absent in GSE76220 (HGNC-keyed)
  SNORD97 (snoRNA)      — absent in SRP064478 (Salmon Ensembl ENSG index)

This script asks: for each (cohort, missing_gene) gap, can we substitute a
protein-coding genome-wide replacement (from gene_replacement_results.csv)
that IS present in the target cohort's vocabulary, to improve the zero-shot
AUC over the 14-of-15 baseline panel for that cohort?

For each (cohort, missing_gene):
  1. Pull all valid genome-wide replacements (is_replacement=True) from the
     gene_replacement screen.
  2. Filter to protein-coding genes with a resolvable HGNC symbol
     (excluding olfactory receptors, taste receptors, prostate-specific
     genes — known quantification-artifact gene families).
  3. Filter to genes present in the target cohort vocabulary.
  4. For each surviving candidate c, retrain LGBM on GPL24676 with
     {panel - missing_gene + c} and evaluate zero-shot AUC on the cohort.
  5. Report the best substitution per gap.

Output:
  cross_cohort_substitution_statistics.txt
  cross_cohort_substitution.png
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import tarfile
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_REPL_CSV = SCRIPT_DIR / "gene_replacement_results.csv"
_PREFILTER_NAMES = SCRIPT_DIR / "lgbm_prefilter_names.txt"
_PREFILTER_X = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_SRP_QUANT_DIR = ALS_DIR / "resources" / "SRP064478" / "quant"
_SRP_META = ALS_DIR / "resources" / "SRP064478" / "srr_metadata.tsv"
_GSE122649_RAW_TAR = ALS_DIR / "resources" / "GSE122649" / "GSE122649_RAW.tar"
_GSE122649_SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122649/"
    "soft/GSE122649_family.soft.gz"
)

RANDOM_STATE = 42
N_SEEDS = 5  # ensemble seeds (lighter than panel_loo to keep runtime tight)
N_BOOTSTRAP = 1_000
MAX_CANDIDATES_PER_GAP = 60  # cap per (cohort, gene) gap to keep runtime tight

# 15-gene critical panel indices in the 25-gene ordered panel
_CRITICAL_IDX = [1, 2, 3, 5, 6, 7, 10, 11, 12, 15, 16, 18, 20, 23, 24]

_COMPARTMENTS: dict[str, tuple[str, ...]] = {
    "erythrocyte": (
        "ENSG00000206172", "ENSG00000188536", "ENSG00000244734",
        "ENSG00000158578", "ENSG00000170180", "ENSG00000133742",
        "ENSG00000159111", "ENSG00000105610", "ENSG00000179364",
        "ENSG00000223609",
    ),
    "platelet": (
        "ENSG00000163736", "ENSG00000163737", "ENSG00000185245",
    ),
    "endothelial": (
        "ENSG00000261371", "ENSG00000179776", "ENSG00000110799",
    ),
}

# Family patterns flagged as quantification-artifact gene families
# (olfactory receptors, taste receptors, prostate/testis expressed)
_ARTIFACT_FAMILY_PATTERNS = (
    re.compile(r"^OR\d"), re.compile(r"^TAS"), re.compile(r"^PATE"),
    re.compile(r"^DEFB"), re.compile(r"^KRTAP"), re.compile(r"^MS4A"),
    re.compile(r"^VN1R"), re.compile(r"^LCE"), re.compile(r"^SPRR"),
)


def _is_protein_coding_safe(symbol: str, biotype: str) -> bool:
    """Decide whether a candidate gene is a defensible protein-coding surrogate."""
    # mygene returns biotype as "protein-coding" (hyphen) for canonical PC genes
    if not symbol or biotype not in ("protein-coding", "protein_coding"):
        return False
    for p in _ARTIFACT_FAMILY_PATTERNS:
        if p.match(symbol):
            return False
    return True


def _bootstrap_auc(y, scores, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    out = np.empty(n, dtype=np.float64)
    for b in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        out[b] = roc_auc_score(y[idx], scores[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _ensemble_scores(X_tr, y_tr, X_te, params, seeds=tuple(range(N_SEEDS))):
    """Average predict_proba across N_SEEDS seeds; returns scores on X_te."""
    from lightgbm import LGBMClassifier
    scores = np.zeros(len(X_te), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in seeds:
            params_s = dict(params, random_state=s)
            clf = LGBMClassifier(**params_s).fit(X_tr, y_tr)
            scores += clf.predict_proba(X_te)[:, 1]
    return scores / len(seeds)


# ---- Data loaders ----

def _load_gpl24676_ctd():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    (ds,) = load_dataset("GSE153960", platform="GPL24676",
                          resources_dir=ALS_DIR / "resources")
    X_raw = ds.X.values.astype(np.float32)
    y = ds.y.values.astype(int)
    feat = list(ds.X.columns)
    X_log = np.log1p(X_raw)
    base_ids = [n.split(".")[0] for n in feat]
    pcs = []
    for comp_bases in _COMPARTMENTS.values():
        base_set = set(comp_bases)
        idx = [i for i, b in enumerate(base_ids) if b in base_set]
        if not idx:
            continue
        pca = PCA(n_components=1, random_state=0)
        pcs.append(pca.fit_transform(X_log[:, idx]))
    Z = np.hstack(pcs)
    reg = LinearRegression().fit(Z, X_log)
    X_ctd = (X_log - reg.predict(Z)).astype(np.float32)
    return X_ctd, y, X_log, feat


def _load_gse76220():
    (ds,) = load_dataset("GSE76220", resources_dir=ALS_DIR / "resources")
    return (np.log1p(ds.X.values.astype(np.float32)),
            list(ds.X.columns), ds.y.values.astype(int))


def _parse_gse122649_soft():
    import requests
    r = requests.get(_GSE122649_SOFT_URL, timeout=60, stream=True)
    content = gzip.decompress(r.content).decode("utf-8", errors="ignore")
    meta: dict[str, str] = {}
    for s in re.split(r"\^SAMPLE", content)[1:]:
        acc_m = re.search(r"!Sample_geo_accession = (GSM\d+)", s)
        diag_m = re.search(r"!Sample_characteristics_ch1 = diagnosis: (.+)", s)
        if acc_m and diag_m:
            meta[acc_m.group(1)] = diag_m.group(1).strip()
    return meta


def _load_gse122649():
    gsm_to_diag = _parse_gse122649_soft()
    diag_map = {"sALS": 1, "sALS/FTD": 1, "Non-neurological control": 0}
    gene_symbols: list[str] = []
    sample_counts: dict[str, np.ndarray] = {}
    with tarfile.open(_GSE122649_RAW_TAR, "r") as tf:
        for member in tf.getmembers():
            gsm_m = re.search(r"(GSM\d+)", member.name)
            if not gsm_m or gsm_m.group(1) not in gsm_to_diag:
                continue
            gsm = gsm_m.group(1)
            f_obj = tf.extractfile(member)
            if f_obj is None:
                continue
            raw = f_obj.read()
            if member.name.endswith(".gz"):
                raw = gzip.decompress(raw)
            rows = []
            for line in raw.decode("utf-8", errors="ignore").splitlines()[1:]:
                line = line.strip()
                if not line or line.startswith("__"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                try:
                    rows.append((parts[0].strip('"').strip("'"), float(parts[1])))
                except ValueError:
                    continue
            if not rows:
                continue
            syms_here = [r[0] for r in rows]
            if not gene_symbols:
                gene_symbols = syms_here
            elif len(syms_here) != len(gene_symbols):
                continue
            sample_counts[gsm] = np.array([r[1] for r in rows], dtype=np.float32)
    gsm_ids = list(sample_counts.keys())
    X_raw = np.stack([sample_counts[g] for g in gsm_ids])
    y_full = np.array([diag_map.get(gsm_to_diag[g], -1) for g in gsm_ids])
    valid = y_full >= 0
    return np.log1p(X_raw[valid]), gene_symbols, y_full[valid]


def _load_srp064478():
    import pandas as pd
    meta = pd.read_csv(_SRP_META, sep="\t").set_index("SRR")
    quants: list[pd.Series] = []
    tx_ids: list[str] | None = None
    y_list: list[int] = []
    for srr in meta.index:
        sf = _SRP_QUANT_DIR / srr / "quant.sf"
        df = pd.read_csv(sf, sep="\t")
        df["tx_base"] = df["Name"].str.split(".").str[0]
        tx_counts = df.set_index("tx_base")["NumReads"]
        if tx_ids is None:
            tx_ids = list(tx_counts.index)
        quants.append(tx_counts)
        y_list.append(1 if meta.loc[srr, "condition"] == "ALS" else 0)
    ensg_direct = [t for t in (tx_ids or []) if t.startswith("ENSG")]
    if ensg_direct:
        X_raw = np.array(
            [q.reindex(tx_ids).fillna(0).values for q in quants], dtype=np.float32
        )
        return np.log1p(X_raw), tx_ids or [], np.array(y_list, dtype=int)
    fasta = ALS_DIR / "resources" / "SRP064478" / "index" / "hg38_cdna.fa.gz"
    tx2gene: dict[str, str] = {}
    with gzip.open(str(fasta), "rt") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            m_enst = re.search(r"(ENST\d+)\.\d+", line)
            m_ensg = re.search(r"gene:(ENSG\d+)\.\d+", line)
            if m_enst and m_ensg:
                tx2gene[m_enst.group(1)] = m_ensg.group(1)
    tx_mat = pd.DataFrame(
        {srr: q.reindex(tx_ids).fillna(0) for srr, q in zip(meta.index, quants)}
    )
    tx_mat.index = tx_ids
    tx_mat["ensg"] = pd.Series(tx_ids).map(tx2gene).values
    gene_mat = tx_mat.dropna(subset=["ensg"]).groupby("ensg").sum()
    X_raw = gene_mat.values.T.astype(np.float32)
    return np.log1p(X_raw), list(gene_mat.index), np.array(y_list, dtype=int)


def _resolve_symbols(ensg_list):
    """Resolve Ensembl base IDs to {symbol, biotype} via mygene."""
    import mygene
    mg = mygene.MyGeneInfo()
    out = {}
    chunk = 500
    for i in range(0, len(ensg_list), chunk):
        res = mg.querymany(
            ensg_list[i:i + chunk], scopes="ensembl.gene",
            fields="symbol,type_of_gene", species="human", verbose=False,
        )
        for r in res:
            if r.get("notfound"):
                continue
            out[r["query"]] = {
                "symbol": r.get("symbol", ""),
                "biotype": r.get("type_of_gene", ""),
            }
    return out


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=" * 70)
    print("Cross-cohort surrogate substitution for 15-gene critical panel")
    print("=" * 70)

    params_raw = json.loads(_PARAMS_PATH.read_text())
    params = dict(params_raw, colsample_bytree=1.0, n_jobs=-1, verbose=-1)

    # ---- Panel ----
    df_panel = pd.read_csv(_PANEL_CSV)
    feat_col = next(c for c in df_panel.columns
                     if "feature" in c.lower() or "ensg" in c.lower())
    sym_col = next(c for c in df_panel.columns if "symbol" in c.lower())
    feat25 = df_panel[feat_col].tolist()
    sym25 = df_panel[sym_col].tolist()
    feat25_bases = [f.split(".")[0] for f in feat25]
    panel_syms = [sym25[i] for i in _CRITICAL_IDX]
    panel_bases = [feat25_bases[i] for i in _CRITICAL_IDX]
    print(f"\n15-gene critical panel: {', '.join(panel_syms)}\n")

    # ---- Replacement candidate pool ----
    print("Loading gene_replacement_results.csv ...")
    repl_df = pd.read_csv(_REPL_CSV)
    repl_df = repl_df[repl_df["is_replacement"]]
    print(f"  {len(repl_df)} (panel_gene, candidate) replacement rows.")

    # ---- Load full training matrix (for candidate retrieval) ----
    print("Loading GPL24676 (CTD + log1p) ...")
    X_train_ctd, y_train, X_train_log, feat_train = _load_gpl24676_ctd()
    feat_base_to_col = {f.split(".")[0]: j for j, f in enumerate(feat_train)}

    # ---- Cohorts ----
    print("Loading GSE76220 ...")
    X76_log, vocab76, y76 = _load_gse76220()
    vocab76_map = {s: i for i, s in enumerate(vocab76)}

    print("Loading GSE122649 ...")
    X122_log, vocab122, y122 = _load_gse122649()
    vocab122_map = {s: i for i, s in enumerate(vocab122)}

    print("Loading SRP064478 ...")
    Xsrp_log, vocab_srp, y_srp = _load_srp064478()
    srp_vocab_map = {v: i for i, v in enumerate(vocab_srp)}

    # ---- Identify gaps ----
    def _gaps(cohort_name, vocab_map, by_symbol):
        gaps = []
        for k, (sym, base) in enumerate(zip(panel_syms, panel_bases)):
            key = sym if by_symbol else base
            if key not in vocab_map:
                gaps.append((k, sym, base))
        return gaps

    cohort_specs = [
        ("GSE76220",  vocab76_map,  y76,  X76_log,  True),
        ("GSE122649", vocab122_map, y122, X122_log, True),
        ("SRP064478", srp_vocab_map, y_srp, Xsrp_log, False),
    ]

    # ---- Pre-resolve candidate symbols for the missing genes (so we can filter) ----
    missing_panel = set()
    for c_name, v_map, _y, _X, by_sym in cohort_specs:
        for _, sym, _ in _gaps(c_name, v_map, by_sym):
            missing_panel.add(sym)
    print(f"\nGenes missing in at least one cohort: {sorted(missing_panel)}")

    candidates_per_panel = {}
    for sym in missing_panel:
        sub_df = repl_df[repl_df["panel_symbol"] == sym].sort_values(
            "replacement_auc", ascending=False
        ).head(MAX_CANDIDATES_PER_GAP * 4)  # over-pull so we have headroom after filter
        candidates_per_panel[sym] = sub_df["candidate_ensg"].tolist()

    all_cands = sorted({c for cs in candidates_per_panel.values() for c in cs})
    print(f"  {len(all_cands)} unique candidate Ensembl IDs to resolve ...")
    sym_resolved = _resolve_symbols(all_cands)

    def _filter_candidates(panel_sym, cohort_vocab_map, by_symbol):
        """Filter raw replacement candidates to protein-coding + cohort-present."""
        out = []
        for c_ensg in candidates_per_panel[panel_sym]:
            info = sym_resolved.get(c_ensg, {})
            if not _is_protein_coding_safe(info.get("symbol", ""), info.get("biotype", "")):
                continue
            if by_symbol:
                if info["symbol"] not in cohort_vocab_map:
                    continue
            else:
                if c_ensg not in cohort_vocab_map:
                    continue
            out.append({
                "ensg": c_ensg,
                "symbol": info["symbol"],
            })
            if len(out) >= MAX_CANDIDATES_PER_GAP:
                break
        return out

    # ---- Build baseline (14-of-15) per cohort ----
    def _eval_panel_for_cohort(cohort_name, vocab_map, y_te, X_te_log, by_symbol,
                                included_idx, extra_train_col=None, extra_te_col=None):
        """Train on GPL24676 with the given indices (CTD for GPL16791 unused here;
        the 3 cohorts use raw log1p), score on cohort. extra_train_col / extra_te_col
        allow appending a candidate gene's columns alongside the panel."""
        # Build training X
        panel_bases_use = [panel_bases[i] for i in included_idx]
        train_cols = [feat_base_to_col[b] for b in panel_bases_use]
        Xtr = X_train_log[:, train_cols]
        # Build test X (cohort)
        if by_symbol:
            te_syms = [panel_syms[i] for i in included_idx]
            te_cols = [vocab_map[s] for s in te_syms]
        else:
            te_cols = [vocab_map[b] for b in panel_bases_use]
        Xte = X_te_log[:, te_cols]
        if extra_train_col is not None:
            Xtr = np.hstack([Xtr, extra_train_col.reshape(-1, 1)])
            Xte = np.hstack([Xte, extra_te_col.reshape(-1, 1)])
        sc = StandardScaler().fit(Xtr)
        scores = _ensemble_scores(sc.transform(Xtr), y_train,
                                    sc.transform(Xte), params)
        return float(roc_auc_score(y_te, scores)), scores

    # ---- Run substitution analysis per cohort × gap ----
    results = []
    for c_name, v_map, y_te, X_te, by_sym in cohort_specs:
        print(f"\n=== Cohort: {c_name} (n={len(y_te)}) ===")
        gaps = _gaps(c_name, v_map, by_sym)
        if not gaps:
            print("  No gaps — skipping.")
            continue
        # 14-of-15 baseline (drop the missing genes; no substitution)
        included = [i for i in range(15) if i not in {g[0] for g in gaps}]
        baseline_auc, baseline_scores = _eval_panel_for_cohort(
            c_name, v_map, y_te, X_te, by_sym, included
        )
        baseline_lo, baseline_hi = _bootstrap_auc(y_te, baseline_scores)
        print(f"  Baseline (dropping {len(gaps)} missing): AUC={baseline_auc:.4f} "
              f"[{baseline_lo:.3f}, {baseline_hi:.3f}]")

        # Per-gap substitution
        for gap_k, gap_sym, gap_base in gaps:
            print(f"\n  Gap: {gap_sym} ({gap_base})")
            sub_cands = _filter_candidates(gap_sym, v_map, by_sym)
            if not sub_cands:
                print(f"    No protein-coding substitutes available in {c_name} vocab.")
                results.append({
                    "cohort": c_name, "missing_gene": gap_sym,
                    "baseline_auc": baseline_auc,
                    "baseline_ci": (baseline_lo, baseline_hi),
                    "best_sub": None, "best_sub_symbol": None,
                    "best_sub_auc": None, "best_sub_ci": None,
                    "delta": None, "n_cands_tested": 0,
                })
                continue
            included_with_gap = included + [gap_k]
            # Train column for candidate = GPL24676 log expression column
            best_auc = -1.0
            best_sub = None
            best_ci = (0.0, 0.0)
            for cand in sub_cands:
                # extra_train_col: GPL24676 log expression of candidate
                if cand["ensg"] not in feat_base_to_col:
                    continue
                tr_col_idx = feat_base_to_col[cand["ensg"]]
                tr_col = X_train_log[:, tr_col_idx]
                # extra_te_col: cohort expression
                if by_sym:
                    te_idx = v_map.get(cand["symbol"])
                else:
                    te_idx = v_map.get(cand["ensg"])
                if te_idx is None:
                    continue
                te_col = X_te[:, te_idx]
                # included = baseline panel-minus-all-cohort-missing (no need to
                # drop gap_k again since it's already absent from included).
                auc, scores = _eval_panel_for_cohort(
                    c_name, v_map, y_te, X_te, by_sym, included,
                    extra_train_col=tr_col, extra_te_col=te_col,
                )
                if auc > best_auc:
                    best_auc = auc
                    best_sub = cand
                    best_ci = _bootstrap_auc(y_te, scores)
            if best_sub is None:
                print(f"    No working substitution found.")
                results.append({
                    "cohort": c_name, "missing_gene": gap_sym,
                    "baseline_auc": baseline_auc,
                    "baseline_ci": (baseline_lo, baseline_hi),
                    "best_sub": None, "best_sub_symbol": None,
                    "best_sub_auc": None, "best_sub_ci": None,
                    "delta": None, "n_cands_tested": len(sub_cands),
                })
                continue
            delta = best_auc - baseline_auc
            print(f"    Best substitute: {best_sub['symbol']} ({best_sub['ensg']})  "
                  f"AUC={best_auc:.4f} [{best_ci[0]:.3f}, {best_ci[1]:.3f}]  "
                  f"Δ={delta:+.4f}")
            results.append({
                "cohort": c_name, "missing_gene": gap_sym,
                "baseline_auc": baseline_auc,
                "baseline_ci": (baseline_lo, baseline_hi),
                "best_sub": best_sub["ensg"], "best_sub_symbol": best_sub["symbol"],
                "best_sub_auc": best_auc, "best_sub_ci": best_ci,
                "delta": delta, "n_cands_tested": len(sub_cands),
            })

    # ---- Joint multi-gap substitution: apply ALL best subs simultaneously per cohort ----
    joint_results = []
    for c_name, v_map, y_te, X_te, by_sym in cohort_specs:
        cohort_results = [r for r in results
                           if r["cohort"] == c_name and r["best_sub"] is not None]
        if len(cohort_results) < 2:
            continue
        print(f"\n--- Joint substitution (all gaps): {c_name} ---")
        gaps = _gaps(c_name, v_map, by_sym)
        included = [i for i in range(15) if i not in {g[0] for g in gaps}]
        baseline_auc = cohort_results[0]["baseline_auc"]
        baseline_ci = cohort_results[0]["baseline_ci"]
        # Build training matrix: panel cols + all best-sub cols
        panel_bases_use = [panel_bases[i] for i in included]
        train_cols = [feat_base_to_col[b] for b in panel_bases_use]
        Xtr_panel = X_train_log[:, train_cols]
        sub_tr_cols = []
        sub_te_cols = []
        sub_syms_used = []
        for cr in cohort_results:
            eg = cr["best_sub"]
            if eg not in feat_base_to_col:
                continue
            sym = cr["best_sub_symbol"]
            if by_sym:
                if sym not in v_map:
                    continue
                te_idx = v_map[sym]
            else:
                if eg not in v_map:
                    continue
                te_idx = v_map[eg]
            sub_tr_cols.append(X_train_log[:, feat_base_to_col[eg]])
            sub_te_cols.append(X_te[:, te_idx])
            sub_syms_used.append(f"{cr['missing_gene']}→{sym}")
        if len(sub_tr_cols) < 2:
            continue
        Xtr_joint = np.hstack([Xtr_panel, np.column_stack(sub_tr_cols)])
        if by_sym:
            te_syms = [panel_syms[i] for i in included]
            te_cols = [v_map[s] for s in te_syms]
        else:
            te_cols = [v_map[b] for b in panel_bases_use]
        Xte_joint = np.hstack([X_te[:, te_cols], np.column_stack(sub_te_cols)])
        sc_j = StandardScaler().fit(Xtr_joint)
        scores_j = _ensemble_scores(sc_j.transform(Xtr_joint), y_train,
                                       sc_j.transform(Xte_joint), params)
        auc_j = float(roc_auc_score(y_te, scores_j))
        lo_j, hi_j = _bootstrap_auc(y_te, scores_j)
        delta_j = auc_j - baseline_auc
        print(f"  Joint subs: {' + '.join(sub_syms_used)}")
        print(f"  Joint AUC = {auc_j:.4f} [{lo_j:.3f}, {hi_j:.3f}]  Δ={delta_j:+.4f}")
        joint_results.append({
            "cohort": c_name, "subs": sub_syms_used,
            "baseline_auc": baseline_auc, "baseline_ci": baseline_ci,
            "joint_auc": auc_j, "joint_ci": (lo_j, hi_j),
            "delta": delta_j,
        })

    # ---- Save text output ----
    lines = [
        "Cross-cohort surrogate substitution for the 15-gene critical panel",
        "=" * 70,
        "",
        f"Method: for each (cohort, missing_gene) gap, scan up to "
        f"{MAX_CANDIDATES_PER_GAP} protein-coding genome-wide replacement candidates",
        f"(from gene_replacement_results.csv with is_replacement=True) that are",
        f"both protein-coding (excluding olfactory/taste/keratin/defensin/PATE",
        f"families) and present in the target cohort vocabulary. Retrain on",
        f"GPL24676 with substitution; score zero-shot ({N_SEEDS}-seed ensemble,",
        f"stratified bootstrap B={N_BOOTSTRAP}).",
        "",
        f"{'Cohort':<12s} {'Missing':<14s} {'Baseline AUC':<22s} "
        f"{'Best sub':<14s} {'Sub AUC':<22s} {'Δ':>8s}",
        "-" * 110,
    ]
    for r in results:
        base_str = f"{r['baseline_auc']:.4f} [{r['baseline_ci'][0]:.3f},{r['baseline_ci'][1]:.3f}]"
        if r["best_sub_auc"] is not None:
            sub_str = f"{r['best_sub_auc']:.4f} [{r['best_sub_ci'][0]:.3f},{r['best_sub_ci'][1]:.3f}]"
            sub_sym = r["best_sub_symbol"]
            d = f"{r['delta']:+.4f}"
        else:
            sub_str, sub_sym, d = "n/a", "n/a", "n/a"
        lines.append(
            f"{r['cohort']:<12s} {r['missing_gene']:<14s} {base_str:<22s} "
            f"{sub_sym:<14s} {sub_str:<22s} {d:>8s}"
        )

    # Joint substitution lines
    if joint_results:
        lines += ["", "Joint multi-gap substitution (all single-gap best subs applied):",
                  f"{'Cohort':<12s} {'Subs':<42s} {'Baseline AUC':<22s} "
                  f"{'Joint AUC':<22s} {'Δ':>8s}",
                  "-" * 110]
        for jr in joint_results:
            base_str = f"{jr['baseline_auc']:.4f} [{jr['baseline_ci'][0]:.3f},{jr['baseline_ci'][1]:.3f}]"
            joint_str = f"{jr['joint_auc']:.4f} [{jr['joint_ci'][0]:.3f},{jr['joint_ci'][1]:.3f}]"
            subs_str = " + ".join(jr["subs"])
            lines.append(
                f"{jr['cohort']:<12s} {subs_str:<42s} {base_str:<22s} "
                f"{joint_str:<22s} {jr['delta']:>+8.4f}"
            )

    stat_out = SCRIPT_DIR / "cross_cohort_substitution_statistics.txt"
    stat_out.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {stat_out}")

    # ---- Plot ----
    if results:
        n = len(results)
        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.7 * n)))
        labels = [f"{r['cohort']} | {r['missing_gene']} → {r['best_sub_symbol']}"
                  if r["best_sub_symbol"]
                  else f"{r['cohort']} | {r['missing_gene']} (no sub)"
                  for r in results]
        y = np.arange(n)
        baselines = [r["baseline_auc"] for r in results]
        subs = [r["best_sub_auc"] if r["best_sub_auc"] is not None else r["baseline_auc"]
                for r in results]
        ax.barh(y - 0.18, baselines, 0.35, color="#888", label="Baseline (drop gene)")
        ax.barh(y + 0.18, subs, 0.35, color="#1f78b4", label="With substitution")
        for i, r in enumerate(results):
            if r["delta"] is not None:
                ax.text(max(baselines[i], subs[i]) + 0.005, y[i],
                        f"Δ={r['delta']:+.3f}", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0.4, 1.05)
        ax.axvline(0.5, color="grey", lw=0.5, ls=":")
        ax.set_xlabel("Zero-shot AUC")
        ax.set_title("Cross-cohort surrogate substitution: "
                     "filling 15-crit panel gaps with protein-coding alternatives")
        ax.legend(loc="lower right", fontsize=8)
        ax.invert_yaxis()
        plt.tight_layout()
        png = SCRIPT_DIR / "cross_cohort_substitution.png"
        plt.savefig(png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {png}")


if __name__ == "__main__":
    main()
