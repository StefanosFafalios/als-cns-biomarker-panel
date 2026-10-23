"""Loaders and helpers for the ALS GEO dataset collection."""

from __future__ import annotations

import gzip
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

RESOURCES_DIR = Path(__file__).parent / "resources"

# Cohorts whose expression matrix must be per-sample rank-normalised at load
# time. GSE76220 is laser-captured motor neurons reported as log2(RPKM): ALS
# samples carry a global upward shift (12,102 genes >0.5 RPKM vs 11,346 in
# controls; per-sample median -1.10 vs -1.48), and that shift alone separates
# the groups at AUC 0.80 across the whole transcriptome. Any panel scored on the
# raw values reads that sample-level RNA-quality/composition difference rather
# than gene-specific biology. Per-sample rank-normalisation removes it by
# construction; without it, cross-cohort transfer on this dataset is confounded.
_RANK_NORMALISE_ON_LOAD = frozenset({"GSE76220"})


def rank_inverse_normal(X: np.ndarray) -> np.ndarray:
    """Rank-normalise each sample to standard-normal scores.

    Ranks the genes within every sample independently and maps those ranks
    through the normal quantile function. This equalises the expression
    distribution across samples, so a per-sample global shift (sequencing
    depth, RNA integrity, cell-type composition) cannot masquerade as signal,
    while the within-sample ordering of genes -- the biology -- is preserved.

    Args:
        X: Expression matrix, shape (n_samples, n_genes).

    Returns:
        Array of the same shape holding per-sample normal scores.
    """
    from scipy.special import ndtri
    from scipy.stats import rankdata

    r = np.apply_along_axis(rankdata, 1, np.asarray(X, dtype=np.float64))
    return ndtri(r / (X.shape[1] + 1.0))


# Per-dataset rules: which metadata field encodes disease status, and how.
# '_title' is a synthetic field populated from !Sample_title (used by GSE76220
# which embeds "sALS"/"Control" in the sample name rather than a characteristic).
_LABEL_CONFIG: dict[str, dict[str, str]] = {
    "GSE153960": {
        "field": "group",
        "als_pattern": r"^ALS Spectrum MND",  # excludes Pre-fALS, Other MND
        "control_pattern": r"^Non-Neurological Control",
    },
    "GSE76220": {
        "field": "_title",
        "als_pattern": r"sALS",
        "control_pattern": r"Control",
    },
    "GSE56504": {
        "field": "patient group",
        "als_pattern": r"ALS",  # matches mtC9ORF72, mtSOD1, sALS, …
        "control_pattern": r"^control$",
    },
    "GSE68605": {
        "field": "patient group",
        "als_pattern": r"ALS",
        "control_pattern": r"control",
    },
    "GSE112676": {
        "field": "diagnosis",
        "als_pattern": r"^ALS$",
        "control_pattern": r"^CON$",
    },
}

# Datasets whose series matrix contains no expression table; expression is
# loaded from a supplementary file instead.  Each entry specifies:
#   file         – filename under resources/<gse_id>/suppl/
#   index_col    – column to use as gene/probe row index
#   drop_cols    – non-expression columns to remove before mapping
#   col_map      – how to map suppl column names to GSM accession IDs:
#                  "title_numeric"  → strip non-digits, match leading number
#                                     in the _title metadata field (GSE76220)
#                  "meta_field"     → match directly against a metadata field;
#                                     also tries stripping a trailing -N suffix
#   meta_field   – (meta_field strategy) which metadata field to match against
# Legacy gene symbols in a cohort's own annotation. Everything downstream matches
# by CURRENT HGNC symbol, so an un-normalised alias does not raise -- it silently
# drops the gene. GSE76220 reports MECOM as EVI1, and MECOM is the rank-1 member
# of the 25-gene panel by mean|SHAP|, so it was being discarded from every
# GSE76220 evaluation on a naming technicality (findings.md Step 61).
#
# Established by querying MyGeneInfo for the aliases of all 25 panel members and
# intersecting with each cohort's vocabulary; MECOM/EVI1 is the only hit. Add to
# this table rather than special-casing at a call site -- the whole point is that
# consumers never see the alias.
_SYMBOL_ALIASES: dict[str, dict[str, str]] = {
    "GSE76220": {"EVI1": "MECOM"},
    # GPL6255 (Illumina HumanRef-8 v2) annotation predates both renames
    "GSE26927": {"EVI1": "MECOM", "TEB1": "SOHLH2"},
}


_SUPPL_EXPR: dict[str, dict[str, object]] = {
    "GSE76220": {
        "file": "GSE76220_ALS_LCM_RPKM.txt.gz",
        "index_col": "Gene_symbol",
        "drop_cols": ["NCBI_mRNA", "NCBI_Protein", "Gene_Name"],
        "col_map": "title_numeric",
    },
    "GSE153960": {
        # Header has N cols but data rows have N+1 (leading row counter).
        # Read with index_col=0 to absorb the counter, then set gene_col as index.
        "file": "GSE153960_Gene_counts_matrix_RSEM_Prudencio_et_al_2020.txt.gz",
        "index_col": 0,
        "gene_col": "EnsemblID",
        "drop_cols": [],
        "col_map": "meta_field",
        "meta_field": "sample id alt",
    },
}


@dataclass
class GEODataset:
    """Parsed GEO series: expression matrix, binary labels, and metadata.

    Attributes:
        gse_id: GEO series accession (e.g. 'GSE112676').
        platform: GPL platform identifier, or None for single-platform series.
        X: Expression matrix, shape (n_samples, n_features), index=GSM ids,
            columns=gene/probe identifiers.
        y: Binary labels, 0=control / 1=ALS, index=GSM ids.
        meta: Sample metadata, shape (n_samples, n_meta_cols), index=GSM ids.
    """

    gse_id: str
    platform: str | None
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_matrix(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        yield from fh


def _parse_row_values(line: str) -> list[str]:
    return [p.strip().strip('"') for p in line.split("\t")[1:]]


def _parse_series_matrix(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a GEO Series Matrix file into sample metadata and expression tables.

    Args:
        path: Path to a .txt or .txt.gz series matrix file.

    Returns:
        Tuple ``(meta_df, expr_df)`` where *meta_df* is indexed by GSM
        accession IDs and *expr_df* has probe/gene IDs as rows and GSM
        accession IDs as columns.  *expr_df* may be empty if the series
        matrix contains no expression table (RNA-seq stub files).
    """
    meta_fields: dict[str, list[str]] = {}
    char_rows: list[list[str]] = []
    sample_ids: list[str] = []
    table_buf = io.StringIO()
    in_table = False

    for raw_line in _open_matrix(path):
        line = raw_line.rstrip("\n")

        if line == "!series_matrix_table_begin":
            in_table = True
            continue
        if line == "!series_matrix_table_end":
            in_table = False
            continue
        if in_table:
            table_buf.write(raw_line)
            continue

        if line.startswith("!Sample_geo_accession"):
            sample_ids = _parse_row_values(line)
        elif line.startswith("!Sample_characteristics_ch1"):
            char_rows.append(_parse_row_values(line))
        elif line.startswith("!Sample_title"):
            meta_fields["_title"] = _parse_row_values(line)
        elif line.startswith("!Sample_source_name_ch1"):
            meta_fields["_source"] = _parse_row_values(line)

    # Each characteristics row encodes one key: "key: value" per sample.
    for row in char_rows:
        key = row[0].split(":")[0].strip() if row else ""
        if not key:
            continue
        meta_fields[key] = [v.split(":", 1)[1].strip() if ":" in v else v for v in row]

    meta_df = pd.DataFrame(meta_fields, index=sample_ids)

    table_buf.seek(0)
    try:
        expr_df = pd.read_csv(table_buf, sep="\t", index_col=0)
    except Exception:
        expr_df = pd.DataFrame()

    if expr_df.columns.tolist() != sample_ids and len(expr_df.columns) == len(
        sample_ids
    ):
        expr_df.columns = sample_ids

    return meta_df, expr_df


def _load_suppl_expression(
    gse_id: str,
    meta_df: pd.DataFrame,
    root: Path,
) -> pd.DataFrame:
    """Load expression data from a supplementary file and re-index to GSM IDs.

    Two column-mapping strategies are supported (controlled by _SUPPL_EXPR):

    * ``title_numeric`` – extracts the leading digit string from each suppl
      column name (e.g. ``'60a'`` → ``'60'``) and matches it against the
      leading token of each ``_title`` field in *meta_df*.
    * ``meta_field``    – matches suppl column names directly against a
      chosen metadata field; also retries after stripping a trailing ``-N``
      suffix to handle duplicate-library entries.

    Args:
        gse_id: GEO series accession.
        meta_df: Sample metadata indexed by GSM accession.
        root: Resources root directory.

    Returns:
        Expression DataFrame with probe/gene IDs as rows and GSM accession
        IDs as columns.  Samples not matched to any GSM ID are dropped.
    """
    cfg = _SUPPL_EXPR[gse_id]
    suppl_path = root / gse_id / "suppl" / str(cfg["file"])
    drop_cols: list[str] = list(cfg["drop_cols"])  # type: ignore[arg-type]

    index_col = cfg.get("index_col", 0)
    if isinstance(index_col, str):
        index_col = str(index_col)

    opener = gzip.open if suppl_path.suffix == ".gz" else open
    with opener(suppl_path, "rt", encoding="utf-8", errors="replace") as fh:
        expr_df = pd.read_csv(fh, sep="\t", index_col=index_col)  # type: ignore[arg-type]

    # Some files store the gene ID as a regular column after numeric row index.
    gene_col = cfg.get("gene_col")
    if gene_col and str(gene_col) in expr_df.columns:
        expr_df = expr_df.set_index(str(gene_col))

    expr_df = expr_df.drop(columns=[c for c in drop_cols if c in expr_df.columns])

    strategy = str(cfg.get("col_map", "title_numeric"))
    rename: dict[str, str] = {}

    if strategy == "title_numeric":
        # e.g. GSE76220: '60a' → '60' → title '60_sALS' → GSM id
        title_series = meta_df.get("_title", pd.Series(dtype=str))
        num_to_gsm: dict[str, str] = {}
        for gsm, title in title_series.items():
            num = str(title).split("_")[0].lstrip("0") or "0"
            num_to_gsm[num] = str(gsm)
        for col in expr_df.columns:
            num = re.sub(r"[^0-9]", "", str(col)).lstrip("0") or "0"
            if num in num_to_gsm:
                rename[col] = num_to_gsm[num]

    elif strategy == "meta_field":
        # e.g. GSE153960: suppl col 'CGND-HRA-00027' matches meta 'sample id alt'
        field = str(cfg["meta_field"])
        field_series = meta_df.get(field, pd.Series(dtype=str))
        val_to_gsm: dict[str, str] = {str(v): str(k) for k, v in field_series.items()}
        for col in expr_df.columns:
            col_str = str(col)
            if col_str in val_to_gsm:
                rename[col] = val_to_gsm[col_str]
            else:
                # Retry after stripping trailing -N (e.g. 'CGND-HRA-00027-2')
                stripped = re.sub(r"-\d+$", "", col_str)
                if stripped in val_to_gsm:
                    rename[col] = val_to_gsm[stripped]

    expr_df = expr_df.rename(columns=rename)

    expr_df = collapse_duplicate_rows(normalise_symbols(expr_df, gse_id))

    gsm_cols = [c for c in expr_df.columns if c.startswith("GSM")]
    return expr_df[gsm_cols]


_HGNC_CACHE = RESOURCES_DIR / "platforms" / "symbol_to_current_hgnc.json"


def _current_symbol_renames(labels: list[str]) -> dict[str, str]:
    """Legacy symbol or ``LOC<Entrez>`` label -> current HGNC symbol (MyGeneInfo, cached).

    Older annotations name a gene by a retired symbol, an alias or, before it had a
    symbol, ``LOC`` plus its Entrez ID, so a symbol match misses the same gene. Only
    unambiguous hits count; Ensembl IDs and labels that are already current map to
    nothing. The cache is shared with multicohort_meta_analysis.py.
    """
    # LOC labels resolve through their Entrez ID, cached under their own key: the meta-analysis
    # stored them as failed *symbol* lookups ("").
    def key(lab: str) -> str:
        return "@entrez:" + lab if re.fullmatch(r"LOC\d+", lab) else lab

    cache = json.loads(_HGNC_CACHE.read_text()) if _HGNC_CACHE.exists() else {}
    todo = sorted({lab for lab in labels if key(lab) not in cache and not lab.startswith("ENSG")})
    if todo:
        import mygene

        loc = [lab for lab in todo if key(lab) != lab]
        hits: dict[str, set[str]] = {}
        for query, scopes in (([lab for lab in todo if lab not in set(loc)], "symbol,alias,retired"),
                              ([lab[3:] for lab in loc], "entrezgene")):
            if not query:
                continue
            for r in mygene.MyGeneInfo().querymany(query, scopes=scopes, fields="symbol", species="human",
                                                   verbose=False):
                if not r.get("notfound") and r.get("symbol"):
                    lab = "LOC" + str(r["query"]) if scopes == "entrezgene" else r["query"]
                    hits.setdefault(lab, set()).add(r["symbol"])
        cache.update({key(lab): (next(iter(hits[lab])) if len(hits.get(lab, ())) == 1 else "") for lab in todo})
        _HGNC_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _HGNC_CACHE.write_text(json.dumps(cache))
    return {lab: cache[key(lab)] for lab in labels if cache.get(key(lab)) and cache[key(lab)] != lab}


def normalise_symbols(expr_df: "pd.DataFrame", gse_id: str) -> "pd.DataFrame":
    """Rewrite legacy gene symbols in a cohort's index to current HGNC symbols.

    Applied once at load time so no consumer has to know about aliases: the
    curated per-cohort table first, then any retired symbol, alias or
    ``LOC<Entrez>`` label MyGeneInfo resolves to exactly one current symbol.

    Args:
        expr_df: Expression frame indexed by gene symbol.
        gse_id: GEO series accession, used to look up the alias table.

    Returns:
        The frame, with any legacy symbols in the index renamed. The input is
        returned unchanged when there is nothing to do.
    """
    labels = [str(i) for i in expr_df.index]
    renames = {**_current_symbol_renames(labels), **_SYMBOL_ALIASES.get(gse_id, {})}
    hits = {a: b for a, b in renames.items() if a in expr_df.index}
    if not hits:
        return expr_df
    curated = [a for a in hits if a in _SYMBOL_ALIASES.get(gse_id, {})]
    print(f"  [{gse_id}] normalised {len(hits)} legacy symbols"
          + (": " + ", ".join(f"{a}->{hits[a]}" for a in curated) if curated else ""))
    return expr_df.rename(index=hits)


def collapse_duplicate_rows(expr_df: "pd.DataFrame") -> "pd.DataFrame":
    """Keep one row per gene label: the row with the highest mean expression.

    Symbol-keyed files can carry several rows per symbol (GSE76220 has 1,203:
    transcript-level rows, and legacy aliases renamed onto an existing gene).
    Consumers build ``{symbol: column}`` maps, so without this the row a script
    used depended on whether it kept the first or the last duplicate. For
    NR4A1 those are an expressed row and a near-constant floor row, and the two
    choices moved the panel's GSE76220 AUC by 0.03. Keeping the most highly
    expressed row is the standard collapse (WGCNA ``collapseRows`` maxMean).

    Args:
        expr_df: Expression frame indexed by gene label, samples as columns.

    Returns:
        The frame with unique labels and the original row order. The input is
        returned unchanged when it has no duplicates.
    """
    if not expr_df.index.duplicated().any():
        return expr_df
    mean = pd.Series(expr_df.mean(axis=1, numeric_only=True).to_numpy())
    keep = mean.groupby(expr_df.index.to_numpy()).idxmax().to_numpy()
    return expr_df.iloc[np.sort(keep)]


def _assign_labels(meta_df: pd.DataFrame, gse_id: str) -> pd.Series:
    """Derive binary labels from sample metadata using per-dataset patterns.

    Args:
        meta_df: Sample metadata indexed by GSM accession.
        gse_id: GEO series accession used to look up label config.

    Returns:
        Float Series with 0.0 (control), 1.0 (ALS), or NaN (excluded).
        ALS takes priority when both patterns match the same sample.
    """
    cfg = _LABEL_CONFIG[gse_id]
    field = cfg["field"]
    als_re = re.compile(cfg["als_pattern"], re.IGNORECASE)
    ctrl_re = re.compile(cfg["control_pattern"], re.IGNORECASE)

    values: pd.Series = meta_df.get(field, pd.Series(dtype=str, index=meta_df.index))
    values = values.fillna("").astype(str)

    is_als = values.apply(lambda v: bool(als_re.search(v)))
    is_ctrl = values.apply(lambda v: bool(ctrl_re.search(v)))

    y = pd.Series(np.nan, index=meta_df.index, dtype=float)
    y[is_ctrl & ~is_als] = 0.0
    y[is_als] = 1.0
    return y


# Synthetic fields added by the parser — never informative as ML features.
_INTERNAL_META_FIELDS: frozenset[str] = frozenset({"_title", "_source"})


def _encode_meta_features(
    meta: pd.DataFrame,
    gse_id: str,
    prefix: str,
) -> pd.DataFrame:
    """Encode metadata columns as numeric features with a name prefix.

    Numeric columns are included as-is.  String columns are label-encoded
    (``pd.Categorical`` integer codes); unknown / NaN values become NaN.
    The label field and internal parser fields are excluded to avoid leakage.

    Args:
        meta: Sample metadata indexed by GSM accession.
        gse_id: GEO series accession, used to identify and exclude the label
            field.
        prefix: String prepended to every output column name, separated by
            ``_``.

    Returns:
        DataFrame of shape ``(n_samples, n_meta_cols)`` with float dtype and
        columns named ``{prefix}_{original_col}``.
    """
    label_field = _LABEL_CONFIG[gse_id]["field"]
    exclude = _INTERNAL_META_FIELDS | {label_field}

    encoded: dict[str, pd.Series] = {}
    for col in meta.columns:
        if col in exclude:
            continue
        col_key = f"{prefix}_{col}"
        numeric = pd.to_numeric(meta[col], errors="coerce")
        if numeric.notna().any():
            encoded[col_key] = numeric
        else:
            codes = pd.Categorical(meta[col]).codes.astype(float)
            codes[codes < 0] = np.nan
            encoded[col_key] = pd.Series(codes, index=meta.index)

    return pd.DataFrame(encoded, index=meta.index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(
    gse_id: str,
    platform: str | None = None,
    resources_dir: Path | str | None = None,
    include_meta: bool = False,
    meta_prefix: str = "meta",
) -> list[GEODataset]:
    """Load all platform files for one GEO accession.

    Samples with ambiguous or missing labels (neither ALS nor control) are
    silently excluded from the returned datasets.  For datasets whose series
    matrix contains no expression table (e.g. GSE76220), expression is loaded
    from the corresponding supplementary file.

    Args:
        gse_id: GEO series accession (e.g. 'GSE112676').
        platform: If given, restrict to files containing this GPL id.
        resources_dir: Directory containing per-GSE subdirectories.  Defaults
            to ``resources/`` relative to this file.
        include_meta: If ``True``, append encoded metadata columns to *X*.
            Numeric fields are included as-is; string fields are label-encoded.
            The disease-label field and internal parser fields are excluded.
        meta_prefix: Prefix applied to every appended metadata column name,
            formatted as ``{meta_prefix}_{field_name}``.  Defaults to
            ``'meta'``.  Useful for downstream filtering, e.g.
            ``X.filter(like='meta_')`` to isolate or drop clinical features.

    Returns:
        List of :class:`GEODataset`, one per platform file found.

    Raises:
        KeyError: If *gse_id* is not in the label config.
        FileNotFoundError: If no matrix files exist for *gse_id*.
    """
    if gse_id not in _LABEL_CONFIG:
        raise KeyError(f"No label config for {gse_id!r}. Add it to _LABEL_CONFIG.")

    root = Path(resources_dir) if resources_dir else RESOURCES_DIR
    matrix_dir = root / gse_id / "matrix"
    files = sorted(matrix_dir.glob("*.txt.gz")) + sorted(matrix_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No matrix files found in {matrix_dir}")
    if platform:
        files = [f for f in files if platform in f.stem]

    datasets: list[GEODataset] = []
    for path in files:
        plat_match = re.search(r"GPL\d+", path.stem)
        plat_id = plat_match.group() if plat_match else None

        meta_df, expr_df = _parse_series_matrix(path)

        # Fall back to supplementary file when the series matrix has no data.
        if expr_df.empty and gse_id in _SUPPL_EXPR:
            expr_df = _load_suppl_expression(gse_id, meta_df, root)

        # Align expression columns and metadata index on common samples.
        common = meta_df.index.intersection(expr_df.columns)
        meta_df = meta_df.loc[common]
        expr_df = expr_df[common]

        y_raw = _assign_labels(meta_df, gse_id)
        mask = y_raw.notna()
        y = y_raw[mask].astype(int)

        X = expr_df[y.index].T  # (n_samples, n_features), columns=gene ids
        if gse_id in _RANK_NORMALISE_ON_LOAD:
            X = pd.DataFrame(
                rank_inverse_normal(X.values.astype(np.float64)),
                index=X.index,
                columns=X.columns,
            )
        meta = meta_df.loc[y.index]

        if include_meta:
            meta_feats = _encode_meta_features(meta, gse_id, meta_prefix)
            X = pd.concat([X, meta_feats], axis=1)

        datasets.append(
            GEODataset(gse_id=gse_id, platform=plat_id, X=X, y=y, meta=meta)
        )

    return datasets


def load_all(
    resources_dir: Path | str | None = None,
    include_meta: bool = False,
    meta_prefix: str = "meta",
) -> dict[str, list[GEODataset]]:
    """Load every configured GEO dataset found under *resources_dir*.

    Datasets whose ``matrix/`` directory is missing are silently skipped.

    Args:
        resources_dir: Override for the default resources directory.
        include_meta: Forwarded to :func:`load_dataset`.
        meta_prefix: Forwarded to :func:`load_dataset`.

    Returns:
        Dict mapping each GSE accession to a list of :class:`GEODataset`
        (one entry per platform file).
    """
    root = Path(resources_dir) if resources_dir else RESOURCES_DIR
    result: dict[str, list[GEODataset]] = {}
    for gse_id in _LABEL_CONFIG:
        matrix_dir = root / gse_id / "matrix"
        if not matrix_dir.exists():
            continue
        try:
            result[gse_id] = load_dataset(
                gse_id,
                resources_dir=root,
                include_meta=include_meta,
                meta_prefix=meta_prefix,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[load_all] Skipping {gse_id}: {exc}")
    return result


# ---------------------------------------------------------------------------
# LightGBM hyperparameter transfer across feature spaces
# ---------------------------------------------------------------------------

# The selected classifier (SELECTED_LGBM_PARAMS below) was chosen on the top-500
# SHAP feature space, where its colsample_bytree selects about 25 features per
# tree.
#
# colsample_bytree is a *fraction*, so reusing that file verbatim on a smaller
# feature space silently changes what the model does. LightGBM uses
# max(1, floor(colsample_bytree * n_features)) features per tree, so on a
# 25- or 15-gene panel the value floors to ONE feature per tree: every tree
# becomes a ~20-split step function of a single gene (num_leaves = 21) and the
# ensemble loses all feature interactions.
#
# The correction requires no re-optimisation: convert the tuned fraction back
# to the absolute features-per-tree count it encodes, then re-express that for
# whatever feature space is actually being modelled. For any panel of <= 33
# features this saturates at 1.0 (every gene available to every tree).
TUNED_FEATURE_SPACE = 500


def drop_if_other_ranking(order_file: Path, names: list[str], *cached: Path) -> None:
    """Delete cached results computed on another gene ranking, so they are only reused for this one.

    A finished curve or LOO table used to be loaded whenever it existed; after the discovery changed
    (2026-10-10) that silently reused the old genes' results. ``order_file`` records the ranking.
    """
    if not order_file.exists() or order_file.read_text().split() != list(names):
        for path in cached:
            path.unlink(missing_ok=True)
        order_file.write_text("\n".join(names))


def adjust_colsample(params: dict, n_features: int) -> dict:
    """Re-express ``colsample_bytree`` for a feature space of *n_features*.

    Args:
        params: LightGBM hyperparameters as tuned on ``TUNED_FEATURE_SPACE``
            features. Must contain ``colsample_bytree``.
        n_features: Number of features actually being modelled.

    Returns:
        A copy of *params* with ``colsample_bytree`` set to
        ``min(1.0, k / n_features)``, where ``k`` is the absolute
        features-per-tree count encoded by the tuned fraction. All other
        hyperparameters are unchanged.

    Raises:
        KeyError: If *params* has no ``colsample_bytree`` entry.
    """
    out = dict(params)
    k = params["colsample_bytree"] * TUNED_FEATURE_SPACE
    out["colsample_bytree"] = float(min(1.0, k / max(int(n_features), 1)))
    return out


# The selected configuration. A Bayesian search evaluated candidate preprocessing,
# feature-selection and classifier configurations by five-fold cross-validation on the
# top 500 genes of the discovery cohort (874 GPL24676 samples) and selected: remove
# constant columns, keep the genes that carry 99.9% of a LightGBM model's cumulative
# gain, and classify with the LightGBM below. Every analysis uses this classifier.
SELECTED_LGBM_PARAMS: dict = {
    "n_estimators": 1000,
    "learning_rate": 0.1816023246650197,
    "num_leaves": 207,
    "max_depth": -1,
    "min_child_samples": 80,
    "colsample_bytree": 0.05029103244738423,
    "subsample": 0.1562498984136434,
    "reg_lambda": 0.013690003726503839,
    "reg_alpha": 0.0005364794617175931,
    "random_state": 0,
    "verbose": -1,
}
SELECTOR_LGBM_PARAMS: dict = {
    "colsample_bytree": 0.99,
    "importance_type": "gain",
    "learning_rate": 0.01,
    "min_child_samples": 7,
    "random_state": 0,
    "reg_lambda": 1e-10,
    "verbose": -1,
}
SELECTOR_GAIN_FRACTION = 0.999


def selected_lgbm_params() -> dict:
    """A fresh copy of the selected classifier's hyperparameters."""
    return dict(SELECTED_LGBM_PARAMS)


def cumulative_gain_mask(importances: np.ndarray, fraction: float) -> np.ndarray:
    """Boolean mask of the smallest top-ranked set carrying ``fraction`` of total importance.

    Args:
        importances: Per-feature importances.
        fraction: Share of the summed importance to keep, in (0, 1].

    Returns:
        Mask aligned with ``importances``; only the top feature when none is positive.
    """
    importances = np.asarray(importances, dtype=float)
    order = np.argsort(-importances, kind="stable")
    total = float(importances.sum())
    n_keep = 1 if total <= 0 else int(np.searchsorted(np.cumsum(importances[order]), fraction * total, side="left")) + 1
    mask = np.zeros(importances.size, dtype=bool)
    mask[order[: min(n_keep, importances.size)]] = True
    return mask


def selected_pipeline():
    """The selected configuration as a scikit-learn pipeline.

    Constant columns are dropped (all finite values equal, or none finite), a LightGBM
    model fitted on the remaining genes in their input order ranks them by gain, the
    smallest top set carrying 99.9% of the total is kept, and the selected classifier is
    fitted on that set with the genes in name order. The order matters because LightGBM
    samples columns by position; it is the order in which the search presented them. Fit
    and predict on a DataFrame whose columns are the gene identifiers. Built lazily so
    importing :mod:`utils` does not require scikit-learn or lightgbm.

    Returns:
        An unfitted :class:`sklearn.pipeline.Pipeline`.
    """
    from lightgbm import LGBMClassifier
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import Pipeline

    class ConstantColumnRemover(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            v = X.to_numpy(dtype=float)
            fin = np.isfinite(v)
            lo = np.where(fin, v, np.inf).min(axis=0)
            hi = np.where(fin, v, -np.inf).max(axis=0)
            self.columns_ = list(X.columns[fin.any(axis=0) & (np.abs(hi - lo) > 1e-50)])
            return self

        def transform(self, X):
            return X[self.columns_]

    class CumulativeGainSelector(TransformerMixin, BaseEstimator):
        def fit(self, X, y):
            self.estimator_ = LGBMClassifier(**SELECTOR_LGBM_PARAMS).fit(X.to_numpy(dtype=float), y)
            keep = cumulative_gain_mask(self.estimator_.feature_importances_, SELECTOR_GAIN_FRACTION)
            self.columns_ = sorted(X.columns[keep])
            return self

        def transform(self, X):
            return X[self.columns_].to_numpy(dtype=float)

    return Pipeline([
        ("constant", ConstantColumnRemover()),
        ("select", CumulativeGainSelector()),
        ("classify", LGBMClassifier(**SELECTED_LGBM_PARAMS)),
    ])


# The exact ``colsample_bytree`` and ``min_child_samples`` of the selected classifier,
# used as signatures so the wrappers below correct only these values and never a
# hand-set or differently-tuned one.
_TOP500_PARAMS = SELECTED_LGBM_PARAMS
TOP500_COLSAMPLE = _TOP500_PARAMS["colsample_bytree"]
_COLSAMPLE_TOL = 1e-9

# min_child_samples is a count, so it too means something different on another
# training-set size. The search scored each configuration by 5-fold CV on the
# 874 GPL24676 samples, i.e. on training sets of 874 * 4/5 samples, so the tuned
# value is a fraction of the training set (tuned / 699.2). Every fit of the tuned
# model keeps that fraction (for 69: 69 on a CV fold, 86 on 874 samples, 2 on 19).
TOP500_MIN_CHILD = _TOP500_PARAMS["min_child_samples"]
TUNED_TRAIN_N = 874 * 4 / 5


def scale_min_child(n_train: int) -> int:
    """Tuned ``min_child_samples`` re-expressed for *n_train* training samples.

    Args:
        n_train: Number of samples the model is fitted on.

    Returns:
        ``round(TOP500_MIN_CHILD * n_train / TUNED_TRAIN_N)``, at least 1.
    """
    return max(1, round(TOP500_MIN_CHILD * n_train / TUNED_TRAIN_N))


_TUNED_LGBM_CLS = None
_PANEL_LGBM_CLS = None


def _tuned_lgbm_class():
    """Build (once) the LGBMClassifier subclass that rescales the tuned leaf size.

    Defined lazily so that importing :mod:`utils` does not require lightgbm.

    Returns:
        The ``_TunedLGBM`` class object.
    """
    global _TUNED_LGBM_CLS
    if _TUNED_LGBM_CLS is not None:
        return _TUNED_LGBM_CLS

    from lightgbm import LGBMClassifier

    class _TunedLGBM(LGBMClassifier):
        """LightGBM classifier that rescales the top-500 min_child_samples at fit time."""

        def fit(self, X, y, **kwargs):  # noqa: D102 - inherits LGBMClassifier.fit
            tuned = self.get_params().get("min_child_samples")
            if tuned != TOP500_MIN_CHILD:
                self.min_child_samples_effective_ = tuned
                return super().fit(X, y, **kwargs)
            self.set_params(min_child_samples=scale_min_child(len(y)))
            self.min_child_samples_effective_ = self.get_params()["min_child_samples"]
            try:
                return super().fit(X, y, **kwargs)
            finally:
                # keep the tuned value, so clones and refits rescale for their own n
                self.set_params(min_child_samples=tuned)

    _TUNED_LGBM_CLS = _TunedLGBM
    return _TUNED_LGBM_CLS


def TunedLGBMClassifier(*args, **kwargs):  # noqa: N802 - mimics a class name
    """Construct an LGBMClassifier that rescales the top-500 ``min_child_samples``.

    Drop-in replacement for :class:`lightgbm.LGBMClassifier`, intended to be
    imported as ``from utils import TunedLGBMClassifier as LGBMClassifier``. Only
    the tuned value ``TOP500_MIN_CHILD`` is rescaled (see ``scale_min_child``);
    any other ``min_child_samples`` is left untouched.

    Args:
        *args: Positional arguments forwarded to ``LGBMClassifier``.
        **kwargs: Hyperparameters forwarded to ``LGBMClassifier``.

    Returns:
        An ``LGBMClassifier`` subclass instance. After fitting,
        ``min_child_samples_effective_`` records the value actually used.
    """
    return _tuned_lgbm_class()(*args, **kwargs)


def _panel_lgbm_class():
    """Build (once) the LGBMClassifier subclass that self-corrects colsample.

    Defined lazily so that importing :mod:`utils` does not require lightgbm.

    Returns:
        The ``_PanelLGBM`` class object.
    """
    global _PANEL_LGBM_CLS
    if _PANEL_LGBM_CLS is not None:
        return _PANEL_LGBM_CLS

    class _PanelLGBM(_tuned_lgbm_class()):
        """LightGBM classifier that rescales the top-500 colsample (and leaf size) at fit time."""

        def fit(self, X, y, **kwargs):  # noqa: D102 - inherits LGBMClassifier.fit
            cs = self.get_params().get("colsample_bytree")
            shape = getattr(X, "shape", None)
            n_features = shape[1] if shape and len(shape) > 1 else 0
            if (
                cs is not None
                and abs(cs - TOP500_COLSAMPLE) < _COLSAMPLE_TOL
                and n_features
            ):
                k = TOP500_COLSAMPLE * TUNED_FEATURE_SPACE
                self.set_params(colsample_bytree=float(min(1.0, k / n_features)))
            self.colsample_bytree_effective_ = self.get_params().get("colsample_bytree")
            return super().fit(X, y, **kwargs)

    _PANEL_LGBM_CLS = _PanelLGBM
    return _PANEL_LGBM_CLS


def PanelLGBMClassifier(*args, **kwargs):  # noqa: N802 - mimics a class name
    """Construct an LGBMClassifier that self-corrects the top-500 ``colsample_bytree``.

    Drop-in replacement for :class:`lightgbm.LGBMClassifier`, intended to be
    imported as ``from utils import PanelLGBMClassifier as LGBMClassifier``.

    The top-500 tuned ``colsample_bytree`` is a *fraction* calibrated against a
    500-feature space (~33 features per tree). Applied unchanged to a 15- or
    25-gene panel it collapses to one feature per tree, which is a different
    model from the one that was tuned. The returned estimator detects that
    specific tuned value and re-expresses it for the feature space actually
    passed to ``fit``. Any other ``colsample_bytree`` is left untouched, so
    hand-set or full-feature-space parameters are unaffected.

    Args:
        *args: Positional arguments forwarded to ``LGBMClassifier``.
        **kwargs: Hyperparameters forwarded to ``LGBMClassifier``.

    Returns:
        An ``LGBMClassifier`` subclass instance with fit-time colsample
        correction. After fitting, ``colsample_bytree_effective_`` records the
        value actually used.
    """
    return _panel_lgbm_class()(*args, **kwargs)


# ---------------------------------------------------------------------------
# Cached dataset downloads
#
# Analysis scripts previously re-fetched GEO SOFT metadata from NCBI on every
# run. That makes results depend on network availability, throttles under
# repeated runs (NCBI returns an HTML error page, which surfaces as
# BadGzipFile), and prevents offline reproduction. Everything below downloads
# once into ``resources/`` and reads the local copy thereafter.
# ---------------------------------------------------------------------------

_MIN_VALID_BYTES = 1_000


def cached_download(
    url: str,
    dest: Path,
    desc: str = "",
    retries: int = 4,
    timeout: int = 900,
) -> Path:
    """Download *url* to *dest* once, reusing the local copy on later calls.

    The download is written to a ``.part`` sibling and atomically renamed, so an
    interrupted transfer can never leave a truncated file that later runs would
    mistake for a valid cache entry.

    Args:
        url: Source URL.
        dest: Destination path. Parent directories are created as needed.
        desc: Human-readable label used in progress messages.
        retries: Number of attempts before giving up.
        timeout: Per-attempt socket timeout in seconds.

    Returns:
        Path to the local file (``dest``).

    Raises:
        RuntimeError: If every attempt fails.
    """
    import time
    import urllib.request

    if dest.exists() and dest.stat().st_size > _MIN_VALID_BYTES:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            print(f"  downloading {desc or dest.name} (attempt {attempt + 1}) ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            tmp = dest.with_suffix(dest.suffix + ".part")
            with (
                urllib.request.urlopen(req, timeout=timeout) as r,
                open(tmp, "wb") as f,
            ):
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as exc:  # pragma: no cover - network dependent
            last = exc
            time.sleep(6)
    raise RuntimeError(f"download failed for {url}: {last}")


def geo_series_dir(gse_id: str) -> str:
    """Return the NCBI FTP series bucket for *gse_id* (e.g. GSE122649 -> GSE122nnn).

    Args:
        gse_id: GEO series accession.

    Returns:
        The bucket directory name used in NCBI's FTP layout.
    """
    digits = gse_id.removeprefix("GSE")
    return f"GSE{digits[:-3] or '0'}nnn" if len(digits) > 3 else "GSEnnn"


def geo_soft_path(gse_id: str, resources_dir: Path | None = None) -> Path:
    """Return the local cache path for a GEO SOFT family file, downloading once.

    Args:
        gse_id: GEO series accession (e.g. ``"GSE122649"``).
        resources_dir: Resources root; defaults to :data:`RESOURCES_DIR`.

    Returns:
        Path to the local ``<gse_id>_family.soft.gz``.
    """
    root = Path(resources_dir) if resources_dir is not None else RESOURCES_DIR
    dest = root / gse_id / "soft" / f"{gse_id}_family.soft.gz"
    url = (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_series_dir(gse_id)}/"
        f"{gse_id}/soft/{gse_id}_family.soft.gz"
    )
    return cached_download(url, dest, f"{gse_id} SOFT")


def geo_soft_text(gse_id: str, resources_dir: Path | None = None) -> str:
    """Return the decompressed SOFT family text for *gse_id*, downloading once.

    Replaces the previous per-script pattern of fetching the SOFT file from
    NCBI on every run.

    Args:
        gse_id: GEO series accession.
        resources_dir: Resources root; defaults to :data:`RESOURCES_DIR`.

    Returns:
        The decompressed SOFT file contents as text.
    """
    path = geo_soft_path(gse_id, resources_dir)
    return gzip.decompress(path.read_bytes()).decode("utf-8", errors="ignore")


def geo_suppl_path(
    gse_id: str, filename: str, resources_dir: Path | None = None
) -> Path:
    """Return the local cache path for a GEO supplementary file, downloading once.

    Args:
        gse_id: GEO series accession.
        filename: Supplementary filename (e.g. ``"GSE122649_RAW.tar"``).
        resources_dir: Resources root; defaults to :data:`RESOURCES_DIR`.

    Returns:
        Path to the local supplementary file. An existing copy directly under
        ``resources/<gse_id>/`` is honoured before any download is attempted.
    """
    root = Path(resources_dir) if resources_dir is not None else RESOURCES_DIR
    legacy = root / gse_id / filename
    if legacy.exists() and legacy.stat().st_size > _MIN_VALID_BYTES:
        return legacy
    dest = root / gse_id / "suppl" / filename
    url = (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_series_dir(gse_id)}/"
        f"{gse_id}/suppl/{filename}"
    )
    return cached_download(url, dest, f"{gse_id} {filename}")


# Cohorts whose supplementary expression matrices are ALREADY on a log scale and
# must NOT be log1p-transformed again. GSE76220's file is named
# "GSE76220_ALS_LCM_RPKM.txt.gz" but holds log2-scale values (range -6.2..10.6,
# 63% negative); RPKM cannot be negative. A second log1p nulls every value <= -1,
# which silently removed 77% of critical-panel measurements. Per-gene
# standardisation within each cohort absorbs the log-base and length-normalisation
# differences against log1p(count) training data.
# Array cohorts (log2 intensities) are on a different scale from log1p counts too, so
# they are standardised within themselves (new_cohort_validation.py).
LOG_SCALE_COHORTS = frozenset({"GSE76220", "GSE26927", "E-MTAB-8635", "E-MTAB-2325"})


def needs_log1p(gse_id: str) -> bool:
    """Return whether *gse_id*'s expression matrix still requires a log transform.

    Args:
        gse_id: GEO series accession.

    Returns:
        ``False`` for cohorts listed in :data:`LOG_SCALE_COHORTS` (already
        log-scale), ``True`` for count-scale cohorts.
    """
    return gse_id not in LOG_SCALE_COHORTS


def zero_shot_scale(
    X_train: np.ndarray, X_test: np.ndarray, gse_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """Standardise a train/test matrix pair for zero-shot cross-cohort transfer.

    Which scaler to use depends on whether the test cohort shares the training
    cohort's expression scale:

    * Same scale (log1p counts) — fit on train and apply to both, so the test
      cohort is projected into the training feature space.
    * Different scale (:data:`LOG_SCALE_COHORTS`, e.g. GSE76220 at log2 RPKM) —
      standardise each side within its own cohort. Reusing the training scaler
      here pushes every test sample below all of the model's split thresholds:
      the classifier emits a single constant probability and AUC pins to exactly
      0.5000. Per-gene z-scoring instead absorbs the log-base and
      length-normalisation differences exactly.

    Args:
        X_train: Training matrix, shape (n_train_samples, n_features).
        X_test: Test matrix with the same columns, shape (n_test_samples,
            n_features).
        gse_id: GEO accession of the *test* cohort.

    Returns:
        The standardised ``(X_train, X_test)`` pair.
    """
    from sklearn.preprocessing import StandardScaler

    if gse_id in LOG_SCALE_COHORTS:
        return (
            StandardScaler().fit_transform(X_train),
            StandardScaler().fit_transform(X_test),
        )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


# ---------------------------------------------------------------------------
# Canonical critical-panel definition
#
# The 15-gene critical panel was previously hardcoded in ~28 analysis scripts,
# so a change to the panel silently left most of them stale. It now lives in
# GSE153960/panel_critical.csv, regenerated by greedy_significance_stop.py, and
# every consumer reads it from here.
# ---------------------------------------------------------------------------

CRITICAL_PANEL_CSV = Path(__file__).parent / "GSE153960" / "panel_critical.csv"

# The incremental CV curve from step 2c. Its value at k = 25 *is* the 25-gene
# panel's 5-fold CV AUC -- the panel is defined as the first 25 points of this
# curve -- so there is no second source to disagree with.
_CURVE_CSV = Path(__file__).parent / "GSE153960" / "lgbm_top500_final_curve.csv"


def break_even_k(auc: "np.ndarray | list[float]") -> int:
    """Core panel cutoff from the incremental curve (supplement, sec:stage3).

    With relative gain ``G(k) = (AUC(k) - AUC(1)) / (AUC(K) - AUC(1))`` over the
    full curve ``k = 1..K``, ``k* = min{k : G(k) - G(k-1) < G(K) / K}``: the first
    gene whose marginal gain is below the mean per-gene contribution.

    Args:
        auc: Mean CV AUC at k = 1..K features, in order.

    Returns:
        The break-even panel size ``k*``.
    """
    a = np.asarray(auc, dtype=float)
    g = (a - a[0]) / (a[-1] - a[0])
    below = np.flatnonzero(np.diff(g) < g[-1] / len(a))
    if not below.size:
        raise ValueError("marginal gain never falls below the mean per-gene contribution")
    return int(below[0]) + 2  # np.diff index i is the step from k = i + 1 to k = i + 2


def panel25_cv_auc(k: int | None = None) -> float:
    """Return the within-cohort 5-fold CV AUC of the SHAP-ranked top-``k`` panel.

    Four scripts carried this as ``TRAIN_CV_AUC = 0.9621``, a literal copied from
    a pre-correction run. The corrected pipeline yields 0.9611 (different CV
    folds), so every statistics file quoting the constant -- and the manuscript
    abstract that quoted those files -- reported a superseded number that no
    current artefact contains. Read it; do not restate it.

    Args:
        k: Panel size to read off the curve. Defaults to CORE_PANEL_SIZE.

    Returns:
        Mean 5-fold stratified CV AUC at ``k`` features.

    Raises:
        FileNotFoundError: If the curve CSV is missing.
        SystemExit: If the curve has no row at ``k``.
    """
    k = k or CORE_PANEL_SIZE
    if not _CURVE_CSV.exists():
        raise FileNotFoundError(
            f"{_CURVE_CSV} not found; regenerate it with "
            "GSE153960/lgbm_top500_final_analysis.py (run_all step 02c)"
        )
    curve = pd.read_csv(_CURVE_CSV)
    row = curve.loc[curve.n_features == k, "mean_auc"]
    if row.empty:
        raise SystemExit(f"{_CURVE_CSV.name} has no row at n_features={k}")
    return float(row.iloc[0])


_LR_SVM_CURVE_CSV = Path(__file__).parent / "GSE153960" / "lr_svm_greedy_forward.csv"


def lr_svm_cv_auc() -> float:
    """Return the 5-fold CV AUC of the full LR/SVM comparison panel.

    The last row of the forward-selection curve is the full panel by
    construction. Its two consumers carried the value as a literal (0.9324);
    it happened to be current, but the literal is what let the LightGBM one go
    stale unnoticed, so both are derived now.

    Returns:
        Mean 5-fold stratified CV AUC at the final forward-selection step.

    Raises:
        FileNotFoundError: If the forward-selection curve is missing.
    """
    if not _LR_SVM_CURVE_CSV.exists():
        raise FileNotFoundError(
            f"{_LR_SVM_CURVE_CSV} not found; regenerate it with "
            "GSE153960/lr_svm_panel_discovery.py"
        )
    curve = pd.read_csv(_LR_SVM_CURVE_CSV)
    return float(curve.sort_values("n_features").cv_auc.iloc[-1])


def load_critical_panel(
    protein_coding_only: bool = False, path: Path | str | None = None
) -> pd.DataFrame:
    """Load the canonical critical-panel definition.

    Args:
        protein_coding_only: Restrict to protein-coding members.
        path: Override the CSV location (defaults to
            :data:`CRITICAL_PANEL_CSV`).

    Returns:
        DataFrame with columns ``symbol``, ``feature`` (versioned Ensembl ID),
        ``ensg``, ``biotype``, ``protein_coding``, ``shap_rank_in_25`` and
        ``mean_abs_shap``, ordered by SHAP rank.

    Raises:
        FileNotFoundError: If the panel CSV is missing.
    """
    p = Path(path) if path is not None else CRITICAL_PANEL_CSV
    if not p.exists():
        raise FileNotFoundError(
            f"critical panel definition not found at {p}; "
            "regenerate it with GSE153960/greedy_significance_stop.py"
        )
    df = pd.read_csv(p)
    if protein_coding_only:
        df = df[df.protein_coding].reset_index(drop=True)
    return df


def critical_panel_symbols(protein_coding_only: bool = False) -> list[str]:
    """Return the critical-panel gene symbols.

    Args:
        protein_coding_only: Restrict to protein-coding members.

    Returns:
        Gene symbols ordered by SHAP rank.
    """
    return list(load_critical_panel(protein_coding_only).symbol)


def critical_panel_features(protein_coding_only: bool = False) -> list[str]:
    """Return the critical-panel versioned Ensembl feature IDs.

    Args:
        protein_coding_only: Restrict to protein-coding members.

    Returns:
        Versioned Ensembl IDs ordered by SHAP rank.
    """
    return list(load_critical_panel(protein_coding_only).feature)


def critical_panel_size(protein_coding_only: bool = False) -> int:
    """Return the number of critical-panel members.

    Use this for any denominator or label reported to a reader ("14/18",
    "18-crit"). Hardcoding the size defeats the point of deriving membership
    from panel_critical.csv: several scripts printed "14/15" for a matched
    count out of an 18-gene panel, and those strings were transcribed into
    the manuscript and baked into figure legends.

    Args:
        protein_coding_only: Count only protein-coding members.

    Returns:
        Number of members in the critical panel.
    """
    return len(load_critical_panel(protein_coding_only).index)


def critical_panel_label(protein_coding_only: bool = False) -> str:
    """Return the short panel label used in figures and tables, e.g. "18-crit".

    Args:
        protein_coding_only: Label the protein-coding subset instead.

    Returns:
        Label of the form "<k>-crit".
    """
    return f"{critical_panel_size(protein_coding_only)}-crit"


CORE_PANEL_CSV = Path(__file__).parent / "GSE153960" / "lgbm_core25_panel.csv"
# Genes kept from the top of the discovery SHAP ranking (known ALS genes excluded upstream);
# the pooled trimming over the external datasets then reduces them.
CORE_PANEL_SIZE = 50
GENE_INFO_CACHE = Path(__file__).parent / "GSE153960" / "gene_info_cache.json"


def gene_info(symbols: list[str]) -> dict[str, dict[str, str]]:
    """MyGeneInfo ``type_of_gene`` and Entrez ID per symbol (cached).

    Args:
        symbols: HGNC symbols (or ENSG IDs, which resolve to nothing).

    Returns:
        ``{symbol: {"type": ..., "entrez": ...}}``; both are ``""`` when the
        symbol has no MyGeneInfo hit.
    """
    import mygene

    cache = json.loads(GENE_INFO_CACHE.read_text()) if GENE_INFO_CACHE.exists() else {}
    todo = sorted({s for s in symbols if s not in cache})
    if todo:
        hits = mygene.MyGeneInfo().querymany(
            todo, scopes="symbol", fields="type_of_gene,entrezgene", species="human", verbose=False
        )
        for r in hits:  # the best-scoring hit comes first
            if not r.get("notfound"):
                cache.setdefault(r["query"], {"type": r.get("type_of_gene", ""), "entrez": str(r.get("entrezgene", ""))})
        cache.update({s: cache.get(s, {"type": "", "entrez": ""}) for s in todo})
        GENE_INFO_CACHE.write_text(json.dumps(cache))
    return {s: cache[s] for s in symbols}


def core_panel_symbols(protein_coding_only: bool = False) -> list[str]:
    """Return the 25-gene panel's symbols (lgbm_core25_panel.csv), in SHAP rank order.

    Scripts read the panel here instead of hardcoding it, so a re-derived panel
    propagates everywhere.

    Args:
        protein_coding_only: Keep only MyGeneInfo protein-coding members.

    Returns:
        Gene symbols ordered by SHAP rank.
    """
    syms = list(pd.read_csv(CORE_PANEL_CSV).sort_values("rank").symbol)
    if protein_coding_only:
        info = gene_info(syms)
        syms = [s for s in syms if info[s]["type"] == "protein-coding"]
    return syms


def core_panel_ensg(protein_coding_only: bool = False) -> dict[str, str]:
    """Return the 25-gene panel's unversioned Ensembl IDs.

    Args:
        protein_coding_only: Keep only protein-coding members.

    Returns:
        ``{symbol: ENSG}`` in SHAP rank order.
    """
    ensg = {s: str(f).split(".")[0] for s, f in pd.read_csv(CORE_PANEL_CSV)[["symbol", "feature"]].itertuples(index=False)}
    return {s: ensg[s] for s in core_panel_symbols(protein_coding_only)}


def core_panel_entrez() -> dict[str, int]:
    """Return Entrez IDs of the panel members that have one, non-coding genes included.

    Entrez-keyed cohorts list non-coding genes under their Entrez IDs too, so a
    protein-coding restriction dropped genes those cohorts measure.

    Returns:
        ``{symbol: entrez}`` in SHAP rank order.
    """
    syms = core_panel_symbols()
    info = gene_info(syms)
    return {s: int(info[s]["entrez"]) for s in syms if info[s]["entrez"]}


GENE_PROFILING_CSV = Path(__file__).parent / "GSE153960" / "gene_profiling_summary.csv"


def critical_panel_directions(protein_coding_only: bool = False) -> dict[str, str]:
    """Return each critical-panel gene's direction of effect in ALS.

    The direction comes from the discovery-cohort differential expression
    (``gene_profiling_summary.csv``), which is independent of the panel-size
    stopping rule — changing panel membership re-selects rows, it does not
    change any gene's direction. Scripts that split the panel by direction
    should call this rather than hand-maintaining up/down lists.

    Args:
        protein_coding_only: Restrict to protein-coding members.

    Returns:
        Mapping of gene symbol to ``"up"`` or ``"down"``, ordered by SHAP rank.

    Raises:
        FileNotFoundError: If the gene-profiling summary is missing.
        KeyError: If a panel member has no direction call.
    """
    return _directions(critical_panel_symbols(protein_coding_only))


def core_panel_directions(protein_coding_only: bool = False) -> dict[str, str]:
    """Return each 25-gene panel member's direction of effect in ALS ("up"/"down").

    Args:
        protein_coding_only: Restrict to protein-coding members.

    Returns:
        Mapping of gene symbol to ``"up"`` or ``"down"``, ordered by SHAP rank.
    """
    return _directions(core_panel_symbols(protein_coding_only))


def _directions(symbols: list[str]) -> dict[str, str]:
    if not GENE_PROFILING_CSV.exists():
        raise FileNotFoundError(
            f"gene profiling summary not found at {GENE_PROFILING_CSV}; "
            "regenerate it with GSE153960/gene_profiling.py"
        )
    prof = pd.read_csv(GENE_PROFILING_CSV).set_index("symbol")["direction"]
    out: dict[str, str] = {}
    for sym in symbols:
        if sym not in prof.index:
            raise KeyError(f"no direction call for panel member {sym}")
        out[sym] = "up" if str(prof[sym]).startswith("up") else "down"
    return out
