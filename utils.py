"""Loaders and helpers for the ALS GEO dataset collection."""

from __future__ import annotations

import gzip
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

RESOURCES_DIR = Path(__file__).parent / "resources"

# Per-dataset rules: which metadata field encodes disease status, and how.
# '_title' is a synthetic field populated from !Sample_title (used by GSE76220
# which embeds "sALS"/"Control" in the sample name rather than a characteristic).
_LABEL_CONFIG: dict[str, dict[str, str]] = {
    "GSE153960": {
        "field": "group",
        "als_pattern": r"^ALS Spectrum MND",   # excludes Pre-fALS, Other MND
        "control_pattern": r"^Non-Neurological Control",
    },
    "GSE76220": {
        "field": "_title",
        "als_pattern": r"sALS",
        "control_pattern": r"Control",
    },
    "GSE56504": {
        "field": "patient group",
        "als_pattern": r"ALS",          # matches mtC9ORF72, mtSOD1, sALS, …
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
        meta_fields[key] = [
            v.split(":", 1)[1].strip() if ":" in v else v for v in row
        ]

    meta_df = pd.DataFrame(meta_fields, index=sample_ids)

    table_buf.seek(0)
    try:
        expr_df = pd.read_csv(table_buf, sep="\t", index_col=0)
    except Exception:
        expr_df = pd.DataFrame()

    if expr_df.columns.tolist() != sample_ids and len(expr_df.columns) == len(sample_ids):
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
    gsm_cols = [c for c in expr_df.columns if c.startswith("GSM")]
    return expr_df[gsm_cols]


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
            to ``als_analysis/resources/`` relative to this file.
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

        X = expr_df[y.index].T     # (n_samples, n_features), columns=gene ids
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
