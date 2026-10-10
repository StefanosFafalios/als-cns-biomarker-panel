"""Is the spinal endothelial decrease a typing artifact? (Step 105l)

Version 1 (canonical markers) finds CLDN5, MECOM and ANGPT2 lower in ALS endothelial
nuclei; version 2 (GSE190442 reference typing) does not. ALS donors have fewer endothelial
nuclei, so a marker-typed "endothelial" group may hold a larger share of non-endothelial
nuclei in ALS, diluting endothelium-specific genes. This types every nucleus both ways and
reports, per donor:

  * purity: the share of marker-typed endothelial nuclei the reference also calls endothelial;
  * the count-model ALS effect of the endothelial genes in three nucleus sets: marker-typed,
    reference-typed, and both (the agreed core).

25-gene panel members are excluded from the reference typing markers, as in version 2.

Outputs: snrna_typing_purity.csv, snrna_typing_purity_statistics.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

import celltype_reference  # noqa: E402
from snrna_percell_validation import MIN_COUNTS, _read_h5_targets  # noqa: E402
from snrna_spinal_cord import (  # noqa: E402
    _MARKERS,
    _h5_files,
    _mwu,
    marker_assign,
    quasi_poisson,
)

GENES = ["CLDN5", "MECOM", "ANGPT2"]
_ENDO = {"endothelial_capven", "endothelial_arterial"}


def main() -> None:
    cent, ref_genes, ref_types = celltype_reference.build()
    excl = set(GENES) | set(pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv").symbol.astype(str))
    use = [i for i, g in enumerate(ref_genes) if g not in excl]
    cent, ref_genes = cent[:, use], [ref_genes[i] for i in use]
    rows = []
    for path in _h5_files():
        m = re.search(r"GSM\d+_(\d+)_(ALS|control)_", path.name)
        sub, totals, present = _read_h5_targets(
            str(path), sorted(set(sum(_MARKERS.values(), [])) | set(GENES) | set(ref_genes)))
        keep = totals >= MIN_COUNTS
        sub, totals = sub[keep], totals[keep]
        gi = {g: j for j, g in enumerate(present)}
        v1 = marker_assign(sub, gi) == "endothelial"
        v2 = np.isin(celltype_reference.assign(sub, present, cent, ref_genes, ref_types), list(_ENDO))
        raw = np.rint(np.expm1(sub[:, [gi[g] for g in GENES]]) * (totals[:, None] / 1e4))
        row = dict(donor=m.group(1), cond=m.group(2), n_marker=int(v1.sum()), n_reference=int(v2.sum()),
                   n_both=int((v1 & v2).sum()), purity=(v1 & v2).sum() / max(v1.sum(), 1))
        for name, mask in (("marker", v1), ("reference", v2), ("both", v1 & v2)):
            row[f"{name}__lib"] = float(totals[mask].sum())
            row.update({f"{name}__{g}": float(raw[mask, k].sum()) for k, g in enumerate(GENES)})
        rows.append(row)
        print(f"  {path.name[:40]:<42} purity {row['purity']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(SCRIPT_DIR / "snrna_typing_purity.csv", index=False)

    a, c = df.cond == "ALS", df.cond == "control"
    lines = ["Endothelial typing purity and the endothelial decrease (GSE287257 spinal cord)", "=" * 80,
             f"Donors: {int(a.sum())} ALS, {int(c.sum())} control. Purity = share of marker-typed endothelial",
             "nuclei that the GSE190442 reference also types as endothelial (capillary/venous or arterial).", "",
             f"{'':<34}{'ALS':>8}{'control':>9}{'MWU p':>9}"]
    for col, lab in (("n_marker", "marker-typed endothelial nuclei"), ("n_reference", "reference-typed endothelial"),
                     ("n_both", "typed endothelial by both"), ("purity", "purity of the marker-typed set")):
        lines.append(f"{lab:<34}{df.loc[a, col].mean():>8.2f}{df.loc[c, col].mean():>9.2f}"
                     f"{_mwu(df.loc[a, col].to_numpy(), df.loc[c, col].to_numpy()):>9.3g}")
    lines += ["", "Count model (quasi-Poisson, offset log library) ALS log2FC [95% CI], p, per nucleus set:",
              f"{'gene':<8}" + "".join(f"{s:>32}" for s in ("marker-typed", "reference-typed", "both"))]
    for g in GENES:
        cells = ""
        for s in ("marker", "reference", "both"):
            ok = df[f"{s}__lib"] > 0
            lfc, se, p = quasi_poisson(df.loc[ok, f"{s}__{g}"].to_numpy(float), df.loc[ok, f"{s}__lib"].to_numpy(float),
                                       (df.loc[ok, "cond"] == "ALS").to_numpy())
            cells += f"{f'{lfc:+.2f} [{lfc - 1.96 * se:+.2f}, {lfc + 1.96 * se:+.2f}] p {p:.2g}':>32}"
        lines.append(f"{g:<8}{cells}")
    (SCRIPT_DIR / "snrna_typing_purity_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
