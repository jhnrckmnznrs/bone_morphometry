#!/usr/bin/env python3
"""Compare manuscript-descriptor output with bundled reference values.

The script uses only the Python standard library. Each descriptor family is
reported separately and is checked only when its columns occur in the Rust CSV.
The bundled reference tables use the blinded sample codes QC001..QC024.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "reference"
CODE = re.compile(r"QC\d{3}", re.I)


def read_csv(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        raw = row[key]
        m = CODE.search(str(raw))
        if not m:
            raise RuntimeError(f"cannot extract QC code from {raw!r} in {path}")
        code = m.group(0).upper()
        if code in out:
            raise RuntimeError(f"duplicate {code} in {path}")
        out[code] = row
    return out


def close(a: float, b: float, atol: float, rtol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, abs_tol=atol, rel_tol=rtol)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rust_csv", type=Path)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--rtol", type=float, default=1e-7)
    ap.add_argument("--w102-atol", type=float, default=2e-4,
                    help="Cross-backend W102 equivalence gate: Rust classic MC vs Python Lewiner MC")
    ap.add_argument("--w102-rtol", type=float, default=5e-4)
    args = ap.parse_args()

    rust = read_csv(args.rust_csv, "filename")
    if set(rust) != {f"QC{i:03d}" for i in range(1,25)}:
        raise RuntimeError(f"Rust CSV must contain QC001..QC024 exactly; found {sorted(rust)}")

    specs = [
        ("skeleton", REF / "skeleton.csv", {
            "skeleton.length_density_mm_per_mm3": "skeleton_length_density_mm_per_mm3",
            "skeleton.branch_density_per_mm3": "branch_density_per_mm3",
            "skeleton.junction_cluster_density_per_mm3": "junction_cluster_density_per_mm3",
            "skeleton.endpoint_density_per_mm3": "endpoint_density_per_mm3",
            "skeleton.cycle_density_per_mm3": "cycle_density_per_mm3",
            "skeleton.graph_component_density_per_mm3": "graph_component_density_per_mm3",
            "skeleton.mean_branch_length_um": "mean_branch_length_um",
            "skeleton.mean_branch_tortuosity": "mean_branch_tortuosity",
        }, args.atol, args.rtol),
        ("mechanical_graph", REF / "mechanical_graph.csv", {
            "mechanical_graph.axial_shortest_path_tortuosity": "axial_shortest_path_tortuosity",
            "mechanical_graph.axial_edge_connectivity": "axial_edge_connectivity",
            "mechanical_graph.Cz_path_length_fraction": "Cz_path_length_fraction",
            "mechanical_graph.axial_normalized_conductance": "axial_normalized_conductance",
            "mechanical_graph.axial_dissipation_gini": "axial_dissipation_gini",
        }, args.atol, args.rtol),
        ("soi_proxy", REF / "soi_proxy.csv", {
            "soi_proxy.SOI_proxy": "SOI_proxy",
            "soi_proxy.pO": "pO",
            "soi_proxy.rO": "rO",
            "soi_proxy.prO": "prO",
            "soi_proxy.n_plate": "n_plate",
            "soi_proxy.n_rod": "n_rod",
        }, args.atol, args.rtol),
        ("orientation_field", REF / "orientation_field.csv", {
            "orientation_field.plate_local_misorientation_deg": "plate_local_misorientation_deg",
            "orientation_field.rod_local_misorientation_deg": "rod_local_misorientation_deg",
            "orientation_field.plate_rod_local_orthogonality_deviation_deg": "plate_rod_local_orthogonality_deviation_deg",
        }, args.atol, args.rtol),
        ("minkowski_w102", REF / "minkowski_w102.csv", {
            "minkowski_w102.W102_DA": "W102_DA",
            "minkowski_w102.W102_mid_over_max": "W102_mid_over_max",
        }, args.w102_atol, args.w102_rtol),
    ]

    any_checked = False
    failed = False
    for family, ref_path, mapping, atol, rtol in specs:
        if not all(col in next(iter(rust.values())) for col in mapping):
            print(f"SKIP {family}: columns not present")
            continue
        any_checked = True
        ref = read_csv(ref_path, "sample")
        max_abs = 0.0
        max_rel = 0.0
        mismatches = []
        for code in sorted(rust):
            if code not in ref:
                mismatches.append((code, "missing reference row", "", "")); continue
            for rust_col, ref_col in mapping.items():
                a = float(rust[code][rust_col]); b = float(ref[code][ref_col])
                if math.isfinite(a) and math.isfinite(b):
                    max_abs = max(max_abs, abs(a-b))
                    max_rel = max(max_rel, abs(a-b)/max(abs(b), 1e-30))
                if not close(a,b,atol,rtol):
                    mismatches.append((code, rust_col, a, b))
        if mismatches:
            failed = True
            print(f"FAIL {family}: {len(mismatches)} mismatches; max_abs={max_abs:.6g}, max_rel={max_rel:.6g}")
            for item in mismatches[:12]: print("  ", item)
        else:
            print(f"PASS {family}: {len(mapping)} columns x {len(rust)} specimens; max_abs={max_abs:.6g}, max_rel={max_rel:.6g}")

    if not any_checked:
        raise RuntimeError("no reference descriptor family could be checked")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
