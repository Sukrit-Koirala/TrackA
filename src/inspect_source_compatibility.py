"""
inspect_source_compatibility.py  --  MVP 4c-2 Stage 1

Verifies that the student source datastore is self-consistent and matches the
provenance events, so that h_train[stream_step] correctly indexes into the right
h vector for every write event.

The student pipeline uses a single datastore throughout:
  source/states/datastore.pt   (GPT-2 base, 768-dim, N rows)

build_write_object_provenance.py loads from this SAME file (line 92).
replay_online_memory_chronologically.py must also load from this SAME file.
This check verifies that alignment.

Checks (all blocking):
  C1  source/states/datastore.pt exists
  C2  h dim == 768  (student GPT-2 base, NOT 1024 teacher)
  C3  required fields present: h, y  (p_gpt_true / nll_gpt advisory)
  C4  y-token spot-check against provenance events  (if events exist)
  C5  max stream_step in events < N_rows  (no out-of-bounds h lookups)

Outputs:
  {output}/validation/source_compatibility.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _load_pt(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{label}: {path} not found")
    print(f"  Loading {label}: {path}")
    return torch.load(path, weights_only=False, map_location="cpu")


def inspect(source: Path, out: Path) -> dict:
    val_dir = out / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {"checks": {}, "artifacts": {}, "pass": False}

    # ── C1: student datastore exists ─────────────────────────────────────────
    ds_path = source / "states" / "datastore.pt"
    c1 = ds_path.exists()
    result["checks"]["C1_datastore_exists"] = {
        "pass": c1, "path": str(ds_path),
        "reason": None if c1 else f"datastore.pt not found at {ds_path}",
    }
    print(f"  C1 datastore exists: {c1}  ({ds_path})")
    if not c1:
        result["artifacts"]["student"] = {"path": str(ds_path), "found": False}
        result["verdict"] = "INCOMPATIBLE"
        _save(result, val_dir)
        return result

    ds = _load_pt(ds_path, "student datastore")
    h  = ds["h"]
    y  = ds.get("y", None)
    N, D = int(h.shape[0]), int(h.shape[1])
    has_p   = "p_gpt_true" in ds
    has_nll = "nll_gpt" in ds

    result["artifacts"]["student"] = {
        "path":             str(ds_path),
        "N_rows":           N,
        "h_dim":            D,
        "has_y":            y is not None,
        "has_p_gpt_true":   has_p,
        "has_nll_gpt":      has_nll,
        "dtype":            str(h.dtype),
    }
    print(f"  Student datastore: N={N:,}  D={D}  has_y={y is not None}  "
          f"has_p_gpt_true={has_p}")

    # ── C2: student dim == 768 ───────────────────────────────────────────────
    c2 = (D == 768)
    result["checks"]["C2_student_dim_768"] = {
        "pass": c2, "h_dim": D,
        "reason": None if c2 else f"Expected 768, got {D} — may be teacher model",
    }
    print(f"  C2 student dim=768: {c2}  (got {D})")

    # ── C3: required fields ──────────────────────────────────────────────────
    c3 = (y is not None)
    result["checks"]["C3_required_fields"] = {
        "pass":          c3,
        "has_y":         c3,
        "has_p_gpt_true": has_p,
        "has_nll_gpt":   has_nll,
        "reason": None if c3 else "y field missing from datastore.pt",
    }
    if not has_p and not has_nll:
        print(f"  C3 WARNING: neither p_gpt_true nor nll_gpt in datastore — "
              f"NLL computation will fall back to zeros")

    # ── C4 + C5: cross-check against provenance events ───────────────────────
    prov_dir  = out / "provenance"
    evt_paths = list(prov_dir.glob("*_events.parquet"))

    if not evt_paths:
        result["checks"]["C4_y_token_spot_check"] = {
            "pass": None, "reason": "no provenance events yet (run Stage 2 first)"
        }
        result["checks"]["C5_stream_step_bounds"] = {
            "pass": None, "reason": "no provenance events yet"
        }
        print("  C4/C5: provenance events not yet available (skipped)")
    else:
        import pandas as pd
        for evt_path in evt_paths:
            traj = evt_path.stem.replace("_events", "")
            evts = pd.read_parquet(evt_path)
            n_evts = len(evts)

            # C5: max stream_step < N
            max_step = int(evts["stream_step"].max()) if n_evts else 0
            c5 = (max_step < N)
            result["checks"][f"C5_bounds_{traj}"] = {
                "pass":     c5,
                "max_step": max_step,
                "N_rows":   N,
                "reason":   None if c5 else f"max stream_step={max_step} >= N={N} (OOB h lookup)",
            }
            print(f"  C5 [{traj}] bounds: max_step={max_step:,} < N={N:,}  → {c5}")

            # C4: y-token spot check (sample 500 events, compare true_token vs y_train)
            if y is not None and "true_token" in evts.columns:
                sample = evts[~evts["action_type"].str.upper().isin({"DEFER"})].sample(
                    min(500, n_evts), random_state=42
                )
                steps      = sample["stream_step"].to_numpy().astype(int)
                valid      = steps[steps < N]
                y_np       = y.numpy().astype(np.int64) if hasattr(y, "numpy") else np.array(y)
                y_ds_at    = y_np[valid]
                y_evt_at   = sample.loc[sample["stream_step"] < N, "true_token"].to_numpy().astype(int)
                n_check    = len(valid)
                n_match    = int((y_ds_at == y_evt_at).sum()) if n_check > 0 else 0
                match_rate = n_match / n_check if n_check > 0 else 0.0
                c4         = match_rate >= 0.99
                result["checks"][f"C4_y_token_{traj}"] = {
                    "pass":       c4,
                    "n_checked":  n_check,
                    "n_match":    n_match,
                    "match_rate": round(match_rate, 6),
                    "reason":     None if c4 else (
                        f"Only {match_rate:.1%} of y-tokens match between datastore and events. "
                        f"The events were built from a different stream than datastore.pt."
                    ),
                }
                print(f"  C4 [{traj}] y-token match (n={n_check}): {match_rate:.2%}  → {c4}")

    # ── Overall verdict ───────────────────────────────────────────────────────
    blocking = ["C1_datastore_exists", "C2_student_dim_768", "C3_required_fields"]
    # Also include any C4/C5 checks that ran
    for k in result["checks"]:
        if k.startswith("C4_") or k.startswith("C5_"):
            blocking.append(k)

    all_pass = all(
        result["checks"].get(k, {}).get("pass") is not False   # None = skip, False = fail
        for k in blocking
    )
    # Require at least C1+C2 to be True
    must_true = ["C1_datastore_exists", "C2_student_dim_768"]
    all_pass = all_pass and all(
        result["checks"].get(k, {}).get("pass") is True for k in must_true
    )

    result["pass"]    = all_pass
    result["verdict"] = "COMPATIBLE" if all_pass else "INCOMPATIBLE"

    _save(result, val_dir)
    print(f"\n  source_compatibility.json: {result['verdict']}")
    return result


def _save(result: dict, val_dir: Path):
    out_path = val_dir / "source_compatibility.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",  required=True,
                    help="Student source dir (e.g. scale_200k_seed42) containing states/datastore.pt")
    ap.add_argument("--output",  required=True)
    ap.add_argument("--force",   action="store_true")
    # Accept (and ignore) --datastore_dir for compatibility with old orchestrator calls
    ap.add_argument("--datastore_dir", default=None,
                    help="(ignored) teacher datastore dir — not used in student-only pipeline")
    args = ap.parse_args()

    if args.datastore_dir:
        print(f"  NOTE: --datastore_dir is not used in the student pipeline "
              f"(received: {args.datastore_dir})")

    out = Path(args.output)
    compat_path = out / "validation" / "source_compatibility.json"
    if compat_path.exists() and not args.force:
        print(f"[cached] {compat_path}")
        return

    print("=== Source Compatibility Check ===")
    result = inspect(Path(args.source), out)
    if not result["pass"]:
        print("\nERROR: Source compatibility check FAILED.")
        print("The student datastore.pt does not align with the provenance events.")
        print("Check that --source points to the directory that was used when building provenance.")
        sys.exit(1)


if __name__ == "__main__":
    main()
