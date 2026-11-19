#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validate a Release B (rehydratable) Zenodo package.

OFF view:
  - MUST have unique global_id within each split
  - Splits MUST be disjoint by global_id (no leakage)

ON view:
  - global_id duplicates are EXPECTED (variants expanded)
  - MUST have non-empty text
  - SHOULD have no duplicate (text,label) rows (should be 0 ideally)
  - (optional) ON global_ids should be subset of OFF global_ids for the same split

Generates:
  docs/QA_REPORT.md
"""

import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd


REQUIRED_COLUMNS = ["global_id", "label", "text"]


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path, dtype=str)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns {missing}. Found: {list(df.columns)}")
    for c in REQUIRED_COLUMNS:
        df[c] = df[c].astype(str)
    return df


def basic_stats(df: pd.DataFrame) -> dict:
    n_rows = len(df)
    n_empty_text = (df["text"].fillna("").str.strip() == "").sum()
    n_dup_gid = df["global_id"].duplicated().sum()
    n_dup_text_label = df.duplicated(subset=["text", "label"]).sum()
    return {
        "rows": int(n_rows),
        "empty_text": int(n_empty_text),
        "dup_global_id": int(n_dup_gid),
        "dup_text_label": int(n_dup_text_label),
    }


def overlap(a: pd.Series, b: pd.Series) -> int:
    return len(set(a.astype(str)) & set(b.astype(str)))


def md_table(rows, headers) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release_dir", type=str, default=".", help="Path to zenodo_release (use '.' if you cd into it).")
    args = ap.parse_args()

    release_dir = Path(args.release_dir).resolve()
    data_dir = release_dir / "data"
    docs_dir = release_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Recommended metadata files (warn if missing)
    recommended_files = [
        release_dir / "README.md",
        release_dir / "requirements.txt",
        release_dir / "LICENSE",
        release_dir / "CITATION.cff",
        docs_dir / "manifest.yaml",
    ]

    # Paths
    off_dir = data_dir / "ablation_off"
    on_dir  = data_dir / "ablation_on"

    off_train = off_dir / "train.csv"
    off_val   = off_dir / "val.csv"
    off_test  = off_dir / "test.csv"

    on_train = on_dir / "train.csv"
    on_val   = on_dir / "val.csv"
    on_test  = on_dir / "test.csv"

    # Load
    off_tr = read_csv_checked(off_train)
    off_va = read_csv_checked(off_val)
    off_te = read_csv_checked(off_test)

    on_tr = read_csv_checked(on_train)
    on_va = read_csv_checked(on_val)
    on_te = read_csv_checked(on_test)

    # OFF leakage check
    leak_tr_va = overlap(off_tr["global_id"], off_va["global_id"])
    leak_tr_te = overlap(off_tr["global_id"], off_te["global_id"])
    leak_va_te = overlap(off_va["global_id"], off_te["global_id"])

    # OFF uniqueness checks
    off_dup = {
        "OFF train": int(off_tr["global_id"].duplicated().sum()),
        "OFF val":   int(off_va["global_id"].duplicated().sum()),
        "OFF test":  int(off_te["global_id"].duplicated().sum()),
    }

    # ON subset sanity check (recommended)
    # ON may have duplicates, but its unique global_ids should match OFF split global_ids.
    subset_ok = {
        "ON train": set(on_tr["global_id"]).issubset(set(off_tr["global_id"])),
        "ON val":   set(on_va["global_id"]).issubset(set(off_va["global_id"])),
        "ON test":  set(on_te["global_id"]).issubset(set(off_te["global_id"])),
    }

    # Stats table
    stats_rows = [
        ["OFF train", *basic_stats(off_tr).values()],
        ["OFF val",   *basic_stats(off_va).values()],
        ["OFF test",  *basic_stats(off_te).values()],
        ["ON train",  *basic_stats(on_tr).values()],
        ["ON val",    *basic_stats(on_va).values()],
        ["ON test",   *basic_stats(on_te).values()],
    ]
    stats_headers = ["Split", "Rows", "Empty text", "Duplicate global_id", "Duplicate (text,label)"]

    missing_recommended = [str(p.relative_to(release_dir)) for p in recommended_files if not p.exists()]

    # Build report
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    report = []
    report.append("# QA Report — GoEmotions Roman Urdu (Rehydratable) Ablation ON/OFF\n")
    report.append(f"Generated: **{now}**\n")

    report.append("## Package structure checks\n")
    if missing_recommended:
        report.append("⚠️ Missing recommended files (not fatal, but strongly suggested before Zenodo upload):\n")
        report.extend([f"- {m}" for m in missing_recommended])
        report.append("")
    else:
        report.append("✅ All recommended metadata files are present.\n")

    report.append("## Data file statistics\n")
    report.append(md_table(stats_rows, stats_headers))
    report.append("")
    report.append("Note: **Duplicate global_id is expected in ON splits** because variants are expanded into multiple rows per original example.\n")

    report.append("## Split leakage checks (OFF view)\n")
    report.append(f"- train ∩ val overlap: **{leak_tr_va}**\n")
    report.append(f"- train ∩ test overlap: **{leak_tr_te}**\n")
    report.append(f"- val ∩ test overlap: **{leak_va_te}**\n\n")
    if leak_tr_va == 0 and leak_tr_te == 0 and leak_va_te == 0:
        report.append("✅ No OFF split leakage detected.\n")
    else:
        report.append("❌ Leakage detected in OFF splits. Fix before publishing.\n")

    report.append("## OFF uniqueness checks\n")
    report.append(f"- OFF train duplicate global_id: **{off_dup['OFF train']}**\n")
    report.append(f"- OFF val duplicate global_id: **{off_dup['OFF val']}**\n")
    report.append(f"- OFF test duplicate global_id: **{off_dup['OFF test']}**\n\n")
    if all(v == 0 for v in off_dup.values()):
        report.append("✅ OFF global_id uniqueness OK.\n")
    else:
        report.append("❌ OFF global_id duplicates found. Fix before publishing.\n")

    report.append("## ON subset sanity checks (recommended)\n")
    report.append(f"- ON train ⊆ OFF train global_id: **{subset_ok['ON train']}**\n")
    report.append(f"- ON val ⊆ OFF val global_id: **{subset_ok['ON val']}**\n")
    report.append(f"- ON test ⊆ OFF test global_id: **{subset_ok['ON test']}**\n\n")

    out_path = docs_dir / "QA_REPORT.md"
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote QA report: {out_path}")

    # Fatal conditions for Zenodo readiness
    fatal = []
    if leak_tr_va or leak_tr_te or leak_va_te:
        fatal.append("OFF split leakage")
    if any(v > 0 for v in off_dup.values()):
        fatal.append("OFF duplicate global_id")
    # empty text is a quality failure in either view
    if (off_tr["text"].str.strip() == "").any() or (off_va["text"].str.strip() == "").any() or (off_te["text"].str.strip() == "").any():
        fatal.append("OFF has empty text")
    if (on_tr["text"].str.strip() == "").any() or (on_va["text"].str.strip() == "").any() or (on_te["text"].str.strip() == "").any():
        fatal.append("ON has empty text")
    # subset checks are warnings, not fatal

    if fatal:
        print("\n❌ NOT READY to upload yet. Fix these fatal issues:")
        for x in fatal:
            print(" -", x)
    else:
        print("\n✅ READY (data integrity): OFF leakage=0, OFF global_id unique, no empty text.")
        if missing_recommended:
            print("⚠️ Add missing metadata files (LICENSE, CITATION.cff, etc.) before Zenodo upload (see QA report).")


if __name__ == "__main__":
    main()
