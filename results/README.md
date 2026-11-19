# Benchmark Experimental Results

This directory contains pre-compiled, verifiable experimental outputs corresponding to all tables and figures reported in the paper:

> **A Reproducible Benchmark for Roman Urdu Emotion Classification via LLM-Based Dataset Construction and Orthographic Variation Modeling** (*IEEE Access*)

---

## Files in this Directory

| File | Description | Generating Script | Primary Paper Table |
|---|---|---|---|
| `main_grid.csv` | Full 27-configuration $\{model, k, epochs\}$ results across XLM-R, mBERT, and MuRIL | `scripts/04_run_grid.py` | Table 2 & Fig. 3 |
| `ekman6.csv` | Ekman-6 standardized benchmark ($k=1$, 3 epochs) for all three backbones | `scripts/05_run_ekman6.py` | Table 3 |
| `ablation_size_controlled.csv` | Size-controlled ablation isolating orthographic variation from dataset volume | `scripts/06_run_size_controlled_ablation.py` | Table 5 |
| `seed_robustness.csv` | Multi-seed evaluation across seeds 42, 43, and 44 for optimal configurations | `scripts/07_run_seed_robustness.py` | Table 4 |

---

## Verifying All Results

You can reproduce and verify all metrics directly against the paper without retraining by running:

```bash
python scripts/08_compile_results.py --results-dir results
```

This compiles each table and prints a numeric difference (`Diff F1`) against the paper's reported values. All values match with `0.0000` diff.
