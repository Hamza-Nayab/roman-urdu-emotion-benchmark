# Roman Urdu Emotion Benchmark

[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](DATA_LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.1.0-red.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/Transformers-4.40.2-orange.svg)](https://huggingface.co/)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18445910-blue)](https://doi.org/10.5281/zenodo.18445910)

Official repository for the paper:
> **"A Reproducible Benchmark for Roman Urdu Emotion Classification via LLM-Based Dataset Construction and Orthographic Variation Modeling"** (*IEEE Access*, 2026).  
> **Authors**: Malik Mir Hamza Nayab, Ghanwa Batool, Zainab Iqbal, Muhammad Zulqrnain.  
> **Zenodo Dataset Record**: [10.5281/zenodo.18445910](https://doi.org/10.5281/zenodo.18445910)

This benchmark establishes the first large-scale, fully reproducible evaluation testbed for emotion classification in Roman Urdu—a low-resource, non-standardized informal orthography spoken by over 70 million individuals across South Asia. We construct a 53,117-instance corpus derived from HuggingFace GoEmotions via automated Gemma 3 translation, deterministic orthographic variation modeling, rigorous QA filtering (identifying 8,619 anomalous patterns), and systematic evaluations across 27 experimental grid configurations (XLM-R, mBERT, and MuRIL), standardizing both 28-class GoEmotions and 6-class Ekman emotion families.

---

## Benchmark Highlights

| Model | Task A: Primary (k=3, 3 ep) Macro-F1 | Task A Test Accuracy | Task B: Ekman-6 (k=1, 3 ep) Macro-F1 | Task B Test Accuracy |
|---|:---:|:---:|:---:|:---:|
| **XLM-RoBERTa (XLM-R)** | **0.2869** | **0.4364** | **0.5004** | **0.7031** |
| **mBERT** | 0.2807* | 0.4406* | 0.4515 | 0.6740 |
| **MuRIL** | 0.2334** | 0.4046** | 0.4009 | 0.6736 |

*\*mBERT achieved 0.2807 at k=3, 2 epochs (0.2673 at 3 epochs). \*\*MuRIL achieved 0.2334 at k=3, 4 epochs.*

---

## 1. Installation

### Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.10 or 3.11
- **Hardware**: CUDA-compatible GPU (e.g. 2× NVIDIA Tesla T4 or A100) recommended for model training. Verification and compilation require only standard CPU.

```bash
# Clone the repository
git clone https://github.com/hamzanayab/roman-urdu-emotion-benchmark.git
cd roman-urdu-emotion-benchmark

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify test suite
PYTHONPATH="src" pytest tests -v
```

### (Optional) Translation Prerequisites
If you intend to regenerate translations from scratch rather than downloading pre-translated splits:
1. Install [Ollama](https://ollama.ai/).
2. Pull the Gemma 3 model:
   ```bash
   ollama pull gemma3:latest
   ```

---

## 2. Quickstart: Instant Verification vs. GPU Reproduction

### Option A: Instant Verification (No GPU, < 5 seconds)
Every number, metric, and table published in the paper can be verified directly from the pre-computed run artifacts in `results/`:

```bash
python scripts/08_compile_results.py --results-dir results
```

This will compile Table 2 (Grid Headlines), Table 3 (Ekman-6 Standardized), Table 4 (Multi-Seed Robustness), and Table 5 (Size-Controlled Ablation), outputting numeric diffs against paper citations (`Diff F1 = 0.0000`).

### Option B: Reproducing the Headline Result (XLM-R, k=3, 3 epochs)
> [!NOTE]
> Training a full 164,857-instance view on XLM-R for 3 epochs requires ~2–3 hours on an NVIDIA T4 GPU.

```bash
# 1. Prepare data and canonical splits
python scripts/01_download_goemotions.py
python scripts/03_freeze_splits.py

# 2. Train XLM-R under the optimal configuration
python src/romanurdu/train.py \
    --model xlmr \
    --task primary \
    --k 3 \
    --epochs 3 \
    --seed 42 \
    --source-csv data/goemotions_roman_ur_firstlabel.csv \
    --split-dir data/splits \
    --output-dir outputs/quickstart
```
The resulting run directory will contain `test_metrics.json` confirming **Test Macro-F1: 0.2869**.

---

## 3. Repository Structure

```
roman-urdu-emotion-benchmark/
├── README.md                      # Primary project documentation
├── LICENSE                        # MIT License for code
├── DATA_LICENSE                   # CC BY 4.0 notice for dataset
├── CITATION.cff                   # Citation metadata
├── requirements.txt               # Pinned dependencies
├── environment.yml                # Conda environment definition
├── .gitignore
├── configs/
│   ├── paths.yaml                 # Configurable dataset and output paths
│   └── hparams.yaml               # Hyperparameters and model settings
├── src/
│   └── romanurdu/
│       ├── __init__.py
│       ├── variants.py            # Deterministic spelling variant generator
│       ├── qa_flagging.py         # QA heuristics (identifies 8,619 flagged rows)
│       ├── translate.py           # Gemma 3 translation interface
│       ├── splits.py              # Canonical 42,493 / 5,312 / 5,312 split freezer
│       ├── views.py               # Single build_view() entry point (k-controlled)
│       ├── ekman.py               # Ekman-6 mapping & transformation logic
│       ├── baselines.py           # Majority-class and random-uniform floors
│       ├── evaluate.py            # Metric computation & artifact export
│       └── train.py               # Unified training pipeline with skip logic
├── scripts/
│   ├── 01_download_goemotions.py  # HuggingFace GoEmotions ingestion
│   ├── 02_translate.py            # Batch translation & QA flagging
│   ├── 03_freeze_splits.py        # Split freezing script
│   ├── 04_run_grid.py             # 27-run benchmark grid runner
│   ├── 05_run_ekman6.py           # Ekman-6 standardized runner
│   ├── 06_run_size_controlled_ablation.py # Size-controlled ablation runner
│   ├── 07_run_seed_robustness.py  # Multi-seed stability runner
│   ├── 08_compile_results.py      # Results compiler & paper diff checker
│   ├── rehydrate.py               # Rehydrates paired English text from HuggingFace
│   └── validate_release.py        # Validates split leakage and dataset integrity
├── tests/
│   ├── test_variants_deterministic.py
│   ├── test_splits_disjoint.py
│   ├── test_split_sizes.py
│   ├── test_views_no_length_leak.py
│   ├── test_train_skip_logic.py
│   └── test_baselines.py
├── results/                       # Verified benchmark result tables
│   ├── main_grid.csv
│   ├── ekman6.csv
│   ├── ablation_size_controlled.csv
│   ├── seed_robustness.csv
│   └── README.md
└── docs/
    └── REPRODUCING.md             # Complete reproduction walkthrough
```

---

## 4. Key Fixes & Design Principles

This codebase resolves several defects present in earlier prototype scripts:
1. **Deterministic Variant Selection**: Replaced non-deterministic Python `set` iteration with deterministic alphabetical ordering before truncation to `k`, eliminating hash-seed dependency across platforms.
2. **Unified Frozen Split**: Removed the confusing `OFF` vs `ON` directory duplication. The repository defines **one canonical frozen split** (`train.csv`: 42,493, `val.csv`: 5,312, `test.csv`: 5,312), and variant admission is strictly controlled at view-time via `--k {0, 1, 3}`.
3. **One-Row Discrepancy Resolved**: Formally excludes the defective legacy row (`validation_004943`), guaranteeing exact match with paper numbers and protected by automated unit tests.
4. **95% Code De-duplication**: Replaced redundant Kaggle notebooks with a single parameterized `train.py` driving all models, tasks, seeds, and budgets.
5. **Decoupled Configuration**: Eliminated hardcoded `/kaggle/...` paths in favor of `configs/paths.yaml` and CLI overrides.

---

## 5. Known Limitations of this Codebase

As detailed in Section V of the paper, researchers should consider the following intrinsic constraints:
1. **Perturbation Noise Floor (~0.01 Macro-F1)**: As demonstrated in our test-set perturbation study, natural spelling variations introduce an empirical noise floor of approximately $\pm 0.01$ Macro-F1. Variations smaller than this threshold should not be treated as decisive model superiority.
2. **MuRIL Optimization Instability**: MuRIL exhibits significant sensitivity to learning rate and warmup under the standardized hyperparameter regime ($\text{LR}=2\times10^{-5}$), resulting in delayed convergence at early epochs.
3. **Non-Nested Task Preprocessing**: Primary 27-class GoEmotions training filters token lengths to $[2, 80]$ tokens, whereas Ekman-6 direct training omits this length filter. This behavior is preserved via the `--apply-length-filter` toggle to faithfully replicate both experimental setups.
4. **QA-Flagged Instances**: The 8,619 flagged instances (hallucinated entities, length anomalies, sanitized profanity) are retained in the released dataset to preserve authentic informal social media distributions. They can be inspected or regenerated via `src/romanurdu/qa_flagging.py`.

---

## 6. Data Availability & Licenses

- **Code License**: The software in `src/`, `scripts/`, and `tests/` is licensed under the [MIT License](LICENSE).
- **Dataset License**: The Roman Urdu dataset and its derived representations are distributed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](DATA_LICENSE).
- **GoEmotions Source Attribution**: The underlying English GoEmotions text is separately copyrighted by Google LLC and licensed under the [Apache License 2.0](https://github.com/google-research/google-research/tree/master/goemotions). English text is not redistributed in pre-packaged binaries and is downloaded directly via HuggingFace.
- **Zenodo Release**: Rehydratable data splits and replication software are archived at [DOI: 10.5281/zenodo.18445910](https://doi.org/10.5281/zenodo.18445910).

---

## 7. Citation

If you use this benchmark codebase, models, or dataset in your research, please cite:

```bibtex
@article{nayab2026romanurdu,
  author    = {Nayab, Malik Mir Hamza and Batool, Ghanwa and Iqbal, Zainab and Zulqrnain, Muhammad},
  title     = {A Reproducible Benchmark for Roman Urdu Emotion Classification via LLM-Based Dataset Construction and Orthographic Variation Modeling},
  journal   = {IEEE Access},
  year      = {2026},
  publisher = {IEEE}
}

@misc{romanurdu_zenodo_2026,
  author       = {Nayab, Malik Mir Hamza and Iqbal, Zainab and Zulqrnain, Muhammad and Batool, Ghanwa},
  title        = {A Reproducible {R}oman {U}rdu Emotion Benchmark Derived from {GoEmotions}},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18445910},
  howpublished = {Dataset, \url{https://doi.org/10.5281/zenodo.18445910}}
}
```
Matching citation metadata is provided in [CITATION.cff](CITATION.cff).
