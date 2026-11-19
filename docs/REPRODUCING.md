# Reproduction Guide

This document details the exact, copy-pasteable commands required to reproduce all experiments, tables, and figures reported in:

> **A Reproducible Benchmark for Roman Urdu Emotion Classification via LLM-Based Dataset Construction and Orthographic Variation Modeling** (*IEEE Access*)

---

## 1. Overview and Estimated Wall Times

| Stage | Command | Hardware Requirements | Estimated Wall Time |
|---|---|---|---|
| **0. Instant Verification** | `python scripts/08_compile_results.py` | Standard CPU | < 5 seconds |
| **1. Dataset Ingestion** | `python scripts/01_download_goemotions.py` | CPU, Internet | ~1 minute |
| **2. LLM Translation** | `python scripts/02_translate.py` | Ollama + Gemma 3 (`gemma3:latest`) | ~24–48 hours (slowest stage) |
| **3. Freeze Canonical Splits** | `python scripts/03_freeze_splits.py` | CPU | < 10 seconds |
| **4. Benchmark Grid (27 runs)** | `python scripts/04_run_grid.py` | GPU (e.g. 2× NVIDIA T4, 16GB) | ~30–45 GPU hours total |
| **5. Ekman-6 Standardization** | `python scripts/05_run_ekman6.py` | GPU | ~3–4 GPU hours |
| **6. Size-Controlled Ablation** | `python scripts/06_run_size_controlled_ablation.py` | GPU | ~2 GPU hours |
| **7. Multi-Seed Robustness** | `python scripts/07_run_seed_robustness.py` | GPU | ~6–8 GPU hours |

> [!NOTE]
> If you simply want to verify the paper's reported numbers without re-running 50+ hours of GPU training, you can skip directly to **Section 2** below, which compiles the verified run artifacts shipped in `results/`.

---

## 2. Instant Verification (Zero GPU Required)

Run the results compiler directly on the pre-computed benchmark outputs:

```bash
# Verify Table 2 (Grid), Table 3 (Ekman-6), Table 4 (Seeds), Table 5 (Ablation)
python scripts/08_compile_results.py --results-dir results
```

This will print the full benchmark tables along with a `Diff F1` column confirming exact alignment with the published paper (`0.0000` difference).

---

## 3. Environment Setup

```bash
# Clone the repository
git clone https://github.com/hamzanayab/roman-urdu-emotion-benchmark.git
cd roman-urdu-emotion-benchmark

# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the unit test suite to verify installation
PYTHONPATH="src" pytest tests -v
```

---

## 4. End-to-End Reproduction Steps

### Step 1: Ingest English GoEmotions
Downloads the HuggingFace GoEmotions `simplified` configuration and extracts primary emotion labels:

```bash
python scripts/01_download_goemotions.py --output-csv data/goemotions_en_all_firstlabel.csv
```

### Step 2: English -> Roman Urdu Translation & QA Flagging
Requires [Ollama](https://ollama.ai/) installed and running locally with the Gemma 3 model pulled:

```bash
# In a separate terminal or background daemon:
ollama pull gemma3:latest

# Run the incremental, chunked translation pipeline (100 rows per batch):
python scripts/02_translate.py \
    --source-csv data/goemotions_en_all_firstlabel.csv \
    --output-csv data/goemotions_roman_ur_firstlabel.csv \
    --flagged-csv data/goemotions_roman_ur_flagged.csv \
    --chunk-size 100 \
    --timeout 10
```

> [!TIP]
> This stage automatically saves every 100 rows and safely resumes if interrupted. If you already downloaded the rehydratable dataset from Zenodo, you can place `goemotions_roman_ur_firstlabel.csv` into `data/` directly and skip this step. To rehydrate the paired English source text locally from Hugging Face into the dataset splits, run `python scripts/rehydrate.py --release_dir .`.

### Step 3: Freeze Canonical Splits
Enforces the canonical split sizes (42,493 train / 5,312 val / 5,312 test) and guarantees zero data leakage:

```bash
python scripts/03_freeze_splits.py \
    --source-csv data/goemotions_roman_ur_firstlabel.csv \
    --output-dir data/splits
```

### Step 4: Run Primary Benchmark Grid (27 Configurations)
Runs all combinations of backbones (`xlmr`, `mbert`, `muril`), variants ($k \in \{0, 1, 3\}$), and epochs ($e \in \{2, 3, 4\}$). This execution is **idempotent**: if any configuration's 4 artifact files already exist, it is skipped.

```bash
# Run all 27 runs
python scripts/04_run_grid.py \
    --source-csv data/goemotions_roman_ur_firstlabel.csv \
    --split-dir data/splits \
    --output-dir outputs/grid

# Or run an individual backbone (e.g. XLM-R):
python scripts/04_run_grid.py --model xlmr --output-dir outputs/grid
```

### Step 5: Run Ekman-6 Standardized Benchmark
Trains XLM-R, mBERT, and MuRIL under the standardized protocol ($k=1$ variant, 3 epochs, early stopping patience 1):

```bash
python scripts/05_run_ekman6.py \
    --source-csv data/goemotions_roman_ur_firstlabel.csv \
    --split-dir data/splits \
    --output-dir outputs/ekman6
```

### Step 6: Run Size-Controlled Ablation
Isolates whether $k=3$ gains come from orthographic diversity or dataset size by subsampling the $k=3$ view to 42,493 rows:

```bash
python scripts/06_run_size_controlled_ablation.py \
    --source-csv data/goemotions_roman_ur_firstlabel.csv \
    --split-dir data/splits \
    --output-dir outputs/ablation_size_controlled
```

### Step 7: Run Multi-Seed Robustness
Quantifies initialization sensitivity across seeds 42, 43, and 44:

```bash
python scripts/07_run_seed_robustness.py \
    --source-csv data/goemotions_roman_ur_firstlabel.csv \
    --split-dir data/splits \
    --output-dir outputs/seed_robustness
```

### Step 8: Compile Run Results and Diff Check
Compiles your newly produced runs in `outputs/` or shipped runs in `results/`:

```bash
python scripts/08_compile_results.py --results-dir outputs
```
