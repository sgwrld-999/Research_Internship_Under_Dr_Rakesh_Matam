# 🗂️ /data Directory Convention

This project follows a modular data lifecycle pattern to ensure reproducibility and clarity.

```
/data/
  ├── raw/         # Untracked original datasets
  ├── interim/     # Temporary or partially processed data (cache)
  ├── processed/   # Clean, model-ready datasets
  └── external/    # Public or third-party reference data
```

---

###  `/data/raw/` — Raw, Untracked Inputs

*   **What:** Original datasets (as downloaded or received), *never modified*.
*   **Examples:** `ciciot_raw/`, `edgeiiot_raw/`.
*   **Tracking:** Not versioned in Git (due to size/licensing). Documented via `/data/DATASETS.md`.
*   **Rule:** Only read from this directory; never write or overwrite.

---

###  `/data/interim/` — Intermediate Artifacts (Temporary)

*   **What:** *Transient or cache* data produced during preprocessing or experiments. Often too large or too volatile for Git tracking.
*   **Examples:**
    *   Encoded or partially cleaned CSVs (before normalization).
    *   Cached label co-occurrence matrices (`C.pkl`, `J.npy`).
    *   PCA or feature-weight arrays (`weights_temp.npy`).
    *   Fold-wise split indices during cross-validation.
*   **Lifecycle:** Generated during preprocessing, reused within the same environment, and can be safely deleted or regenerated.
*   **Typical writers:** Scripts in `src/preprocessing/` and `src/features/`.

---

###  `/data/processed/` — Final Model-Ready Data

*   **What:** Fully preprocessed datasets (scaled, encoded, cleaned) ready for model training and evaluation.
*   **Examples:**
    *   `train_fold_1.csv`, `test_fold_1.csv`.
    *   Normalized tabular files.
    *   Fold-wise label splits.
*   **Tracking:** Optionally version-controlled (if small); otherwise, re-generated from scripts.

---

###  `/data/external/` — Public or Reference Data

*   **What:** External or benchmark data **not directly used in training**, but useful for validation, comparisons, or metadata enrichment.
*   **Examples:**
    *   Pre-trained word/graph embeddings.
    *   Label taxonomies (e.g., attack ontology).
    *   Public feature ranking references.
*   **Usage in your pipeline:**
    *   Could be used to load semantic embeddings for distance calculations or for mapping features.

---

###  In Short

| Directory          | Purpose                       | Typical Contents                      | Git-tracked? |
| ------------------ | ----------------------------- | ------------------------------------- | ------------ |
| `/data/raw/`       | Original datasets (untouched) | CSVs, PCAPs                           | ❌ No         |
| `/data/interim/`   | Intermediate caches           | Encoded, partials, PCA, co-occurrence | ❌ No         |
| `/data/processed/` | Final model-ready data        | train/test splits, normalized files   | ⚙️ Optional  |
| `/data/external/`  | Public or reference data      | embeddings, ontologies                | ⚙️ Optional  |
