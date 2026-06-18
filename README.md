# 🔋 Battery RUL Prediction Framework

## Reproducible Machine Learning Pipeline for Lithium-Ion Battery Remaining Useful Life (RUL)

This repository provides a fully reproducible machine learning framework for predicting the Remaining Useful Life (RUL) of lithium-ion batteries using heterogeneous public datasets.

---

## 📌 Key Contributions

- Unified pipeline for 8 public battery datasets
- Phase-level modeling of charge and discharge cycles
- Vector-aware feature extraction (V(Q), Q(V))
- Robust cross-dataset evaluation:
  - Global training
  - Per-dataset evaluation
  - Leave-One-Dataset-Out (LODO)
- Machine learning models:
  - Random Forest
  - Extra Trees
  - Elastic Net
  - MLP
  - HistGradientBoosting

---

## ⚙️ Pipeline Overview

Raw Data → Dataset Adapters → Cycle Cleaning → Phase-Level Feature Extraction → Modeling Dataset Construction → HPC Campaign → Metrics Aggregation

---

## 📂 Repository Structure

- src/ → Core pipeline
- scripts/ → Execution entry points
- configs/ → Experiment configuration files
- docs/ → Technical documentation
- battery_rul_modeling_final_campaign_hpc/ → Final campaign

---

## 🚀 Quick Start

pip install -r requirements.txt

python scripts/run_adapter.py --dataset matr --stage all
python scripts/run_feature_extraction_v3.py --fast-mode --no-stages
python scripts/run_modeling_dataframe.py

---

## ⚠️ Important Notes

- No stage-level extraction in final campaign
- Energy represented via derived trajectory features (no explicit E(Q))
- Trained models are not included
- Raw datasets excluded due to licensing

---

## 📜 Citation

@software{galdino2026battery_rul,
  author = {Galdino, Giovanny},
  title = {Battery RUL Prediction Framework},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/GiovannyGaldino/Battery_RUL_Prediction}
}
