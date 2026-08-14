# Explainability-Driven Analysis of Transformer Ensemble Calibration Failure in Indonesian Social Media Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.43%2B-orange)](https://github.com/shap/shap)
[![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-f37626?logo=jupyter&logoColor=white)](notebooks/)
[![Dataset](https://img.shields.io/badge/Dataset-IndoNLU%20SmSA-6a5acd)](https://github.com/indobenchmark/indonlu)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Conference](https://img.shields.io/badge/Conference-ICIMTECH%202026-lightgrey)](paper/)

> Empirical analysis and explainability-driven diagnosis showing that naïve stacking of heterogeneous transformer and non-neural base learners degrades ensemble performance through probability calibration mismatch — identified via SHAP attribution on the meta-learner.

---

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Repository Structure](#repository-structure)
- [Directory Guide](#directory-guide)
- [Experimental Pipeline](#experimental-pipeline)
- [Main Results](#main-results)
- [Installation](#installation)
- [Reproducibility](#reproducibility)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This repository accompanies the paper *"Explainability-Driven Analysis of Transformer Ensemble Calibration Failure in Indonesian Social Media Sentiment Analysis"* (ICIMTECH). The work investigates a systematic performance degradation that emerges when stacking IndoBERT, IndoBERTweet, and a TF-IDF SVM on the **IndoNLU SmSA** benchmark — 12,760 Indonesian Twitter sentences labeled with three sentiment classes.

The central finding is that the full stacking ensemble scores only F1-Macro 0.8299, while IndoBERTweet alone achieves 0.9015 — a **7.16-point degradation**. SHAP analysis of the Logistic Regression meta-learner identifies the root cause: Platt-scaled SVM probabilities exhibit a bimodal distribution with variance approximately 3.1× higher than IndoBERT and 4.8× higher than IndoBERTweet, causing SVM features to dominate 60.3% of meta-learner attribution despite the SVM being the weakest base learner (F1-Macro 0.741). This calibration mismatch is invisible without post-hoc interpretability and provides a replicable diagnostic template for heterogeneous ensemble failures in low-resource NLP settings.

---

## Key Contributions

- **Calibration failure diagnosis.** Empirical demonstration that heterogeneous stacking degrades performance when base learners produce probability outputs with incompatible distributional properties.
- **SHAP-based meta-learner attribution.** Application of SHAP LinearExplainer at the ensemble input level — where features are base learner probability vectors — to quantify each component's marginal contribution to the meta-learner's class decisions.
- **Calibration distribution analysis.** Variance analysis of model output distributions reveals SVM probabilities are 3.1× (vs. IndoBERT) and 4.8× (vs. IndoBERTweet) more extreme, explaining coefficient-space dominance under logistic regression optimization.
- **Ablation evidence.** Controlled one-at-a-time removal of base learners confirms the SVM as the primary degradation source; removing it recovers 6.46 F1-Macro points (0.8299 → 0.8945).
- **Failure case analysis.** Inspection of 61 misclassified test samples identifies three systematic error patterns: sarcasm, neutral boundary confusion exacerbated by class imbalance, and code-mixed text.
- **Indonesian NLP benchmark contribution.** A complete experimental record on IndoNLU SmSA including efficiency metrics, per-class analysis, token-level attribution, and failure case annotations.
- **Transferable methodology.** The SHAP-based ensemble diagnostic framework is model-agnostic and applicable to any heterogeneous stacking setup beyond the Indonesian NLP context.

---

## Repository Structure

```text
.
├── README.md                     # This file
├── LICENSE                       # MIT license
├── requirements.txt              # Python dependencies
├── .gitignore                    # Ignore rules for archives, checkpoints, and caches
│
├── notebooks/                    # Executable research pipeline
│   ├── final_code_explainable_transformer_experiment.ipynb   # Main end-to-end experiment
│   └── generate_updated_figures.ipynb                        # Standalone figure regeneration
│
├── outputs/                      # All generated research artifacts (git-tracked where lightweight)
│   ├── Diagram.png                                      # Stacking ensemble architecture diagram
│   ├── figures/                                         # Publication figures (fig1–fig8)
│   ├── metrics/                                         # CSV files: model metrics, ablation, per-class, SHAP, efficiency
│   ├── models/                                          # Saved model weights, tokenizer configs, meta-learner
│   │   ├── indobert/
│   │   ├── indobertweet/
│   │   ├── probas/                                      # Saved base learner probability arrays (.npy)
│   │   ├── meta_learner.pkl
│   │   ├── tfidf.pkl
│   │   ├── svm_clf.pkl
│   │   └── timing.json
│   └── predictions/
│       ├── test_predictions.csv
│       └── failure_cases.csv
│
├── paper/                        # Everything related to the written paper
│   ├── paper.tex                                        # LaTeX source of the conference paper
│   ├── submission/                                      # Camera-ready PDF, submitted .tex, reviewer replies
│   │   └── rev-1/                                       # Round-1 revision package
│   └── turnitin/                                        # Plagiarism / AI-detection compliance reports
│       └── archived/                                    # Superseded similarity reports
│
├── tasks/                        # Internal task prompts used to direct iterative repo work
│                                  # (drafting, revision, restructuring instructions — not part of the
│                                  #  research pipeline itself)
│
└── archive/                      # Preserved legacy / non-core artifacts (excluded from Git)
    ├── notebooks/                                       # Superseded experiment notebook
    ├── documents/                                       # Old paper drafts and PDF versions
    └── experiments/                                     # Early standalone experiment scripts
```

---

## Directory Guide

| Directory | Purpose | Tracked by Git? |
|---|---|---|
| `notebooks/` | The executable research pipeline. `final_code_explainable_transformer_experiment.ipynb` runs the full pipeline (preprocessing → training → stacking → SHAP → ablation → failure analysis) end to end and is fully resumable via checkpoints in `outputs/`. `generate_updated_figures.ipynb` regenerates Figures 2, 3, and 8 only, from already-saved CSVs, without retraining. | Yes |
| `outputs/` | Every artifact that can be *regenerated* by rerunning the notebooks: figures, metrics tables, predictions, and model checkpoints. Lightweight files (figures, metrics, tokenizer configs) are tracked; large binaries (`.pt`, `.bin`, `.safetensors`, `.pkl`, `.npy`) are git-ignored — see [Reproducibility](#reproducibility). | Partially (see `.gitignore`) |
| `paper/` | The paper itself and everything tied to its submission: LaTeX source, the camera-ready/submitted packages, reviewer correspondence, and Turnitin/AI-detection compliance reports. | No (git-ignored; kept locally for reference) |
| `tasks/` | Free-form instruction files used to direct AI-assisted work on this repository over time (paper revisions, README/figure upgrades, this restructuring). Not part of the experimental pipeline — safe to ignore if you're only interested in reproducing the research. | No (git-ignored) |
| `archive/` | Superseded material kept for historical reference instead of being deleted: an earlier experiment notebook, older paper drafts, and standalone scripts from before the pipeline was consolidated into `notebooks/`. Nothing here is required to reproduce the reported results. | No (git-ignored) |

**Rule of thumb:** if a file can be recreated by rerunning the notebooks, it belongs in `outputs/`. If it documents or supports the paper itself, it belongs in `paper/`. Anything superseded goes to `archive/` rather than being deleted.

---

## Experimental Pipeline

The experiment proceeds in five stages:

1. **Dataset preprocessing.** The IndoNLU SmSA corpus (11,000 / 1,260 / 500 train/val/test) is preprocessed with URL and mention removal, hashtag normalization, lowercasing, and slang normalization via a 3,000-entry colloquial dictionary. SVM inputs additionally undergo Indonesian stop-word removal and PySastrawi lemmatization before TF-IDF extraction. Transformer inputs are tokenized directly using model-specific tokenizers (max length 128), as both IndoBERT and IndoBERTweet handle informal text natively from pre-training.

2. **Base learner training.** IndoBERT and IndoBERTweet are each fine-tuned for 3 epochs (lr = 2×10⁻⁵, batch size 32, AdamW with 10% linear warmup, weight decay 0.01) on a single NVIDIA T4 GPU in Google Colab. The TF-IDF SVM uses unigram/bigram features (50,000 vocabulary, sublinear TF) with Platt scaling for probability output.

3. **Stacking meta-learner.** A two-level stacking architecture concatenates the three base learners' three-class probability vectors into a 9-dimensional feature space. A Logistic Regression meta-learner (L2, C=1.0) is trained on held-out validation predictions to prevent target leakage — a common source of inflated stacking performance in reported benchmarks.

4. **Explainability analysis.** SHAP LinearExplainer is applied to the meta-learner with a 100-sample background distribution. Because the meta-learner is linear, SHAP values are analytically exact. Variance analysis of model output distributions characterizes the calibration mismatch quantitatively. Token-level importance is additionally extracted from both transformer models for qualitative comparison.

5. **Calibration, ablation, and failure analysis.** Per-class and macro F1 are computed across all configurations. One-at-a-time ablations isolate each base learner's contribution. Sixty-one misclassified test samples are inspected for systematic error patterns, with instance-level SHAP attribution to identify which base learner drives each failure.

---

## Main Results

### Model Performance on IndoNLU SmSA Test Set (500 samples)

| Model | F1-Macro | Accuracy | F1-Neg | F1-Neu | F1-Pos |
|---|---|---|---|---|---|
| TF-IDF + SVM | 0.7410 | 0.8040 | 0.8557 | 0.5255 | 0.8416 |
| IndoBERT | 0.8944 | 0.9200 | 0.9375 | 0.7975 | 0.9484 |
| **IndoBERTweet** | **0.9015** | **0.9280** | **0.9547** | **0.8054** | 0.9444 |
| Full Ensemble | 0.8299 | 0.8820 | 0.9298 | 0.6438 | **0.9161** |

### Ablation Results

| Configuration | F1-Macro | Accuracy |
|---|---|---|
| Full Ensemble | 0.8299 | 0.8820 |
| Ensemble w/o SVM | 0.8945 | 0.9200 |
| Ensemble w/o IndoBERTweet | 0.8243 | 0.8760 |
| Ensemble w/o IndoBERT | 0.8120 | 0.8620 |

### SHAP Attribution by Model Group

| Base Learner | F1-Neg SHAP | F1-Neu SHAP | F1-Pos SHAP | Attribution Share |
|---|---|---|---|---|
| IndoBERT | 0.209 | 0.038 | 0.234 | 28.4% |
| IndoBERTweet | 0.075 | 0.021 | 0.094 | 11.2% |
| **TF-IDF SVM** | **0.353** | **0.229** | **0.438** | **60.3%** |

### Efficiency Analysis

| Model | Params (M) | Latency (ms/sample) | GPU Mem (MB) | F1-Macro |
|---|---|---|---|---|
| TF-IDF + SVM | 0.05 | 0.008 | 0 | 0.741 |
| IndoBERT | 124.4 | 8.04 | 1,312 | 0.894 |
| IndoBERTweet | 110.6 | 8.21 | 1,312 | **0.902** |
| Full Ensemble | 235.1 | 16.26 | 2,624 | 0.830 |

**Key finding.** The full ensemble doubles compute cost and memory over any single transformer while scoring 7.16 F1-Macro points below IndoBERTweet. SHAP reveals SVM dominance (60.3% attribution) arises not from predictive strength but from extreme Platt-scaled confidence values with variance 3.1–4.8× higher than the transformer outputs — a structural consequence of combining heterogeneous model families without calibration alignment.

---

## Installation

**Requirements:** Python 3.9+, CUDA-capable GPU recommended for transformer fine-tuning.

```bash
git clone https://github.com/ResearchSpirit/transformer-calibration-analysis
cd transformer-calibration-analysis
pip install -r requirements.txt
```

Run the main experiment notebook (executes the full pipeline; resumable via `outputs/` checkpoints):

```bash
jupyter notebook notebooks/final_code_explainable_transformer_experiment.ipynb
```

To regenerate Figures 2, 3, and 8 only, from already-saved CSVs (no retraining):

```bash
jupyter notebook notebooks/generate_updated_figures.ipynb
```

> Both notebooks read and write paths relative to the **repository root** (e.g. `outputs/figures/...`). Launch Jupyter from the repository root — not from inside `notebooks/` — so these relative paths resolve correctly.

---

## Reproducibility

All experiments use fixed random seeds across PyTorch, NumPy, and scikit-learn. Transformer fine-tuning hyperparameters are held constant across IndoBERT and IndoBERTweet to isolate the effect of pre-training domain rather than optimization differences. Saved model weights, probability arrays (`.npy`), and the fitted meta-learner are provided in `outputs/models/` so downstream evaluation and SHAP analysis can be reproduced without re-training.

The meta-learner is trained exclusively on held-out validation predictions — not training-set predictions — preventing target leakage and ensuring reported metrics reflect out-of-distribution generalization. All metrics reported in the paper are computed directly from `outputs/metrics/` CSV files for full traceability.

Large binary artifacts (fine-tuned transformer weights `.safetensors`, the SVM/meta-learner `.pkl` files, and cached probability `.npy` arrays) are excluded from Git via `.gitignore` to keep the repository lightweight — see [`outputs/models/`](outputs/models/) locally after running the pipeline, or rerun `notebooks/final_code_explainable_transformer_experiment.ipynb` to regenerate them from scratch. Lightweight, directly citable artifacts (figures, metrics CSVs, tokenizer configs, predictions) remain version-controlled.

---

## Documentation

- **This README** covers the research overview, results, and how to run the pipeline.
- **[Directory Guide](#directory-guide)** above explains what each top-level folder contains and why.
- **`paper/paper.tex`** is the authoritative LaTeX source of the paper; `paper/submission/` contains the camera-ready and revision packages actually sent to the conference.
- **In-notebook documentation.** `notebooks/final_code_explainable_transformer_experiment.ipynb` contains a "Paper → Output Mapping" section at the end that maps every reported table and figure to its exact source file in `outputs/`.
- **`tasks/`** documents the process behind non-research repository work (paper revisions, this restructuring) for anyone curious how the repository evolved; it is not required reading to reproduce the experiments.

---

## Citation

If you use this code or find the diagnostic methodology useful, please cite:

```bibtex
@inproceedings{wijaya2026explainability,
  title     = {Explainability-Driven Analysis of Transformer Ensemble Calibration Failure
               in Indonesian Social Media Sentiment Analysis},
  author    = {Wijaya, Stanley Nathanael and Anderies},
  booktitle = {Proceedings of the International Conference on Informatics, Multimedia,
               and Technology (ICIMTECH)},
  year      = {2026},
  address   = {Jakarta, Indonesia}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

This work builds on the Indonesian NLP ecosystem, particularly the [IndoNLU benchmark](https://github.com/indobenchmark/indonlu) and the pre-trained models released by the Indonesian NLP research community. We thank the [Hugging Face](https://huggingface.co/) team for the Transformers library and model hub, the [PyTorch](https://pytorch.org/) team, the authors of [SHAP](https://github.com/shap/shap) for the interpretability tools central to this analysis, and the NusaCrowd initiative for aggregating Indonesian NLP resources.

Latest update: 14/08/2026 (GMT+7)
