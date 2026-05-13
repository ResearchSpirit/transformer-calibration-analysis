# Explainability-Driven Analysis of Transformer Ensemble Calibration Failure in Indonesian Social Media Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![SHAP](https://img.shields.io/badge/SHAP-0.43%2B-orange)](https://github.com/shap/shap)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Conference](https://img.shields.io/badge/Conference-ICIMTECH-lightgrey)]()

> Empirical analysis and explainability-driven diagnosis showing that naïve stacking of heterogeneous transformer and non-neural base learners degrades ensemble performance through probability calibration mismatch — identified via SHAP attribution on the meta-learner.

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

```
.
├── paper.tex                                            # LaTeX source of the conference paper
├── requirements.txt                                     # Python dependencies
├── final_code_explainable_transformer_experiment.ipynb  # Main experiment notebook
├── generate_updated_figures.ipynb                       # Figure regeneration notebook
├── outputs/
│   ├── figures/                                         # All publication figures (fig1–fig8)
│   ├── metrics/                                         # CSV files: model metrics, ablation, per-class, SHAP, efficiency
│   ├── models/                                          # Saved model weights and tokenizer configs
│   │   ├── indobert/
│   │   ├── indobertweet/
│   │   ├── probas/                                      # Saved base learner probability arrays (.npy)
│   │   ├── meta_learner.pkl
│   │   ├── tfidf.pkl
│   │   └── svm_clf.pkl
│   ├── predictions/
│   │   ├── test_predictions.csv
│   │   └── failure_cases.csv
│   └── Diagram.png                                      # Stacking ensemble architecture diagram
```

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

Run the main experiment notebook:

```bash
jupyter notebook final_code_explainable_transformer_experiment.ipynb
```

To regenerate updated figures independently:

```bash
jupyter notebook generate_updated_figures.ipynb
```

---

## Reproducibility

All experiments use fixed random seeds across PyTorch, NumPy, and scikit-learn. Transformer fine-tuning hyperparameters are held constant across IndoBERT and IndoBERTweet to isolate the effect of pre-training domain rather than optimization differences. Saved model weights, probability arrays (`.npy`), and the fitted meta-learner are provided in `outputs/models/` so downstream evaluation and SHAP analysis can be reproduced without re-training.

The meta-learner is trained exclusively on held-out validation predictions — not training-set predictions — preventing target leakage and ensuring reported metrics reflect out-of-distribution generalization. All metrics reported in the paper are computed directly from `outputs/metrics/` CSV files for full traceability.

---

## Citation

If you use this code or find the diagnostic methodology useful, please cite:

```bibtex
@inproceedings{wijaya2025explainability,
  title     = {Explainability-Driven Analysis of Transformer Ensemble Calibration Failure
               in Indonesian Social Media Sentiment Analysis},
  author    = {Wijaya, Stanley Nathanael and Anderies},
  booktitle = {Proceedings of the International Conference on Informatics, Multimedia,
               and Technology (ICIMTECH)},
  year      = {2025},
  address   = {Jakarta, Indonesia}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

This work builds on the Indonesian NLP ecosystem, particularly the [IndoNLU benchmark](https://github.com/indobenchmark/indonlu) and the pre-trained models released by the Indonesian NLP research community. We thank the [Hugging Face](https://huggingface.co/) team for the Transformers library and model hub, the [PyTorch](https://pytorch.org/) team, the authors of [SHAP](https://github.com/shap/shap) for the interpretability tools central to this analysis, and the NusaCrowd initiative for aggregating Indonesian NLP resources.

Latest update: 13/05/2026 - 21:36 (GMT+7)