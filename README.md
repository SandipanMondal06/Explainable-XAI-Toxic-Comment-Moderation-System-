# Explainable (XAI) Toxic Comment Moderation System

## Overview
This project implements an explainable toxic comment moderation system using an Artificial Neural Network (ANN).
The focus is on building a transparent NLP pipeline where model decisions are interpretable using SHAP.

Unlike black-box approaches, this system emphasizes explainability, error analysis, and threshold-aware evaluation.

---

## Key Objectives
- Detect toxic comments for automated content moderation
- Preserve feature-level interpretability
- Explain model predictions at global and local levels
- Analyze false positives and false negatives using SHAP

---

## Methodology

### 1. Text Preprocessing
- Lowercasing and noise removal
- Punctuation and digit filtering
- Stopword removal
- Token-preserving cleaning (no embeddings)

### 2. Feature Engineering
- TF-IDF features with controlled vocabulary size
- Auxiliary features (length, profanity indicators, sentiment scores)
- Feature scaling for non-text features
- Sparse + dense feature fusion

### 3. Model Architecture
- Fully connected ANN (Dense layers)
- Dropout-based regularization
- Binary classification with sigmoid output
- Class imbalance handled using class weights

### 4. Evaluation Strategy
- ROC-AUC as primary metric
- Precision–Recall analysis for imbalanced data
- Threshold tuning using ROC and PR curves
- High-threshold configuration for precision-oriented moderation

---

## Explainability (XAI)

SHAP is used extensively to interpret model behavior:

- Global feature importance (bar plots)
- Feature impact distribution (beeswarm plots)
- Dependence plots for key features
- Local force plots for individual predictions
- Separate SHAP analysis for:
  - False Positives
  - False Negatives

This enables detailed understanding of why the model flags or misses certain comments.

---

## Error Analysis
- High-threshold predictions analyzed for moderation reliability
- FP and FN cases extracted and explained using SHAP
- Insights used to identify model blind spots and overconfidence patterns

---

## Tools & Libraries
- Python
- Scikit-learn
- TensorFlow / Keras
- SHAP
- NLTK
- NumPy, Pandas, Matplotlib

---

## Key Takeaways
- Demonstrates practical use of XAI in NLP systems
- Shows responsible evaluation beyond accuracy
- Bridges model performance with interpretability
- Suitable for real-world moderation and compliance use cases

---

## Project Status
Completed  
Further extensions may include robustness testing and bias analysis.
