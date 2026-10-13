# CV Entry

## Project Entry

**Machine Learning-Based Network Intrusion Detection System** | 2026

- Evaluated four ML classifiers (Logistic Regression, Random Forest, HistGradientBoosting, XGBoost) for binary and multiclass intrusion detection on the CIC-IDS2017 benchmark dataset (2.83M network flow records, 15 traffic classes).
- Implemented a reproducible preprocessing pipeline with strict leakage prevention: normalization fitted exclusively on training data, stratified 80/20 train/test split with fixed random seed.
- Compared four class imbalance strategies (no resampling, class weighting, RandomUnderSampler, SMOTE); found that tree-based models achieve Macro F1 ≥ 0.996 regardless of strategy, while Logistic Regression improved recall from 85.4% to 96.3% under SMOTE at the cost of reduced precision.
- Quantified false positive and false negative rates: XGBoost produced 29 false positives and 25 false negatives on 40,000 test samples (FN rate: 0.38%), versus 965 false negatives for Logistic Regression (FN rate: 14.58%).
- Analyzed feature importance using tree-based Gini gain; identified flow-rate and inter-arrival-time features as most discriminative, consistent with volumetric attack dominance in the dataset.
- Documented five dataset and methodological limitations including dataset age (2017), controlled lab environment, and rare class unlearnable problem (Heartbleed: 11 samples), motivating future research in anomaly detection and concept drift adaptation.
- All code, results, and analysis are reproducible from a fixed random seed and pinned dependency versions.

**Technologies:** Python, scikit-learn, XGBoost, pandas, numpy, matplotlib, seaborn, hypothesis (property-based testing)
