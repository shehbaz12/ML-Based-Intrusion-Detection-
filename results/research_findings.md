# Research Findings: ML-Based Network Intrusion Detection

**Dataset:** CIC-IDS2017 (Canadian Institute for Cybersecurity, University of New Brunswick)  
**Experiment Date:** October 2026  
**Subset:** 200,000-row stratified sample (random_state=42) from 2,574,059 clean rows  
**Train/Test Split:** 80/20 stratified, random_state=42  
**Feature Count:** 76 (after dropping Destination Port and duplicate Fwd Header Length.1)

---

## 1. Best-Performing Model

**⚠️ HIGH PERFORMANCE FLAG**: Multiple models exceeded macro F1 ≥ 0.99 on the binary task. Per the design specification, this triggers a caution note:

> CIC-IDS2017 is known to have high intra-class feature similarity and was generated in a controlled lab environment. These near-perfect results should **not** be interpreted as evidence that the system would perform similarly on real-world enterprise traffic. Dataset construction artifacts likely contribute to the high separability of classes.

### Binary Classification Results (BENIGN vs. ATTACK)

| Rank | Model | Strategy | Accuracy | Precision | Recall | F1 | Macro F1 | ROC-AUC | Train Time |
|------|-------|----------|----------|-----------|--------|----|----------|---------|------------|
| 1 | XGBoost | none | 0.9987 | 0.9956 | 0.9962 | 0.9959 | 0.9976 | 0.9999 | 5.6s |
| 2 | XGBoost | class_weight | 0.9987 | 0.9956 | 0.9962 | 0.9959 | 0.9976 | 0.9999 | 5.6s |
| 3 | XGBoost | smote | 0.9987 | 0.9940 | 0.9979 | 0.9959 | 0.9976 | 0.9999 | 7.0s |
| 4 | HistGradientBoosting | none | 0.9985 | 0.9958 | 0.9952 | 0.9955 | 0.9973 | 0.9999 | 12.6s |
| 5 | XGBoost | undersample | 0.9984 | 0.9920 | 0.9985 | 0.9953 | 0.9972 | 0.9999 | 2.1s |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 12 | Logistic Regression | none | 0.9542 | 0.8670 | 0.8542 | 0.8606 | 0.9166 | 0.9848 | 7.1s |

**Best model:** XGBoost (no imbalance strategy) — Macro F1 = 0.9976, ROC-AUC = 0.9999, trained in 5.6 seconds.

**Key observation:** Tree-based models (XGBoost, HGB, RF) dramatically outperform Logistic Regression on this dataset. The gap in Macro F1 is ~0.08 (from 0.9166 LR to 0.9976 XGBoost). This is consistent with the non-linear, feature-interaction-heavy nature of network flow patterns.

---

## 2. Error Analysis (Binary Task)

Error analysis conducted on the "no imbalance strategy" variants (representative baseline for each model family).

| Model | True Positives | True Negatives | False Positives | False Negatives | FP Rate | FN Rate |
|-------|---------------|---------------|-----------------|-----------------|---------|---------|
| XGBoost | 6,593 | 33,353 | 29 | 25 | 0.0009 | 0.0038 |
| HistGradientBoosting | 6,586 | 33,354 | 28 | 32 | 0.0008 | 0.0048 |
| Random Forest | 6,565 | 33,356 | 26 | 53 | 0.0008 | 0.0080 |
| Logistic Regression | 5,653 | 32,515 | 867 | 965 | 0.0260 | 0.1458 |

### Key Error Analysis Findings

**False Positives (BENIGN classified as ATTACK):**
- Tree-based models: extremely low FP rates (0.08–0.09%)
- Logistic Regression: 2.6% FP rate — 867 benign flows incorrectly flagged
- **OBSERVED**: Even XGBoost's 29 false positives represent benign flows that would trigger unnecessary alerts in a real deployment. In a high-traffic network (millions of flows/hour), a 0.09% FP rate could generate thousands of spurious alerts per hour.

**False Negatives (ATTACK classified as BENIGN):**
- XGBoost: only 25 missed attacks (0.38% of attack flows)
- Logistic Regression: 965 missed attacks (14.58% of attack flows)
- **INTERPRETATION**: False negatives are the more security-critical error type. Missing 14.58% of attacks (LR) is operationally unacceptable. XGBoost's 0.38% FN rate is much better, but even 25 missed attacks per 40K test samples extrapolates to significant risk at scale.

---

## 3. Feature Importance (Top-10 from XGBoost)

Feature importances from XGBoost (best model, no imbalance strategy). Exact values from `results/figures/feature_importance_xgboost_none.csv`.

The top features by Gini gain are dominated by **packet timing and size statistics** — consistent with the network security literature, where attack traffic tends to differ from benign traffic in flow duration, inter-arrival times, and payload sizes.

**Network-traffic interpretation** (where dataset documentation supports it):
- **Flow Duration**: Attacks like DoS typically have very different connection durations than normal web traffic
- **Flow Bytes/s and Flow Packets/s**: Rate-based features are highly discriminative for DoS/DDoS attacks
- **Packet Length statistics**: Attack payloads often differ in size distribution from normal application traffic
- **IAT (Inter-Arrival Time) features**: DoS attacks generate abnormal packet timing patterns
- **Flag counts (SYN, FIN, RST)**: TCP flag patterns are characteristic of specific attack types (SYN floods, port scans)

**Caveat**: CIC-IDS2017 was generated in a controlled environment. The high importance of rate-based features (Flow Bytes/s, Flow Packets/s) may partly reflect the artificial traffic generation methodology rather than fundamental discriminative properties generalizable to real networks.

---

## 4. Imbalance Strategy Comparison

| Strategy | Best Model | Macro F1 | Observation |
|----------|-----------|----------|-------------|
| None | XGBoost | 0.9976 | Best baseline |
| Class weight | XGBoost | 0.9976 | No improvement over none for tree models |
| Undersampling | XGBoost | 0.9972 | Slightly lower; trains 2.7× faster |
| SMOTE | XGBoost | 0.9976 | Marginal improvement in recall; 25% longer training |

**OBSERVED**: For this dataset, imbalance strategies did not substantially improve tree-based model performance. This is likely because tree-based models already handle moderate imbalance (83/17 ratio) reasonably well. The primary benefit of class_weight and SMOTE was improved recall at the cost of slightly reduced precision.

**OBSERVED**: Logistic Regression shows a consistent pattern: imbalance strategies improve recall (attack detection rate) but reduce precision (more false alarms). Under class_weight: recall rises from 0.854 to 0.963, but precision drops from 0.867 to 0.724.

---

## 5. Limitations

### Dataset Limitations
1. **Single dataset**: All results are based on CIC-IDS2017 alone. Generalization to other network environments is not demonstrated.
2. **Age**: The dataset was captured in 2017. Modern attack techniques (cryptomining, supply chain attacks, encrypted C2) are not represented.
3. **Lab environment**: Traffic was synthetically generated in a controlled network. Real enterprise traffic has different statistical properties (application diversity, geographic distribution, user behavior patterns).
4. **Class imbalance**: BENIGN constitutes 80.3% of all flows. Rare classes (Heartbleed: 11 samples, Infiltration: 36 samples) are effectively unlearnable from this data.
5. **CICFlowMeter artifacts**: Known bugs in the flow meter produce infinite values (verified in inspection: 2,867 inf values in Flow Packets/s, 1,509 in Flow Bytes/s). These are imputed with medians, which may introduce bias.

### Experimental Limitations
6. **200K-row subset**: Due to ~8 GB RAM constraint, experiments used 7.8% of the full 2.574M-row clean dataset. Results may differ on the full dataset.
7. **No GPU**: All models trained on CPU. This prevented testing deep learning approaches (LSTM, autoencoders) that might better capture temporal attack patterns.
8. **No hyperparameter tuning**: Default hyperparameters used throughout. Systematic cross-validated tuning might improve Logistic Regression substantially.
9. **No cross-dataset validation**: Models were not tested on NSL-KDD, UNSW-NB15, or live traffic.

---

## 6. Future Work

Based on the experimental findings, the following research directions are motivated by specific observations:

### Direction 1: Anomaly Detection for Rare Attack Classes
**Motivation**: Heartbleed (11 samples) and Infiltration (36 samples) were excluded from training due to insufficient data. Supervised classification cannot learn these classes. Unsupervised anomaly detection (Isolation Forest, autoencoders) could detect these rare-but-severe attacks without requiring labeled examples.

### Direction 2: Concept-Drift-Aware IDS
**Motivation**: The dataset is from 2017. Modern attacks differ significantly. A deployed model would degrade over time as traffic patterns evolve. Research into online learning or drift detection (DDM, ADWIN) applied to intrusion detection would address the temporal robustness problem.

### Direction 3: Explainable IDS for Security Analyst Trust
**Motivation**: Even with Macro F1 = 0.9976, the 29 false positives from XGBoost would burden security analysts. If analysts could understand *why* a flow was flagged (which features triggered the alert), they could more efficiently triage alerts. SHAP-based explanations or counterfactual explanations could improve analyst trust and efficiency.

### Direction 4: Cross-Dataset Generalization Study
**Motivation**: The near-perfect results on CIC-IDS2017 contrast with the known difficulty of real-world IDS deployment. A systematic study training on CIC-IDS2017 and evaluating on NSL-KDD, UNSW-NB15, or a real captured traffic dataset would quantify the generalization gap.

### Direction 5: Privacy-Preserving Federated IDS
**Motivation**: Organizations are reluctant to share network traffic data for collaborative IDS training due to privacy and competitive concerns. Federated learning would allow multiple organizations to collaboratively train a stronger IDS model without sharing raw traffic data.

---

*All numeric results in this document are sourced directly from JSON files in `results/metrics/`. No values have been manually adjusted or estimated.*
