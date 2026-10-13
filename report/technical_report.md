# Machine Learning-Based Network Intrusion Detection System
## Technical Report

**Author:** [Your Name]  
**Date:** October 2026  
**Institution:** [Your University]

---

## 1. Abstract

This report presents an experimental evaluation of classical machine learning methods for network intrusion detection using the CIC-IDS2017 benchmark dataset. We trained and compared four classifiers — Logistic Regression, Random Forest, HistGradientBoosting, and XGBoost — across four class imbalance handling strategies on a 200,000-row stratified subset of the dataset. For binary classification (BENIGN vs. ATTACK), tree-based models achieved Macro F1 scores above 0.996 and ROC-AUC above 0.999, while Logistic Regression achieved Macro F1 of 0.917 and ROC-AUC of 0.985. Error analysis revealed that tree-based models produce very low false negative rates (0.38–0.80%) compared to Logistic Regression (14.58%). Feature importance analysis indicates that rate-based and timing features (Flow Bytes/s, Flow IAT statistics, Packet Length statistics) are the most discriminative. However, the near-perfect results should be interpreted cautiously: CIC-IDS2017 was generated in a controlled laboratory environment, and generalization to real-world traffic is not demonstrated. We identify five research directions motivated by the experimental findings.

---

## 2. Introduction

Network intrusion detection systems (IDS) are a critical component of cybersecurity infrastructure. Traditional signature-based IDS require manual rule creation and cannot detect novel attacks. Machine learning-based IDS can potentially learn attack patterns from labeled data automatically. However, deploying ML-based IDS in practice raises challenges: class imbalance (benign traffic dominates), evolving attack patterns (concept drift), false alarm rates, and computational constraints.

This project investigates whether classical, CPU-friendly machine learning methods can achieve reliable intrusion detection on a publicly available benchmark dataset, and what limitations and open problems emerge from a careful experimental evaluation.

---

## 3. Problem Statement

Given labeled network flow records extracted from CIC-IDS2017, train and evaluate ML classifiers to:
1. Distinguish malicious from benign traffic (binary classification)
2. Identify the specific attack category (multiclass classification)
3. Quantify false positive and false negative rates
4. Identify which network features are most discriminative
5. Identify limitations and open research problems

---

## 4. Research Questions

**RQ1:** Can classical ML classifiers reliably distinguish malicious from benign network traffic on CIC-IDS2017?

**RQ2:** Do class imbalance handling strategies (class weighting, undersampling, SMOTE) substantially improve detection performance on this dataset?

**RQ3:** Which network flow features are most important for classification, and do they have a plausible security interpretation?

**RQ4:** What are the false positive and false negative characteristics of each model, and what do they imply for real-world deployment?

**RQ5:** What limitations of the dataset and experimental setup constrain the conclusions that can be drawn?

---

## 5. Background

### 5.1 CIC-IDS2017 Dataset

Sharafaldin et al. (2018) introduced the CIC-IDS2017 dataset to address the lack of realistic, labeled intrusion detection benchmarks. The dataset was generated using CICFlowMeter to extract 76 bidirectional flow features from five days of network traffic. It contains 15 traffic classes: BENIGN and 14 attack types including DoS, DDoS, PortScan, brute force attacks, web attacks, and infiltration.

### 5.2 Machine Learning for IDS

Tree-based ensemble methods (Random Forest, Gradient Boosting) have become standard baselines for ML-based IDS due to their ability to handle high-dimensional, mixed-distribution feature spaces without extensive preprocessing. Logistic Regression serves as a linear baseline. XGBoost's histogram-based tree construction makes it particularly efficient on CPU hardware.

### 5.3 Class Imbalance

In network traffic datasets, benign traffic typically dominates (80+ percent in CIC-IDS2017). Class imbalance can cause classifiers to optimize for majority-class accuracy at the expense of minority-class (attack) recall. Strategies include cost-sensitive learning (class weights), undersampling the majority class, and oversampling the minority class with SMOTE.

---

## 6. Dataset

**Source:** Canadian Institute for Cybersecurity, University of New Brunswick  
**Download:** https://www.unb.ca/cic/datasets/ids-2017.html (MachineLearningCSV.zip)

### 6.1 Raw Dataset Statistics

| Property | Value |
|----------|-------|
| Total rows | 2,830,743 |
| Columns | 79 |
| Duplicate rows | 256,684 (9.07%) |
| Infinite values | 4,376 (in Flow Bytes/s, Flow Packets/s) |
| Classes | 15 |

### 6.2 Label Distribution (after cleaning)

| Class | Count | Proportion |
|-------|-------|------------|
| BENIGN | 2,148,196 | 83.46% |
| DoS Hulk | 172,849 | 6.72% |
| DDoS | 128,016 | 4.97% |
| PortScan | 90,819 | 3.53% |
| DoS GoldenEye | 10,282 | 0.40% |
| FTP-Patator | 5,933 | 0.23% |
| DoS slowloris | 5,374 | 0.21% |
| DoS Slowhttptest | 5,228 | 0.20% |
| SSH-Patator | 3,219 | 0.13% |
| Bot | 1,953 | 0.08% |
| Web Attack – Brute Force | 1,470 | 0.06% |
| Web Attack – XSS | 652 | 0.03% |
| Infiltration | 36 | 0.001% |
| Web Attack – Sql Injection | 21 | 0.001% |
| Heartbleed | 11 | 0.0004% |

### 6.3 Preprocessing

- Replaced ±∞ with NaN (4,376 values)
- Dropped 256,684 duplicate rows
- Dropped `Destination Port` (high cardinality) and `Fwd Header Length.1` (duplicate column)
- Imputed remaining NaN with per-feature median (fit on training data only)
- Standardized all features (StandardScaler, fit on training data only)
- **Final feature count:** 76

---

## 7. Methodology

### 7.1 Experimental Setup

Due to ~8 GB RAM constraint, a 200,000-row stratified subset was sampled (random_state=42, fraction=0.078). This is explicitly documented as a limitation. All results apply to this subset.

**Train/Test Split:** 80/20 stratified (160,000 train / 40,000 test), random_state=42  
**Leakage prevention:** Preprocessing pipeline (imputer + scaler) fitted on training data only, then applied to test data.

### 7.2 Models

| Model | Hyperparameters |
|-------|----------------|
| Logistic Regression | max_iter=1000, random_state=42 |
| Random Forest | n_estimators=100, random_state=42, n_jobs=-1 |
| HistGradientBoosting | max_iter=100, random_state=42 |
| XGBoost | n_estimators=100, tree_method="hist", random_state=42 |

### 7.3 Imbalance Strategies

| Strategy | Description |
|----------|-------------|
| None | No resampling |
| Class weight | Inverse class frequency weights passed to model |
| Undersample | RandomUnderSampler: 133,529 → 26,471 majority class samples |
| SMOTE | Synthetic minority oversampling: 26,471 → 133,529 minority class samples |

### 7.4 Evaluation Metrics

Primary: **Macro F1** (unweighted average F1 across classes — appropriate for imbalanced data)  
Secondary: Accuracy, Precision, Recall, F1 (binary), ROC-AUC, PR-AUC  
Error analysis: FP rate, FN rate, per-class recall

---

## 8. Experimental Setup

All experiments conducted on:
- Windows 11, Intel Core i5 (Family 6 Model 142), ~8 GB RAM
- Python 3.12.10, scikit-learn 1.9.1, XGBoost 3.4.1
- No GPU acceleration

---

## 9. Results

### 9.1 Binary Classification

**⚠️ Note:** Results exceeding Macro F1 = 0.99 on CIC-IDS2017 should be interpreted cautiously due to dataset construction artifacts (see Section 12).

| Model | Strategy | Accuracy | Precision | Recall | F1 | Macro F1 | ROC-AUC | Train Time |
|-------|----------|----------|-----------|--------|----|----------|---------|------------|
| XGBoost | none | 0.9987 | 0.9956 | 0.9962 | 0.9959 | 0.9976 | 0.9999 | 5.6s |
| XGBoost | class_weight | 0.9987 | 0.9956 | 0.9962 | 0.9959 | 0.9976 | 0.9999 | 5.6s |
| XGBoost | smote | 0.9987 | 0.9940 | 0.9979 | 0.9959 | 0.9976 | 0.9999 | 7.0s |
| HistGradientBoosting | none | 0.9985 | 0.9958 | 0.9952 | 0.9955 | 0.9973 | 0.9999 | 12.6s |
| XGBoost | undersample | 0.9984 | 0.9920 | 0.9985 | 0.9953 | 0.9972 | 0.9999 | 2.1s |
| Random Forest | class_weight | 0.9982 | 0.9946 | 0.9949 | 0.9947 | 0.9968 | 0.9998 | 17.4s |
| HistGradientBoosting | smote | 0.9982 | 0.9917 | 0.9976 | 0.9947 | 0.9968 | 0.9999 | 13.3s |
| Random Forest | smote | 0.9980 | 0.9937 | 0.9946 | 0.9941 | 0.9965 | 0.9998 | 34.9s |
| Random Forest | none | 0.9980 | 0.9961 | 0.9920 | 0.9940 | 0.9964 | 0.9999 | 34.8s |
| Random Forest | undersample | 0.9977 | 0.9899 | 0.9959 | 0.9929 | 0.9958 | 0.9998 | 5.0s |
| HistGradientBoosting | undersample | 0.9976 | 0.9871 | 0.9982 | 0.9926 | 0.9956 | 0.9999 | 3.6s |
| Logistic Regression | none | 0.9542 | 0.8670 | 0.8542 | 0.8606 | 0.9166 | 0.9848 | 7.1s |
| Logistic Regression | smote | 0.9349 | 0.7283 | 0.9671 | 0.8309 | 0.8953 | 0.9853 | 10.0s |
| Logistic Regression | class_weight | 0.9330 | 0.7236 | 0.9630 | 0.8263 | 0.8924 | 0.9848 | 6.5s |
| Logistic Regression | undersample | 0.9302 | 0.7135 | 0.9663 | 0.8209 | 0.8888 | 0.9838 | 1.2s |

**OBSERVED RESULT:** XGBoost with no imbalance strategy achieves the best Macro F1 (0.9976) and is also the fastest to train (5.6s). Tree-based models outperform Logistic Regression by ~0.08 Macro F1.

**OBSERVED RESULT:** Imbalance strategies do not substantially improve tree-based model performance. For Logistic Regression, they improve recall but reduce precision.

### 9.2 Error Analysis (Binary Task)

| Model | FP | FN | FP Rate | FN Rate |
|-------|----|----|---------|---------|
| XGBoost | 29 | 25 | 0.09% | 0.38% |
| HistGradientBoosting | 28 | 32 | 0.08% | 0.48% |
| Random Forest | 26 | 53 | 0.08% | 0.80% |
| Logistic Regression | 867 | 965 | 2.60% | 14.58% |

**OBSERVED RESULT:** XGBoost has the lowest false negative rate (0.38%). In security contexts, false negatives (missed attacks) are more critical than false positives. Logistic Regression's 14.58% FN rate is operationally problematic.

---

## 10. Feature Analysis

Feature importances extracted from XGBoost (best model). Top features by Gini gain are dominated by flow-rate and timing statistics:

- **Flow Bytes/s, Flow Packets/s**: Rate-based features highly discriminate DoS/DDoS
- **Packet Length statistics** (Mean, Variance, Max): Attack payloads differ from benign
- **IAT features** (Flow IAT Mean/Std): Abnormal timing patterns in attacks
- **TCP flag counts** (SYN, FIN, RST, ACK): Characteristic of specific attack protocols

**INTERPRETATION (hypothesis):** Rate-based features dominate importance likely because DoS/DDoS constitute ~14% of all attack traffic. These attacks are inherently high-rate. This interpretation should be tested by examining per-class feature importance — a direction for future work.

**CAVEAT:** CICFlowMeter's known artifact of producing infinite values in rate features (replaced by median imputation) may cause these features' computed importance to be lower than their true discriminative power.

---

## 11. Error Analysis

### 11.1 Binary Task

Logistic Regression produces 867 false positives and 965 false negatives per 40,000 test samples. Tree-based models reduce both dramatically — XGBoost: 29 FP, 25 FN. The large gap suggests that attack/benign separation requires non-linear decision boundaries that Logistic Regression cannot represent.

### 11.2 Multiclass Task (Qualitative)

The multiclass experiment revealed a data partitioning challenge: after stratified sampling and train/test splitting, the test set contained 12 classes while training covered 13 (Web Attack – Sql Injection had no test samples). This reflects a fundamental problem with rare classes in IDS: classes with very few samples are unlearnable from limited data and will have unpredictable behavior at test time.

---

## 12. Discussion

### OBSERVED RESULTS:
- XGBoost achieves Macro F1 = 0.9976, ROC-AUC = 0.9999 on binary classification
- Tree-based models dramatically outperform linear models
- Imbalance strategies have minimal effect on tree-based models for this dataset
- False negative rate for XGBoost (0.38%) is much better than Logistic Regression (14.58%)

### INTERPRETATIONS:
- The high performance likely reflects the controlled, homogeneous nature of CIC-IDS2017 traffic rather than fundamental ease of network intrusion detection
- The dominance of rate-based features may reflect the prevalence of volumetric attacks (DoS/DDoS) in the dataset
- The large performance gap between tree models and Logistic Regression suggests non-linear attack patterns

### RESEARCH IDEAS (not demonstrated in this project):
- SHAP analysis could identify whether rate features truly drive classification or if there are confounding factors
- Testing on a different dataset (NSL-KDD, UNSW-NB15) would quantify the generalization gap
- Temporal analysis could reveal whether attack patterns change over time within the dataset

---

## 13. Limitations

1. **Single dataset**: No cross-dataset validation
2. **7.8% subset**: Due to RAM constraint; full-dataset results may differ
3. **2017 data**: Attack techniques have evolved significantly
4. **Lab environment**: Traffic generated synthetically, not from real enterprise networks
5. **Rare classes**: Heartbleed (11 samples) and Infiltration (36 samples) unlearnable
6. **No hyperparameter tuning**: Default parameters used throughout
7. **No temporal analysis**: Flows treated as IID; temporal patterns not exploited
8. **No deployment testing**: No live traffic evaluation

---

## 14. Future Work

1. **Anomaly detection for rare attack classes** — unsupervised methods for Heartbleed, Infiltration
2. **Concept drift adaptation** — online learning or drift detection for temporal robustness
3. **Explainable IDS** — SHAP-based alert explanations for security analyst trust
4. **Cross-dataset generalization study** — quantify performance on NSL-KDD, UNSW-NB15
5. **Federated IDS** — privacy-preserving collaborative training across organizations

---

## 15. Conclusion

This project demonstrates that XGBoost achieves strong binary intrusion detection performance (Macro F1 = 0.9976, FN rate = 0.38%) on a 200K-row stratified subset of CIC-IDS2017. However, the near-perfect results reflect characteristics of the benchmark dataset rather than evidence that the system is ready for real-world deployment. The most important research finding is not the metric values but the identified limitations: dataset age, lab-environment artifacts, rare class unlearnable, and the open question of generalization to real traffic. These limitations directly motivate the five future research directions identified. This work demonstrates a complete, reproducible ML research pipeline — from data inspection through model comparison and error analysis — appropriate for further development in graduate research on ML-based cybersecurity.

---

## 16. References

[1] Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, 108–116.

[2] Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

[3] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

[4] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

[5] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

[6] Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.

[7] Buczak, A. L., & Guven, E. (2016). A Survey of Data Mining and Machine Learning Methods for Cyber Security Intrusion Detection. *IEEE Communications Surveys & Tutorials*, 18(2), 1153–1176.
