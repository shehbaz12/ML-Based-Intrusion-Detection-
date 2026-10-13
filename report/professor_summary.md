# Research Project Summary

## Machine Learning-Based Network Intrusion Detection

I recently completed a research project investigating the application of classical machine learning methods to network intrusion detection using the CIC-IDS2017 benchmark dataset published by the Canadian Institute for Cybersecurity at the University of New Brunswick.

The project involved loading and inspecting 2.83 million labeled network flow records across 15 traffic classes, performing rigorous data cleaning (removing 256,684 duplicate rows and handling infinite values in flow-rate features), and training four classifiers — Logistic Regression, Random Forest, HistGradientBoosting, and XGBoost — under four class imbalance strategies. All preprocessing was performed with strict leakage prevention: the normalization pipeline was fitted exclusively on training data. Experiments used a reproducible 200,000-row stratified subset due to hardware memory constraints (~8 GB RAM, CPU only).

For binary classification (benign vs. attack), XGBoost achieved Macro F1 of 0.9976 and a false negative rate of only 0.38%, while Logistic Regression achieved Macro F1 of 0.917 with a 14.58% false negative rate. These results raised an important research question: do near-perfect benchmark metrics reflect genuine intrusion detection capability, or artifacts of controlled dataset construction?

The most significant finding is not the metric values but what they reveal about the limitations of current IDS benchmarking: the dataset's lab-generated traffic, 2017 vintage, and severe class imbalance (Heartbleed: 11 samples; Infiltration: 36 samples) make it unsuitable as a proxy for real-world performance. This motivates research into anomaly detection for rare attack classes, concept-drift-aware models, and cross-dataset generalization evaluation.

I am interested in extending this work toward explainable intrusion detection and anomaly-based approaches for novel attack discovery — directions I understand are central to your research group's interests.
