# ML-Based-Intrusion-Detection-

Machine Learning-Based Network Intrusion Detection

A research project investigating classical machine-learning methods for network intrusion detection using the CIC-IDS2017 benchmark dataset.

Overview

This project implements a reproducible machine-learning pipeline for classifying network traffic as benign or malicious. It includes dataset inspection, data cleaning, leakage-aware preprocessing, model training, class-imbalance analysis, feature-importance analysis, and error analysis.

The project compares four CPU-friendly classifiers:

•
Logistic Regression

•
Random Forest

•
HistGradientBoosting

•
XGBoost

Because of hardware memory limitations, the experiments use a reproducible 200,000-row stratified subset of CIC-IDS2017.

Research Questions

1.
Can classical machine-learning models distinguish benign and malicious network traffic?

2.
How do class-imbalance strategies affect intrusion-detection performance?

3.
Which network-flow features are most important for classification?

4.
What types of errors do different models produce?

5.
How well do benchmark results represent real-world intrusion detection?

Dataset

This project uses the CIC-IDS2017 dataset from the Canadian Institute for Cybersecurity.

The dataset is not included in this repository because of its large size. Download it from:



https://www.unb.ca/cic/datasets/ids-2017.html

After downloading and extracting the dataset, place all CSV files in:

Plain Text


data/raw/



See data/README.md for detailed instructions.

Methodology

The preprocessing pipeline includes:

•
Replacement of infinite values

•
Removal of duplicate rows

•
Removal of selected duplicated or excluded features

•
Stratified subset sampling

•
Stratified train/test splitting

•
Median imputation

•
Standardization fitted on training data only

•
Comparison of class-imbalance strategies

The evaluated imbalance strategies are:

•
No imbalance handling

•
Class weighting

•
Random undersampling

•
SMOTE

Preliminary Results

In the preliminary binary-classification experiment, XGBoost achieved approximately:

•
Macro F1: 0.998

•
ROC-AUC: 1.000

•
False-negative rate: 0.38%

These results were obtained on a stratified subset of CIC-IDS2017. They should not be interpreted as evidence of production-level or real-world performance. The high scores may be influenced by the controlled laboratory environment, duplicated or highly similar traffic records, and random row-level train/test splitting.

The multiclass evaluation is being refined to ensure that feature rows and target labels remain correctly aligned during preprocessing and evaluation.

Limitations

•
Only one benchmark dataset is used.

•
CIC-IDS2017 was collected in a controlled laboratory environment in 2017.

•
Experiments use a 200,000-row subset because of memory limitations.

•
Several attack classes contain very few examples.

•
Cross-dataset validation has not yet been completed.

•
No live-network or production deployment evaluation has been performed.

•
Random row-level splitting may produce optimistic results when similar records appear in both training and test sets.

Repository Structure

Plain Text


.
├── data/
│   ├── raw/                 # Downloaded dataset files
│   ├── processed/           # Generated processed arrays
│   └── README.md
├── models/                  # Saved model artifacts
├── results/
│   ├── figures/             # Evaluation figures
│   ├── metrics/             # Metrics in JSON format
│   ├── tables/              # Comparison tables
│   └── research_findings.md
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── feature_analysis.py
├── tests/
├── run_preprocessing.py
├── run_eda.py
├── run_experiment.py
├── run_multiclass.py
├── requirements.txt
└── README.md



Installation

Bash


git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

python -m venv .venv



Linux/macOS

Bash


source .venv/bin/activate



Windows

Bash


.venv\Scripts\activate



Install the dependencies:

Bash


pip install -r requirements.txt



Usage

After placing the CIC-IDS2017 CSV files in data/raw/, run:

Bash


python run_preprocessing.py
python run_eda.py
python run_experiment.py --task binary
python run_multiclass.py



Run the tests with:

Bash


pytest tests/ -v



Reproducibility

The experiments use:

•
random_state=42

•
An 80/20 stratified train/test split

•
A fixed 200,000-row stratified subset

•
Training-only preprocessing

•
JSON files for storing evaluation metrics

•
PNG figures generated at 300 DPI

Results should be regenerated from a clean environment after downloading the dataset.

Future Work

Planned research directions include:

•
Temporal and day-based validation

•
Duplicate-aware train/test splitting

•
Cross-dataset evaluation

•
Anomaly detection for rare attacks

•
Concept-drift-aware intrusion detection

•
Explainable intrusion detection

•
Evaluation on modern network-traffic datasets

Citation

If you use this project, please cite the CIC-IDS2017 dataset paper:

Plain Text


@inproceedings{sharafaldin2018toward,
  title={Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  author={Sharafaldin, Iman and Habibi Lashkari, Arash and Ghorbani, Ali A},
  booktitle={Proceedings of the 4th International Conference on Information Systems Security and Privacy},
  pages={108--116},
  year={2018}
}



Author

Your Name
Your university or institution
Your email address

