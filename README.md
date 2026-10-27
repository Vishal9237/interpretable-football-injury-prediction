# Interpretable Football Injury Risk Prediction

A machine learning-based approach to predicting football injury risk using player-related features, model benchmarking, and SHAP-based explainability.

## Overview

This project explores how machine learning can be used to identify patterns associated with injury risk in university football players. Multiple classification algorithms are evaluated, and SHAP (SHapley Additive exPlanations) is used to interpret model predictions and understand feature importance.

The project is based on the research paper by Ma, Liu, and Pei (2025), published in *Scientific Reports*.

## Objectives

- Explore machine learning techniques for football injury risk prediction.
- Preprocess and analyse player-related features.
- Compare the performance of multiple classification algorithms.
- Evaluate models using standard classification metrics.
- Interpret model predictions using SHAP visualizations.

## Methodology

1. **Data generation:** Create a synthetic dataset using published group-level statistics.
2. **Preprocessing:** Encode categorical variables and standardize numerical features.
3. **Model training:** Train and compare 10 machine learning classification algorithms.
4. **Evaluation:** Compare models using accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and Brier score.
5. **Explainability:** Use SHAP to examine feature contributions to model predictions.
6. **Visualization:** Generate performance comparisons, ROC curves, confusion matrices, and SHAP plots.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- SHAP

## Project Structure

```text
interpretable-football-injury-prediction/
├── full_implementation.py
├── requirements.txt
├── README.md
├── report.pdf
├── presentation.pptx
└── results/
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── roc_curves.png
    └── shap_bar.png
```

*The structure above is illustrative. Keep only the files and output images that actually exist in your repository.*

## Installation

Clone the repository:

```bash
git clone https://github.com/Vishal9237/interpretable-football-injury-prediction.git
cd interpretable-football-injury-prediction
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run the implementation:

```bash
python full_implementation.py
```

The script performs data generation, preprocessing, model training, evaluation, and SHAP-based interpretation. Generated plots are saved to the output locations defined in the script.

## Dataset and Limitations

**Important:** This implementation uses a synthetic dataset generated from published group-level means and standard deviations. It does not reproduce the original athlete-level dataset.

Consequently, model performance on this synthetic data should be interpreted as an experimental demonstration of the pipeline, not as evidence of real-world predictive accuracy. The system is intended for educational and research purposes and is not a medical diagnostic tool.

## Reference Paper

Ma, J., Liu, S., & Pei, Y. (2025). *SHAP-based interpretable machine learning for injury risk prediction in university football players.* Scientific Reports, 15, Article 40252.

[Read the research paper](https://doi.org/10.1038/s41598-025-24144-y)

## Author

**Vishal9237**

GitHub: [@Vishal9237](https://github.com/Vishal9237)
