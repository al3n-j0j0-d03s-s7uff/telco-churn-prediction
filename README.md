# Telecom Customer Churn Prediction using Machine Learning

A reproducible machine-learning project that analyzes telecom customer data and predicts customer churn. The workflow is based on the supplied project report and includes data cleaning, exploratory data analysis (EDA), feature preprocessing, model comparison, and evaluation.

> **Dataset note:** The CSV dataset is not included in this repository. Download the Telco Customer Churn dataset separately and place it in the `data/` directory. The script will report a clear error if the dataset is missing.

## Project highlights

- Cleans the `TotalCharges` field and handles missing values through preprocessing pipelines.
- Removes `customerID`, which is an identifier rather than a predictive feature.
- Explores churn patterns using plots for churn distribution, tenure, monthly charges, and numerical correlations.
- One-hot encodes categorical features and scales numerical features for Logistic Regression.
- Compares five classifiers:
  - Tuned Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Extra Trees
- Uses stratified train/test splitting and five-fold cross-validation.
- Reports Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrix.
- Saves model comparison tables and plots to `results/`.

## Repository structure

```text
Telco-Churn-Prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # add this yourself
├── docs/
│   └── project_report.pdf
├── notebooks/
│   └── telco_churn_analysis.ipynb
├── results/
│   └── .gitkeep
├── src/
│   └── churn_prediction.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset setup

Use the **Telco Customer Churn** dataset commonly distributed as `WA_Fn-UseC_-Telco-Customer-Churn.csv`. It contains customer account and service attributes and a `Churn` target column.

1. Obtain the dataset from the source you are permitted to use (for example, the IBM sample dataset distributed through public data-science repositories).
2. Create a folder named `data` in the repository root.
3. Place the CSV there with this exact filename:

   `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Do not commit private, restricted, or personally identifiable customer data. This project is intended for the publicly available sample dataset.

## Setup

Python 3.10 or newer is recommended.

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the analysis

From the repository root:

```bash
python src/churn_prediction.py
```

If the CSV is stored elsewhere:

```bash
python src/churn_prediction.py --data "path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv"
```

Choose a different output directory if needed:

```bash
python src/churn_prediction.py --output results
```

## Outputs

After a successful run, the script creates:

- `results/model_comparison.csv` — cross-validation and test metrics for all five models.
- `results/best_model_classification_report.txt` — classification report for the model with the highest held-out test ROC-AUC.
- `results/run_metadata.json` — run settings and selected model.
- `results/figures/churn_distribution.png`
- `results/figures/tenure_vs_churn.png`
- `results/figures/monthly_charges_vs_churn.png`
- `results/figures/correlation_heatmap.png`
- `results/figures/model_comparison.png`
- `results/figures/best_model_confusion_matrix.png`
- `results/figures/best_model_roc_curve.png`

Metrics are calculated when you run the project; they are not hard-coded in the repository.

## Methodology

1. **Load and clean:** read the CSV, remove the customer identifier, convert `TotalCharges` to numeric, and map `Churn` from Yes/No to 1/0.
2. **Explore:** generate plots to inspect class balance and relationships between churn, tenure, and charges.
3. **Split:** use an 80/20 stratified train/test split with a fixed random seed.
4. **Preprocess:** impute numeric and categorical missing values; one-hot encode categories; scale numeric features for Logistic Regression.
5. **Train and compare:** use five-fold stratified cross-validation and tune Logistic Regression's `C` parameter using ROC-AUC.
6. **Evaluate:** calculate Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix, and ROC curve.

## Important notes

- The repository reconstructs the workflow described in the supplied report; the code has been organized into a reproducible script with preprocessing inside scikit-learn pipelines.
- Results may differ from the original report because of implementation details, library versions, data splits, and model-selection choices.
- The script selects the final model using held-out test ROC-AUC for reporting convenience. For rigorous model selection, choose the model using cross-validation and reserve the test set for a single final evaluation.
- This is an educational project, not a production churn prediction service.

## Tech stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn`

## License

No license is included yet. Add a license only after deciding how you want others to use, modify, and redistribute this work.
