"""
Telecom Customer Churn Prediction
---------------------------------
Reproducible machine-learning workflow based on the supplied project report.

Expected input:
    data/WA_Fn-UseC_-Telco-Customer-Churn.csv

Run from the repository root:
    python src/churn_prediction.py
    python src/churn_prediction.py --data path/to/your.csv --output results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier

RANDOM_STATE = 42
TARGET = "Churn"


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate telecom churn classifiers.")
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        help="Path to the Telco Customer Churn CSV."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results"),
        help="Directory where metrics, plots, and model metadata are saved."
    )
    return parser.parse_args()


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Download the Telco Customer Churn CSV (see README.md), then place it at "
            "data/WA_Fn-UseC_-Telco-Customer-Churn.csv or pass --data PATH."
        )

    df = pd.read_csv(csv_path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in the CSV.")

    # CustomerID is an identifier rather than a useful predictive feature.
    df = df.drop(columns=["customerID"], errors="ignore").copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # The source report maps Yes/No to 1/0. Validate rather than silently invent labels.
    if df[TARGET].dtype == object:
        mapped = df[TARGET].astype(str).str.strip().map({"No": 0, "Yes": 1})
        if mapped.isna().any():
            unknown = sorted(df.loc[mapped.isna(), TARGET].astype(str).unique().tolist())
            raise ValueError(f"Unexpected values in '{TARGET}': {unknown}")
        df[TARGET] = mapped.astype(int)
    else:
        df[TARGET] = df[TARGET].astype(int)

    return df


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])

    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ], remainder="drop")


def save_eda(df: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    plot_dir = out_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET, ax=ax)
    ax.set_title("Churn Distribution (0 = No, 1 = Yes)")
    ax.set_xlabel("Churn")
    fig.tight_layout()
    fig.savefig(plot_dir / "churn_distribution.png", dpi=160)
    plt.close(fig)

    if "tenure" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x=TARGET, y="tenure", ax=ax)
        ax.set_title("Tenure vs Churn")
        fig.tight_layout()
        fig.savefig(plot_dir / "tenure_vs_churn.png", dpi=160)
        plt.close(fig)

    if "MonthlyCharges" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x=TARGET, y="MonthlyCharges", ax=ax)
        ax.set_title("Monthly Charges vs Churn")
        fig.tight_layout()
        fig.savefig(plot_dir / "monthly_charges_vs_churn.png", dpi=160)
        plt.close(fig)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Numeric Feature Correlation Heatmap")
        fig.tight_layout()
        fig.savefig(plot_dir / "correlation_heatmap.png", dpi=160)
        plt.close(fig)


def build_models():
    return {
        "Logistic Regression (Tuned)": (
            Pipeline([
                ("preprocessor", None),  # replaced after X is known
                ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
            ]),
            {"classifier__C": [0.01, 0.1, 1, 10, 50]}
        ),
        "Decision Tree": (
            Pipeline([
                ("preprocessor", None),
                ("classifier", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE))
            ]),
            {}
        ),
        "Random Forest": (
            Pipeline([
                ("preprocessor", None),
                ("classifier", RandomForestClassifier(
                    n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
                ))
            ]),
            {}
        ),
        "Gradient Boosting": (
            Pipeline([
                ("preprocessor", None),
                ("classifier", GradientBoostingClassifier(
                    n_estimators=300, learning_rate=0.1, random_state=RANDOM_STATE
                ))
            ]),
            {}
        ),
        "Extra Trees": (
            Pipeline([
                ("preprocessor", None),
                ("classifier", ExtraTreesClassifier(
                    n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
                ))
            ]),
            {}
        ),
    }


def main():
    args = parse_args()
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_data(args.data)
    print(f"Dataset shape after dropping customerID: {df.shape}")
    print("\nChurn distribution:")
    print(df[TARGET].value_counts().rename(index={0: "No Churn", 1: "Churn"}))

    save_eda(df, out_dir)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted = {}
    model_defs = build_models()

    for name, (base_pipeline, param_grid) in model_defs.items():
        scale_numeric = name.startswith("Logistic Regression")
        pipeline = base_pipeline.set_params(
            preprocessor=make_preprocessor(X_train, scale_numeric=scale_numeric)
        )

        if param_grid:
            estimator = GridSearchCV(
                pipeline, param_grid=param_grid, cv=cv, scoring="roc_auc",
                n_jobs=-1, refit=True
            )
            estimator.fit(X_train, y_train)
            model = estimator.best_estimator_
            cv_auc = float(estimator.best_score_)
            best_params = estimator.best_params_
        else:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            cv_auc = float(np.mean(cv_scores))
            pipeline.fit(X_train, y_train)
            model = pipeline
            best_params = {}

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        row = {
            "Model": name,
            "CV ROC-AUC": cv_auc,
            "Test Accuracy": float(accuracy_score(y_test, predictions)),
            "Precision": float(precision_score(y_test, predictions, zero_division=0)),
            "Recall": float(recall_score(y_test, predictions, zero_division=0)),
            "F1 Score": float(f1_score(y_test, predictions, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, probabilities)),
            "Best Parameters": json.dumps(best_params, sort_keys=True)
        }
        rows.append(row)
        fitted[name] = (model, probabilities, predictions)
        print(f"Finished: {name}")

    results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    results.to_csv(out_dir / "model_comparison.csv", index=False)
    print("\nModel comparison:")
    print(results[["Model", "CV ROC-AUC", "Test Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]].to_string(index=False))

    # Compare model metrics.
    metric_cols = ["Test Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    ax = results.set_index("Model")[metric_cols].plot(kind="bar", figsize=(12, 6))
    ax.set_title("Performance Comparison of Machine Learning Models")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "model_comparison.png", dpi=160)
    plt.close()

    best_name = results.iloc[0]["Model"]
    best_model, best_prob, best_pred = fitted[best_name]
    cm = confusion_matrix(y_test, best_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "best_model_confusion_matrix.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, best_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{best_name} (AUC = {roc_auc_score(y_test, best_prob):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Best Test ROC-AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "best_model_roc_curve.png", dpi=160)
    plt.close(fig)

    report = classification_report(y_test, best_pred, target_names=["No Churn", "Churn"], zero_division=0)
    (out_dir / "best_model_classification_report.txt").write_text(
        f"Best model by held-out test ROC-AUC: {best_name}\n\n{report}", encoding="utf-8"
    )
    metadata = {
        "best_model_by_test_roc_auc": best_name,
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "cross_validation_folds": 5,
        "decision_threshold": 0.5,
        "dataset_path": str(args.data),
        "note": "Metrics are generated locally when the script is run with the dataset."
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nBest model by held-out test ROC-AUC: {best_name}")
    print(f"Results saved to: {out_dir.resolve()}")
    print("\nClassification report:")
    print(report)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
