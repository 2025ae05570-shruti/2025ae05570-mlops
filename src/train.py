"""Train, compare, evaluate and persist models; log each run to MLflow."""
import argparse, json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from src.data_processing import load_raw_data, clean_data, FEATURE_COLUMNS, TARGET_COLUMN

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"

def build_preprocessor():
    # Treat binary/categorical clinical codes as categorical; continuous measures are scaled.
    categorical = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    numeric = [c for c in FEATURE_COLUMNS if c not in categorical]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)])

def main(data_path=None):
    data_path = Path(data_path) if data_path else ROOT / "data" / "raw" / "heart.csv"
    df = clean_data(load_raw_data(data_path))
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "logistic_regression": Pipeline([("prep", build_preprocessor()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))]),
        "random_forest": Pipeline([("prep", build_preprocessor()),
            ("model", RandomForestClassifier(class_weight="balanced", random_state=42))])
    }
    param_grid = {
        "logistic_regression": {"model__C": [0.1, 1.0, 10.0]},
        "random_forest": {"model__n_estimators": [150, 300], "model__max_depth": [None, 6, 12]}
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True); MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("heart-disease-classification")
    best_name, best_search, best_auc = None, None, -1
    summary = {}
    for name, pipe in candidates.items():
        with mlflow.start_run(run_name=name):
            search = GridSearchCV(pipe, param_grid[name], scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
            search.fit(X_train, y_train)
            pred = search.predict(X_test)
            proba = search.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, proba)),
                "cv_roc_auc_mean": float(search.best_score_)
            }
            mlflow.log_params(search.best_params_)
            mlflow.log_param("model_type", name)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(search.best_estimator_, artifact_path="model")
            ConfusionMatrixDisplay.from_predictions(y_test, pred)
            plt.title(f"Confusion matrix: {name}")
            cm_path = ARTIFACTS / f"{name}_confusion_matrix.png"
            plt.tight_layout(); plt.savefig(cm_path, dpi=150); plt.close()
            RocCurveDisplay.from_predictions(y_test, proba)
            plt.title(f"ROC curve: {name}")
            roc_path = ARTIFACTS / f"{name}_roc_curve.png"
            plt.tight_layout(); plt.savefig(roc_path, dpi=150); plt.close()
            mlflow.log_artifact(str(cm_path)); mlflow.log_artifact(str(roc_path))
            summary[name] = {"best_params": search.best_params_, **metrics}
            if metrics["roc_auc"] > best_auc:
                best_name, best_search, best_auc = name, search, metrics["roc_auc"]
    joblib.dump(best_search.best_estimator_, MODEL_DIR / "heart_model.joblib")
    (MODEL_DIR / "metadata.json").write_text(json.dumps({
        "model_name": best_name, "features": FEATURE_COLUMNS,
        "target_definition": "1 = target > 0 (heart disease present in source coding); 0 = target == 0",
        "metrics": summary[best_name]
    }, indent=2))
    (ROOT / "reports" / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"selected_model": best_name, "test_metrics": summary[best_name],
                      "saved_model": str(MODEL_DIR / "heart_model.joblib")}, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to CSV dataset")
    args = parser.parse_args()
    main(args.data)
