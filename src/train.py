from __future__ import annotations
import json
from pathlib import Path
import joblib
from sklearn.model_selection import GridSearchCV

from src.paths import ARTIFACTS_DIR, MODEL_PATH, METRICS_PATH
from src.data import load_processed_or_build
from src.features import add_numeric_text_features, clip_outliers
from src.preprocessing import make_splits
from src.modeling import build_model_pipelines
from src.evaluation import evaluate, misclassified_examples

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_processed_or_build()
    df = add_numeric_text_features(df)
    df = clip_outliers(df, col="word_len", upper_q=0.995)

    splits = make_splits(df, label_col="Label", test_size=0.2, val_size=0.1, random_state=42)

    best_name = None
    best_cv = float("-inf")
    best_search = None
    all_results = {}

    for name, (pipe, grid) in build_model_pipelines().items():
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="f1_macro",
            cv=5,
            n_jobs=-1,
            verbose=0
        )
        search.fit(splits.train_X, splits.train_y)

        cv_score = float(search.best_score_)
        val_metrics = evaluate(search.best_estimator_, splits.val_X, splits.val_y)
        test_metrics = evaluate(search.best_estimator_, splits.test_X, splits.test_y)

        all_results[name] = {
            "best_params": search.best_params_,
            "best_cv_f1_macro": cv_score,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        if cv_score > best_cv:
            best_cv = cv_score
            best_name = name
            best_search = search

    final_model = best_search.best_estimator_
    joblib.dump(final_model, MODEL_PATH)

    # store error examples from test split (for report)
    errors = misclassified_examples(final_model, splits.test_X.reset_index(drop=True), splits.test_y, n=8)

    payload = {
        "rows_after_dedup": int(len(df)),
        "label_counts": df["Label"].value_counts().to_dict(),
        "best_model": best_name,
        "best_cv_f1_macro": best_cv,
        "all_results": all_results,
        "error_samples": errors,
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("✅ Saved:", MODEL_PATH)
    print("✅ Saved:", METRICS_PATH)
    print("Best model:", best_name, "| CV macro-F1:", best_cv)

if __name__ == "__main__":
    main()
