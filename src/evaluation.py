from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

def evaluate(model, X, y) -> Dict[str, Any]:
    pred = model.predict(X)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    # probabilistic metrics if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        out["roc_auc"] = float(roc_auc_score(y, proba))
        out["pr_auc"] = float(average_precision_score(y, proba))
    return out

def misclassified_examples(model, X, y, n=8):
    pred = model.predict(X)
    idx = np.where(pred != y)[0][:n]
    rows = []
    for i in idx:
        rows.append({"abstract": X.iloc[i]["Abstract"], "true": int(y[i]), "pred": int(pred[i])})
    return rows
