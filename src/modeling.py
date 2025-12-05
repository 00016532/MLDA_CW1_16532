from __future__ import annotations
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.preprocessing import build_preprocessor

def build_model_pipelines() -> Dict[str, Tuple[Pipeline, Dict[str, Any]]]:
    pre = build_preprocessor()

    nb = Pipeline([("pre", pre), ("clf", MultinomialNB())])
    nb_grid = {
        "pre__text__tfidf__ngram_range": [(1,1),(1,2)],
        "pre__text__tfidf__min_df": [2,3],
        "pre__text__tfidf__max_df": [0.9,0.95],
        "clf__alpha": [0.3,0.7,1.0,2.0],
    }

    lr = Pipeline([("pre", pre),
                   ("clf", LogisticRegression(
                       max_iter=2000,
                       solver="liblinear",
                       class_weight="balanced",
                       random_state=42
                   ))])
    lr_grid = {
        "pre__text__tfidf__ngram_range": [(1,1),(1,2)],
        "pre__text__tfidf__min_df": [2,3],
        "pre__text__tfidf__max_df": [0.9,0.95],
        "clf__C": [0.3,1.0,3.0,10.0],
    }

    svc = Pipeline([("pre", pre),
                    ("clf", CalibratedClassifierCV(
                        estimator=LinearSVC(class_weight="balanced", random_state=42),
                        method="sigmoid",
                        cv=3
                    ))])
    svc_grid = {
        "pre__text__tfidf__ngram_range": [(1,1),(1,2)],
        "pre__text__tfidf__min_df": [2,3],
        "pre__text__tfidf__max_df": [0.9,0.95],
        "clf__estimator__C": [0.3,1.0,3.0,10.0],
    }

    return {
        "MultinomialNB": (nb, nb_grid),
        "LogisticRegression": (lr, lr_grid),
        "LinearSVC_Calibrated": (svc, svc_grid),
    }
