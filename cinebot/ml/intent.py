"""Clasificador de intenciones del chatbot.

Pipeline:
    TF-IDF de palabras (1-2 gramas) + TF-IDF de caracteres (3-5 gramas)
    -> Regresión logística multinomial.

Se combinan palabras y caracteres para tolerar tildes, abreviaturas y errores
de escritura. La evaluación reporta accuracy, F1 macro, validación cruzada y
matriz de confusión.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from cinebot import config
from cinebot.ml.features import STOPWORDS
from cinebot.ml.intent_data import all_examples
from cinebot.text_utils import normalize_text


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                    stop_words=list(STOPWORDS),
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "clf",
                LogisticRegression(
                    C=6.0,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )


@dataclass
class IntentClassifier:
    pipeline: Pipeline
    labels: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)

    @classmethod
    def fit(cls, texts: list[str] | None = None, labels: list[str] | None = None) -> "IntentClassifier":
        if texts is None or labels is None:
            texts, labels = all_examples()
        texts = [normalize_text(text) for text in texts]
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.20,
            random_state=config.RANDOM_STATE,
            stratify=labels,
        )
        pipeline = build_pipeline()
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        accuracy = float(accuracy_score(y_test, predictions))
        macro_f1 = float(f1_score(y_test, predictions, average="macro"))
        weighted_f1 = float(f1_score(y_test, predictions, average="weighted"))
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        label_order = sorted(set(labels))
        matrix = confusion_matrix(y_test, predictions, labels=label_order)

        cv_scores = cross_val_score(
            build_pipeline(), texts, labels, cv=5, scoring="accuracy", n_jobs=1
        )
        metrics = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "cv_accuracy_mean": float(np.mean(cv_scores)),
            "cv_accuracy_std": float(np.std(cv_scores)),
            "n_examples": float(len(texts)),
            "n_intents": float(len(label_order)),
            "n_features": float(len(pipeline.named_steps["features"].get_feature_names_out())),
        }
        return cls(
            pipeline=pipeline,
            labels=label_order,
            metrics=metrics,
            report=report,
            confusion=matrix.tolist(),
        )

    def save(self, path: str | None = None) -> str:
        target = path or config.model_path("intent")
        joblib.dump(self, target, compress=3)
        return target

    @classmethod
    def load(cls, path: str | None = None) -> "IntentClassifier":
        model = joblib.load(path or config.model_path("intent"))
        if not isinstance(model, cls):
            raise TypeError("El artefacto de intenciones no tiene el formato esperado")
        return model

    def predict(self, text: str) -> tuple[str, float, list[tuple[str, float]]]:
        normalized = normalize_text(text)
        probabilities = self.pipeline.predict_proba([normalized])[0]
        classes = list(self.pipeline.classes_)
        order = np.argsort(probabilities)[::-1]
        top = [(str(classes[i]), float(probabilities[i])) for i in order[:3]]
        return top[0][0], top[0][1], top
