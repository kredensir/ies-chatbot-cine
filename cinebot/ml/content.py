"""Modelo de recomendación basado en contenido (TF-IDF + similitud coseno).

Ventajas:
- resuelve el arranque en frío (cold start) de usuarios nuevos,
- permite explicar por qué se recomienda un ítem,
- funciona igual para películas y series,
- no necesita valoraciones de otros usuarios.

Limitación: solo captura similitud textual; por eso se combina con el modelo
colaborativo en el recomendador híbrido.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cinebot import config
from cinebot.ml.features import STOPWORDS, catalog_documents


@dataclass
class ContentModel:
    """Artefacto entrenado del recomendador de contenido."""

    vectorizer: TfidfVectorizer
    matrix: csr_matrix
    item_ids: np.ndarray
    item_index: dict[int, int]
    neighbors: np.ndarray
    neighbor_scores: np.ndarray
    documents: list[str] = field(default_factory=list)
    n_features: int = 0
    vocabulary_size: int = 0
    top_k: int = 40

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, catalog: pd.DataFrame, top_k: int = 40) -> "ContentModel":
        documents = catalog_documents(catalog)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            max_features=45_000,
            sublinear_tf=True,
            stop_words=list(STOPWORDS),
            norm="l2",
        )
        matrix = vectorizer.fit_transform(documents).astype(np.float32)
        item_ids = catalog["item_id"].astype(int).to_numpy()
        item_index = {int(item_id): idx for idx, item_id in enumerate(item_ids)}

        neighbors, neighbor_scores = _top_neighbors(matrix, top_k=top_k)
        return cls(
            vectorizer=vectorizer,
            matrix=matrix,
            item_ids=item_ids,
            item_index=item_index,
            neighbors=neighbors,
            neighbor_scores=neighbor_scores,
            documents=documents,
            n_features=int(matrix.shape[0]),
            vocabulary_size=int(matrix.shape[1]),
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------
    def save(self, path: str | None = None) -> str:
        target = path or config.model_path("content")
        joblib.dump(self, target, compress=3)
        return target

    @classmethod
    def load(cls, path: str | None = None) -> "ContentModel":
        source = path or config.model_path("content")
        model = joblib.load(source)
        if not isinstance(model, cls):
            raise TypeError("El artefacto de contenido no tiene el formato esperado")
        return model

    # ------------------------------------------------------------------
    # Vectores
    # ------------------------------------------------------------------
    def _normalize(self, vector: Any) -> Any:
        norm = np.sqrt(vector.multiply(vector).sum()) if hasattr(vector, "multiply") else 0.0
        if norm and float(norm) > 0:
            return vector / float(norm)
        return vector

    def vector_for_item(self, item_id: int):
        index = self.item_index.get(int(item_id))
        if index is None:
            return None
        return self.matrix.getrow(index)

    def user_vector(self, ratings: dict[int, float]):
        """Perfil de contenido a partir de las valoraciones del usuario.

        Las notas altas pesan más; las notas bajas aportan una señal negativa
        pequeña para no contaminar el perfil.
        """
        rows = []
        weights = []
        for item_id, rating in ratings.items():
            index = self.item_index.get(int(item_id))
            if index is None:
                continue
            rating = float(rating)
            weight = (rating - 2.5) / 2.5  # 0.5 -> -0.8; 3 -> 0.2; 5 -> 1.0
            if rating < 3:
                weight = max(0.05, (rating - 3.0) / 5.0)
            if weight <= 0:
                continue
            rows.append(index)
            weights.append(weight)
        if not rows:
            return None
        subset = self.matrix[rows]
        weighted = subset.multiply(np.asarray(weights, dtype=np.float32).reshape(-1, 1))
        vector = np.asarray(weighted.sum(axis=0), dtype=np.float32)
        return self._normalize(vector)

    def query_vector(
        self,
        genres: Iterable[str] = (),
        moods: Iterable[str] = (),
        keywords: Iterable[str] = (),
        text: str = "",
    ):
        terms: list[str] = []
        for genre in genres:
            slug = str(genre).lower().replace("-", "_").replace(" ", "_")
            terms.append(f"genero_{slug}")
        terms.extend(str(m) for m in moods)
        terms.extend(str(k) for k in keywords)
        if text:
            terms.append(str(text))
        if not terms:
            return None
        vector = self.vectorizer.transform([" ".join(terms)])
        return self._normalize(vector)

    # ------------------------------------------------------------------
    # Similitud / puntuación
    # ------------------------------------------------------------------
    def similar_items(self, item_id: int, limit: int = 10) -> list[tuple[int, float]]:
        index = self.item_index.get(int(item_id))
        if index is None:
            return []
        pairs = []
        for neighbor, score in zip(self.neighbors[index], self.neighbor_scores[index]):
            if int(neighbor) >= 0:
                pairs.append((int(self.item_ids[int(neighbor)]), float(score)))
        return pairs[:limit]

    def similarity_between(self, item_a: int, item_b: int) -> float:
        index_a = self.item_index.get(int(item_a))
        index_b = self.item_index.get(int(item_b))
        if index_a is None or index_b is None:
            return 0.0
        return float(self.matrix.getrow(index_a).dot(self.matrix.getrow(index_b).T).toarray()[0, 0])

    def score_candidates(
        self,
        candidate_ids: list[int],
        ratings: dict[int, float] | None = None,
        genres: Iterable[str] = (),
        moods: Iterable[str] = (),
        keywords: Iterable[str] = (),
        query_text: str = "",
        similar_to: int | None = None,
    ) -> dict[int, float]:
        """Devuelve un puntaje de contenido normalizado [0, 1] por candidato."""
        indices = [self.item_index[int(i)] for i in candidate_ids if int(i) in self.item_index]
        valid_ids = [int(i) for i in candidate_ids if int(i) in self.item_index]
        if not indices:
            return {}

        candidate_matrix = self.matrix[indices]
        components: list[tuple[Any, float]] = []

        user_vec = self.user_vector(ratings or {})
        if user_vec is not None:
            components.append((user_vec, 1.0))

        query_vec = self.query_vector(genres=genres, moods=moods, keywords=keywords, text=query_text)
        if query_vec is not None:
            components.append((query_vec, 0.85))

        if similar_to is not None:
            source = self.vector_for_item(similar_to)
            if source is not None:
                components.append((source, 1.25))

        if not components:
            return {item_id: 0.5 for item_id in valid_ids}

        total = np.zeros(len(valid_ids), dtype=np.float32)
        weight_sum = 0.0
        for vector, weight in components:
            sims = cosine_similarity(vector, candidate_matrix).ravel().astype(np.float32)
            if sims.max() > sims.min():
                sims = (sims - sims.min()) / (sims.max() - sims.min())
            total += weight * sims
            weight_sum += weight
        total /= max(weight_sum, 1e-9)
        return {item_id: float(score) for item_id, score in zip(valid_ids, total)}

    def best_explanation_source(
        self, item_id: int, rated_items: Iterable[int], threshold: float = 0.22
    ) -> tuple[int, float] | None:
        """Busca el ítem valorado más parecido para explicar una recomendación."""
        target = self.item_index.get(int(item_id))
        if target is None:
            return None
        best: tuple[int, float] | None = None
        target_row = self.matrix.getrow(target)
        for rated_id in rated_items:
            index = self.item_index.get(int(rated_id))
            if index is None or int(rated_id) == int(item_id):
                continue
            score = float(target_row.dot(self.matrix.getrow(index).T).toarray()[0, 0])
            if best is None or score > best[1]:
                best = (int(rated_id), score)
        if best and best[1] >= threshold:
            return best
        return None


def _top_neighbors(matrix: csr_matrix, top_k: int = 40, block_size: int = 256):
    """Calcula los K vecinos más similares de cada ítem por bloques.

    Se evita materializar la matriz completa de similitud (9.832 x 9.832),
    que ocuparía cientos de MB. Cada bloque se procesa y se descarta.
    """
    n_items = matrix.shape[0]
    k = min(top_k, max(1, n_items - 1))
    neighbors = np.full((n_items, k), -1, dtype=np.int32)
    scores = np.zeros((n_items, k), dtype=np.float32)

    for start in range(0, n_items, block_size):
        end = min(start + block_size, n_items)
        block = cosine_similarity(matrix[start:end], matrix).astype(np.float32)
        for offset in range(end - start):
            row = block[offset]
            row[start + offset] = -1.0  # excluir el propio ítem
            kth = np.argpartition(row, -k)[-k:]
            order = kth[np.argsort(row[kth])[::-1]]
            neighbors[start + offset] = order
            scores[start + offset] = row[order]
    return neighbors, scores
