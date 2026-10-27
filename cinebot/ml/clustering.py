"""Segmentación de usuarios con K-Means.

Cada usuario se representa por su valoración media en cada género. Para que los
grupos reflejen *preferencias* y no el nivel general de exigencia, el perfil se
centra dos veces: se resta la media del propio usuario y la media global de cada
género. Así un cluster describe "qué géneros prefiere este grupo por encima de
lo habitual", no "qué géneros puntúa más alto todo el mundo". Las desviaciones
se ponderan por log(1 + número de valoraciones) para reducir el ruido de los
géneros con muy pocas observaciones.

Conceptos aplicados:
- vectorización de usuarios,
- suavizado bayesiano y doble centrado de perfiles,
- escalado de características,
- elección de K con codo (inercia) y coeficiente de silueta,
- ajuste, etiquetado e interpretación de centroides,
- proyección PCA para visualización 2D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from cinebot import config


def _user_genre_matrix(
    ratings: pd.DataFrame, catalog: pd.DataFrame
) -> tuple[np.ndarray, list[int], list[str], pd.DataFrame]:
    genres_by_item = {
        int(row.item_id): [
            g for g in str(row.genres).split("|")
            if g and g.lower() not in {"nan", "none", ""}
        ]
        for row in catalog[["item_id", "genres"]].itertuples(index=False)
    }
    exploded = []
    for row in ratings[["userId", "movieId", "rating"]].itertuples(index=False):
        for genre in genres_by_item.get(int(row.movieId), []):
            exploded.append((int(row.userId), genre, float(row.rating)))
    long = pd.DataFrame(exploded, columns=["userId", "genre", "rating"])
    if long.empty:
        raise ValueError("No hay datos suficientes para construir perfiles por género")

    sums = long.pivot_table(
        index="userId", columns="genre", values="rating", aggfunc="sum"
    )
    counts = long.pivot_table(
        index="userId", columns="genre", values="rating", aggfunc="count"
    )
    sums = sums.reindex(sorted(sums.columns), axis=1).fillna(0.0)
    counts = counts.reindex(columns=sums.columns).fillna(0.0)

    # Suavizado bayesiano por género: evita que un usuario que valoró una sola
    # película de Film-Noir parezca un experto en cine negro. El prior son las
    # medias globales y equivale a "5 valoraciones imaginarias".
    global_means = sums.sum(axis=0) / counts.sum(axis=0).replace(0, np.nan)
    global_means = global_means.fillna(float(long["rating"].mean()))
    prior = 5.0
    matrix = (sums + global_means * prior) / (counts + prior)
    return matrix.to_numpy(dtype=np.float64), [int(u) for u in matrix.index], list(matrix.columns), counts


@dataclass
class UserClustering:
    kmeans: KMeans
    scaler: StandardScaler
    pca: PCA
    genre_columns: list[str]
    user_ids: list[int]
    user_labels: np.ndarray
    cluster_names: dict[int, str]
    cluster_descriptions: dict[int, str]
    genre_means: dict[str, float] = field(default_factory=dict)
    grand_mean: float = 0.0
    evaluation: dict[str, Any] = field(default_factory=dict)
    coordinates: np.ndarray | None = None

    @classmethod
    def fit(cls, ratings: pd.DataFrame, catalog: pd.DataFrame, max_k: int = 8) -> "UserClustering":
        matrix, user_ids, genre_columns, counts = _user_genre_matrix(ratings, catalog)

        # Doble centrado: preferencia relativa del usuario por cada género una
        # vez descontados su propia severidad y el nivel medio del género.
        row_means = matrix.mean(axis=1, keepdims=True)
        global_profile = matrix.mean(axis=0)
        grand_mean = float(matrix.mean())
        residual = matrix - row_means - global_profile + grand_mean
        weights = np.log1p(counts.to_numpy(dtype=np.float64))
        features = residual * weights

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        candidates = range(3, max_k + 1)
        inertias: dict[int, float] = {}
        silhouettes: dict[int, float] = {}
        fitted: dict[int, KMeans] = {}
        for k in candidates:
            if len(user_ids) <= k:
                continue
            model = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
            labels = model.fit_predict(scaled)
            inertias[k] = float(model.inertia_)
            silhouettes[k] = float(silhouette_score(scaled, labels, sample_size=min(500, len(user_ids)), random_state=config.RANDOM_STATE))
            fitted[k] = model
        best_k = max(silhouettes, key=silhouettes.get)
        kmeans = fitted[best_k]
        labels = kmeans.labels_
        pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
        coordinates = pca.fit_transform(scaled)

        # Interpretación de centroides: se usa la media del perfil centrado de
        # los miembros, de modo que los géneros mostrados son los que el grupo
        # prefiere por encima de su propia media y de la media del catálogo.
        names: dict[int, str] = {}
        descriptions: dict[int, str] = {}
        for cluster_id in range(best_k):
            mask = labels == cluster_id
            profile = residual[mask].mean(axis=0)
            order = np.argsort(profile)[::-1]
            top = [genre_columns[i] for i in order[:3] if profile[i] > 0.02]
            if not top:
                top = [genre_columns[i] for i in order[:3]]
            avg = float(matrix[mask].mean())
            names[cluster_id] = _cluster_name(top[0])
            descriptions[cluster_id] = (
                f"Géneros afines: {', '.join(top)}. "
                f"Nota media del grupo: {avg:.2f}/5."
            )
        # Si dos grupos comparten nombre, se distinguen por su nivel de exigencia.
        used: dict[str, int] = {}
        global_avg = float(matrix.mean())
        for cluster_id, name in list(names.items()):
            if name in used:
                group_avg = float(matrix[labels == cluster_id].mean())
                suffix = "entusiasta" if group_avg >= global_avg else "exigente"
                names[cluster_id] = f"{name} ({suffix})"
            else:
                used[name] = cluster_id

        genre_means = {
            genre: float(global_profile[idx]) for idx, genre in enumerate(genre_columns)
        }
        evaluation = {
            "best_k": best_k,
            "silhouette": silhouettes[best_k],
            "inertias": {str(k): v for k, v in inertias.items()},
            "silhouettes": {str(k): v for k, v in silhouettes.items()},
            "n_users": len(user_ids),
            "explained_variance": float(pca.explained_variance_ratio_.sum()),
        }
        return cls(
            kmeans=kmeans,
            scaler=scaler,
            pca=pca,
            genre_columns=genre_columns,
            user_ids=user_ids,
            user_labels=labels,
            cluster_names=names,
            cluster_descriptions=descriptions,
            genre_means=genre_means,
            grand_mean=grand_mean,
            evaluation=evaluation,
            coordinates=coordinates,
        )

    def save(self, path: str | None = None) -> str:
        target = path or config.model_path("clusters")
        joblib.dump(self, target, compress=3)
        return target

    @classmethod
    def load(cls, path: str | None = None) -> "UserClustering":
        model = joblib.load(path or config.model_path("clusters"))
        if not isinstance(model, cls):
            raise TypeError("El artefacto de clustering no tiene el formato esperado")
        return model

    def profile_for_user(self, user_id: int) -> tuple[int, str, str] | None:
        try:
            index = self.user_ids.index(int(user_id))
        except ValueError:
            return None
        cluster_id = int(self.user_labels[index])
        return (
            cluster_id,
            self.cluster_names.get(cluster_id, f"Grupo {cluster_id}"),
            self.cluster_descriptions.get(cluster_id, ""),
        )

    def predict_profile(
        self, ratings: dict[int, float], genres_by_item: dict[int, list[str]]
    ) -> tuple[int, str, str] | None:
        """Asigna un perfil a un usuario nuevo a partir de sus valoraciones."""
        values: dict[str, list[float]] = {}
        for item_id, rating in ratings.items():
            for genre in genres_by_item.get(int(item_id), []):
                values.setdefault(genre, []).append(float(rating))
        if not values:
            return None
        prior = 5.0
        # Perfil suavizado en el mismo espacio de géneros que el entrenamiento.
        vector = np.asarray(
            [
                (
                    (float(np.sum(values[genre])) + self.genre_means.get(genre, self.grand_mean) * prior)
                    / (len(values[genre]) + prior)
                )
                if genre in values
                else self.genre_means.get(genre, self.grand_mean)
                for genre in self.genre_columns
            ],
            dtype=np.float64,
        )
        counts = np.asarray(
            [len(values.get(genre, [])) for genre in self.genre_columns],
            dtype=np.float64,
        )
        global_profile = np.asarray(
            [self.genre_means.get(genre, self.grand_mean) for genre in self.genre_columns],
            dtype=np.float64,
        )
        residual = vector - float(vector.mean()) - global_profile + self.grand_mean
        features = (residual * np.log1p(counts)).reshape(1, -1)
        scaled = self.scaler.transform(features)
        cluster_id = int(self.kmeans.predict(scaled)[0])
        return (
            cluster_id,
            self.cluster_names.get(cluster_id, f"Grupo {cluster_id}"),
            self.cluster_descriptions.get(cluster_id, ""),
        )


def _cluster_name(top_genre: str) -> str:
    if top_genre in {"Horror", "Thriller", "Mystery", "Crime"}:
        return "Explorador del suspenso"
    if top_genre in {"Comedy", "Children", "Animation", "Romance", "Family"}:
        return "Espectador familiar"
    if top_genre in {"Sci-Fi", "Fantasy", "Adventure", "Action"}:
        return "Fan de la aventura"
    if top_genre in {"Documentary", "Musical", "Western", "Film-Noir"}:
        return "Buscador de rarezas"
    if top_genre in {"Drama", "War", "History"}:
        return "Cinéfilo de autor"
    return "Cinéfilo ecléctico"
