"""Filtrado colaborativo con factorización de matrices (FunkSVD + sesgos).

El modelo aprende un vector latente por usuario y por ítem minimizando el error
cuadrático regularizado mediante descenso de gradiente estocástico (SGD).

    r_hat(u,i) = mu + b_u + b_i + p_u · q_i

Conceptos aplicados:
- sesgo global (mu), sesgo de usuario (b_u) y de ítem (b_i),
- factores latentes (p_u, q_i),
- tasa de aprendizaje, épocas, regularización L2 y early stopping,
- partición train/validación/test,
- "fold-in" para incorporar a un usuario nuevo sin reentrenar todo el modelo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cinebot import config


@dataclass
class UserFactor:
    user_id: int
    bias: float
    vector: np.ndarray
    n_items: int


@dataclass
class SVDModel:
    mu: float
    bu: np.ndarray
    bi: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_index: dict[int, int]
    item_index: dict[int, int]
    params: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        validation: pd.DataFrame | None = None,
        n_factors: int = config.SVD_FACTORS,
        epochs: int = config.SVD_EPOCHS,
        learning_rate: float = config.SVD_LR,
        reg: float = config.SVD_REG,
        patience: int = config.SVD_PATIENCE,
        seed: int = config.RANDOM_STATE,
        verbose: bool = True,
    ) -> "SVDModel":
        rng = np.random.default_rng(seed)
        user_ids = np.sort(train["userId"].unique())
        item_ids = np.sort(train["movieId"].unique())
        user_index = {int(u): i for i, u in enumerate(user_ids)}
        item_index = {int(i): j for j, i in enumerate(item_ids)}

        n_users, n_items = len(user_ids), len(item_ids)
        mu = float(train["rating"].mean())
        bu = np.zeros(n_users, dtype=np.float64)
        bi = np.zeros(n_items, dtype=np.float64)
        P = rng.normal(0.0, 0.05, size=(n_users, n_factors)).astype(np.float64)
        Q = rng.normal(0.0, 0.05, size=(n_items, n_factors)).astype(np.float64)

        users = train["userId"].map(user_index).to_numpy(dtype=np.int64)
        items = train["movieId"].map(item_index).to_numpy(dtype=np.int64)
        ratings = train["rating"].to_numpy(dtype=np.float64)
        n_ratings = len(ratings)

        val_users = val_items = val_ratings = None
        if validation is not None and len(validation):
            mask = validation["userId"].isin(user_index) & validation["movieId"].isin(item_index)
            val = validation.loc[mask]
            if len(val):
                val_users = val["userId"].map(user_index).to_numpy(dtype=np.int64)
                val_items = val["movieId"].map(item_index).to_numpy(dtype=np.int64)
                val_ratings = val["rating"].to_numpy(dtype=np.float64)

        history: list[dict[str, float]] = []
        best_val = float("inf")
        best_state: tuple[np.ndarray, ...] | None = None
        best_epoch = 0
        stale = 0

        for epoch in range(1, epochs + 1):
            order = rng.permutation(n_ratings)
            for pos in order:
                u = users[pos]
                i = items[pos]
                r = ratings[pos]
                pred = mu + bu[u] + bi[i] + float(P[u] @ Q[i])
                error = r - pred
                bu[u] += learning_rate * (error - reg * bu[u])
                bi[i] += learning_rate * (error - reg * bi[i])
                p_u = P[u].copy()
                q_i = Q[i].copy()
                P[u] += learning_rate * (error * q_i - reg * p_u)
                Q[i] += learning_rate * (error * p_u - reg * q_i)

            train_rmse = _rmse(
                mu + bu[users] + bi[items] + np.einsum("ij,ij->i", P[users], Q[items]),
                ratings,
            )
            val_rmse = (
                _rmse(
                    mu + bu[val_users] + bi[val_items]
                    + np.einsum("ij,ij->i", P[val_users], Q[val_items]),
                    val_ratings,
                )
                if val_users is not None
                else train_rmse
            )
            history.append(
                {"epoch": float(epoch), "train_rmse": train_rmse, "val_rmse": val_rmse}
            )
            if verbose:
                print(
                    f"  época {epoch:02d}/{epochs} · train RMSE={train_rmse:.4f} "
                    f"· val RMSE={val_rmse:.4f}"
                )

            if val_rmse < best_val - 1e-4:
                best_val = val_rmse
                best_epoch = epoch
                best_state = (bu.copy(), bi.copy(), P.copy(), Q.copy())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    if verbose:
                        print(f"  Early stopping en época {epoch} (mejor: {best_epoch})")
                    break

        if best_state is not None:
            bu, bi, P, Q = best_state

        return cls(
            mu=mu,
            bu=bu,
            bi=bi,
            P=P,
            Q=Q,
            user_ids=user_ids,
            item_ids=item_ids,
            user_index=user_index,
            item_index=item_index,
            params={
                "n_factors": n_factors,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "reg": reg,
                "patience": patience,
                "seed": seed,
                "n_users": int(n_users),
                "n_items": int(n_items),
                "n_ratings": int(n_ratings),
                "best_epoch": int(best_epoch),
                "best_val_rmse": float(best_val),
            },
            history=history,
            best_epoch=best_epoch,
        )

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------
    def save(self, path: str | None = None) -> str:
        target = path or config.model_path("collaborative")
        joblib.dump(self, target, compress=3)
        return target

    @classmethod
    def load(cls, path: str | None = None) -> "SVDModel":
        model = joblib.load(path or config.model_path("collaborative"))
        if not isinstance(model, cls):
            raise TypeError("El artefacto colaborativo no tiene el formato esperado")
        return model

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------
    def _clip(self, value: float) -> float:
        return float(min(5.0, max(0.5, value)))

    def knows_user(self, user_id: int) -> bool:
        return int(user_id) in self.user_index

    def knows_item(self, item_id: int) -> bool:
        return int(item_id) in self.item_index

    def predict(self, user_id: int, item_id: int) -> float:
        u = self.user_index.get(int(user_id))
        i = self.item_index.get(int(item_id))
        if u is not None and i is not None:
            return self._clip(self.mu + self.bu[u] + self.bi[i] + float(self.P[u] @ self.Q[i]))
        if i is not None:
            return self._clip(self.mu + self.bi[i])
        if u is not None:
            return self._clip(self.mu + self.bu[u])
        return self._clip(self.mu)

    def predict_many(self, user_id: int, item_ids: list[int]) -> dict[int, float]:
        u = self.user_index.get(int(user_id))
        result: dict[int, float] = {}
        known_ids: list[int] = []
        known_idx: list[int] = []
        for item_id in item_ids:
            i = self.item_index.get(int(item_id))
            if i is None:
                result[int(item_id)] = self._clip(self.mu + self.bu[u]) if u is not None else self._clip(self.mu)
            else:
                known_ids.append(int(item_id))
                known_idx.append(i)
        if known_ids and u is not None:
            idx = np.asarray(known_idx, dtype=np.int64)
            predictions = self.mu + self.bu[u] + self.bi[idx] + self.Q[idx] @ self.P[u]
            for item_id, pred in zip(known_ids, predictions):
                result[item_id] = self._clip(float(pred))
        elif known_ids:
            for item_id, i in zip(known_ids, known_idx):
                result[item_id] = self._clip(self.mu + self.bi[i])
        return result

    # ------------------------------------------------------------------
    # Fold-in: usuario nuevo sin reentrenar
    # ------------------------------------------------------------------
    def fold_in_user(self, ratings: dict[int, float], reg: float = 0.15) -> UserFactor | None:
        """Ajusta un vector latente para un usuario nuevo con sus valoraciones.

        Resuelve la regresión ridge:
            p_u = (Q^T Q + λI)^-1 Q^T (r - mu - b_u - b_i)
        donde Q son los factores de los ítems que el usuario valoró.
        """
        pairs = [
            (int(item_id), float(rating))
            for item_id, rating in ratings.items()
            if int(item_id) in self.item_index
        ]
        if not pairs:
            return None
        idx = np.asarray([self.item_index[item_id] for item_id, _ in pairs], dtype=np.int64)
        values = np.asarray([rating for _, rating in pairs], dtype=np.float64)
        user_bias = float(np.sum(values - self.mu - self.bi[idx]) / (len(values) + 5.0))
        residual = values - self.mu - user_bias - self.bi[idx]
        Q_sub = self.Q[idx]
        gram = Q_sub.T @ Q_sub + reg * np.eye(Q_sub.shape[1])
        vector = np.linalg.solve(gram, Q_sub.T @ residual)
        return UserFactor(user_id=-1, bias=user_bias, vector=vector, n_items=len(values))

    def predict_with_factor(self, factor: UserFactor, item_ids: list[int]) -> dict[int, float]:
        result: dict[int, float] = {}
        for item_id in item_ids:
            i = self.item_index.get(int(item_id))
            if i is None:
                result[int(item_id)] = self._clip(self.mu)
            else:
                pred = self.mu + factor.bias + self.bi[i] + float(factor.vector @ self.Q[i])
                result[int(item_id)] = self._clip(pred)
        return result

    def predict_user(
        self,
        user_id: int,
        item_ids: list[int],
        local_ratings: dict[int, float] | None = None,
    ) -> dict[int, float]:
        """Predice para un usuario conocido y, si hay notas locales, las integra."""
        base = self.predict_many(user_id, item_ids)
        if local_ratings:
            factor = self.fold_in_user(local_ratings)
            if factor is not None and len(local_ratings) >= 2:
                folded = self.predict_with_factor(factor, item_ids)
                # Mezcla 50/50: historia histórica + gustos recientes.
                return {
                    item_id: 0.5 * base[item_id] + 0.5 * folded[item_id]
                    for item_id in item_ids
                }
        return base

    # ------------------------------------------------------------------
    # Métricas de rating
    # ------------------------------------------------------------------
    def evaluate(self, test: pd.DataFrame) -> dict[str, float]:
        mask = test["userId"].isin(self.user_index) & test["movieId"].isin(self.item_index)
        data = test.loc[mask]
        if data.empty:
            return {"rmse": float("nan"), "mae": float("nan"), "n": 0.0}
        u = data["userId"].map(self.user_index).to_numpy(dtype=np.int64)
        i = data["movieId"].map(self.item_index).to_numpy(dtype=np.int64)
        pred = self.mu + self.bu[u] + self.bi[i] + np.einsum("ij,ij->i", self.P[u], self.Q[i])
        pred = np.clip(pred, 0.5, 5.0)
        real = data["rating"].to_numpy(dtype=np.float64)
        error = pred - real
        return {
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "n": float(len(data)),
        }


def _rmse(predicted: np.ndarray, real: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(real)) ** 2)))


def split_ratings(
    ratings: pd.DataFrame,
    train_fraction: float = config.TRAIN_FRACTION,
    valid_fraction: float = config.VALID_FRACTION,
    test_fraction: float = config.TEST_FRACTION,
    seed: int = config.RANDOM_STATE,
    min_ratings: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partición por usuario para que todos aporten a train/val/test."""
    rng = np.random.default_rng(seed)
    train_parts, valid_parts, test_parts = [], [], []
    for _, group in ratings.groupby("userId", sort=False):
        indices = np.arange(len(group))
        rng.shuffle(indices)
        n = len(indices)
        if n < min_ratings:
            train_parts.append(group)
            continue
        n_test = max(1, int(round(n * test_fraction)))
        n_valid = max(1, int(round(n * valid_fraction)))
        n_train = n - n_test - n_valid
        if n_train < 1:
            train_parts.append(group)
            continue
        train_parts.append(group.iloc[indices[:n_train]])
        valid_parts.append(group.iloc[indices[n_train : n_train + n_valid]])
        test_parts.append(group.iloc[indices[n_train + n_valid :]])
    train = pd.concat(train_parts, ignore_index=True) if train_parts else ratings.iloc[0:0]
    valid = pd.concat(valid_parts, ignore_index=True) if valid_parts else ratings.iloc[0:0]
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0]
    return train, valid, test
