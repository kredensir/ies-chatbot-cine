"""CineBot: recomendador conversacional de películas y series.

Paquete principal. Integra:
- ingestión y limpieza de datos (MovieLens + TVmaze),
- persistencia relacional en SQLite,
- modelos de machine learning (TF-IDF, SVD, K-Means, regresión logística),
- un motor híbrido de recomendación,
- una interfaz de chat construida con Flet.
"""

__version__ = "1.0.0"
__author__ = "Cruz Robinson Boris"
