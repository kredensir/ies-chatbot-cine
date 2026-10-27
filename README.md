# CineBot 🎬

**Chatbot de recomendaciones de películas y series** construido con Python, Flet, SQLite y modelos de machine learning entrenados localmente.

CineBot combina un recomendador híbrido (contenido + filtrado colaborativo + calidad + popularidad), un clasificador de intenciones en español y una base de datos relacional que registra conversaciones, valoraciones y recomendaciones mostradas.

> Trabajo práctico de Programación, Bases de Datos e Inteligencia Artificial.
> La aplicación funciona sin claves de API ni servicios externos: los datos y los modelos ya vienen preparados en el repositorio.

---

## 1. Inicio rápido

Requisitos: **Python 3.11 o superior** (probado con Python 3.14 en Windows).

```powershell
cd ia
python -m pip install -r requirements.txt

# Aplicación de escritorio
python app.py

# Opcional: versión web local en http://127.0.0.1:8550
python app.py --web
```

La primera ejecución solo carga la base SQLite y los modelos ya entrenados. No vuelve a descargar ni a entrenar nada.

### Si faltan datos o modelos

Los artefactos incluidos permiten abrir la aplicación directamente. Para reconstruir todo desde cero:

```powershell
cd ia

# 1. Descargar MovieLens y el catálogo de series desde TVmaze
python scripts/download_data.py --series

# 2. Construir el catálogo unificado y la base SQLite
python scripts/build_series_catalog.py
python -m cinebot.data.catalog
python scripts/build_database.py --force --rebuild-catalog

# 3. Entrenar TF-IDF, FunkSVD, K-Means y el clasificador de intenciones
python scripts/train_models.py

# 4. Evaluar y comprobar el sistema en vivo
python scripts/evaluate_models.py --live
```

El entrenamiento completo tarda unos **8 minutos** en un equipo de escritorio. Genera los modelos en `models/`, las figuras en `reports/figures/` y el informe en `reports/evaluacion_modelos.md`.

---

## 2. Funcionalidades

- Recomendaciones de **películas y series** con explicaciones en lenguaje natural.
- Filtros por **género, ánimo, año, nota mínima, cantidad de temporadas y tipo**.
- Frases como *“series de ciencia ficción cortas”*, *“algo para reír”*, *“las mejores de los 90”* o *“parecido a Inception”*.
- Búsqueda de títulos con **FTS5** y tolerancia a artículos y tildes (*“el padrino”* → *Godfather, The*).
- Valoración con estrellas desde las tarjetas, feedback *me gusta / no me interesa* e historial.
- Flujo conversacional con estado: si falta el título o la nota, el bot pregunta y completa la acción.
- Perfiles de demostración de MovieLens para probar el filtrado colaborativo con historiales reales.
- Perfil local de invitado y reinicio de valoraciones.
- Panel con métricas reales de los modelos entrenados.
- Interfaz oscura responsive construida con **Flet 1.0**.

---

## 3. Arquitectura

```text
┌─────────────────────────────────────────────────────────────┐
│                    Interfaz Flet (app.py)                    │
│  sidebar de perfiles · chat · tarjetas · panel de métricas   │
└──────────────────────────────┬──────────────────────────────┘
                               │ ChatResponse / eventos
┌──────────────────────────────▼──────────────────────────────┐
│                  ChatEngine (chatbot/engine.py)              │
│  estado del diálogo · intenciones · entidades · persistencia │
└───────────────┬──────────────────────────────┬──────────────┘
                │                              │
┌───────────────▼──────────────┐  ┌────────────▼─────────────┐
│ NLU (chatbot/nlu.py)         │  │ Recomendador híbrido     │
│ TF-IDF + LogisticRegression  │  │ contenido + FunkSVD +    │
│ reglas de entidades          │  │ calidad + popularidad    │
└───────────────┬──────────────┘  └────────────┬─────────────┘
                │                              │
┌───────────────▼──────────────────────────────▼─────────────┐
│              SQLite (database/cinebot.db)                   │
│  catálogo · usuarios · ratings · conversaciones · métricas  │
└────────────────────────────────────────────────────────────┘
```

### Estructura de carpetas

```text
ia/
├── app.py                     # entrada Flet (escritorio/web)
├── requirements.txt
├── cinebot/
│   ├── config.py              # rutas e hiperparámetros
│   ├── domain.py              # dataclasses del dominio
│   ├── bootstrap.py           # verificación/construcción de artefactos
│   ├── data/catalog.py        # catálogo unificado
│   ├── db/                    # conexión y repositorios SQLite
│   ├── ml/
│   │   ├── features.py        # documentos y stopwords
│   │   ├── content.py         # TF-IDF + similitud coseno
│   │   ├── collaborative.py   # FunkSVD + fold-in
│   │   ├── clustering.py      # K-Means + PCA
│   │   ├── intent.py          # clasificador de intenciones
│   │   ├── evaluation.py      # métricas de ranking
│   │   ├── recommender.py     # recomendador híbrido
│   │   └── training.py        # orquestador de entrenamiento
│   ├── chatbot/               # NLU y motor de diálogo
│   └── ui/                    # tema, componentes y vista Flet
├── data/
│   ├── raw/                   # MovieLens y respuestas de TVmaze
│   ├── curated/               # series_catalog.csv
│   └── processed/             # catalog.csv
├── database/
│   ├── schema.sql
│   └── cinebot.db
├── models/                    # modelos joblib + metadata.json
├── reports/                   # evaluación y figuras
├── scripts/                   # descarga, construcción, entrenamiento
├── tests/                     # pruebas unittest
└── docs/                      # documento del trabajo (.doc/.docx/.md)
```

---

## 4. Datos

### MovieLens `ml-latest-small`

- **100.836 valoraciones** reales de **610 usuarios** sobre **9.742 películas**.
- Cada valoración va de 0,5 a 5 estrellas.
- Se usa para entrenar y evaluar el filtrado colaborativo y para los perfiles de demostración.
- Fuente: [GroupLens Research](https://grouplens.org/datasets/movielens/). Uso académico; consultar la licencia original incluida en `data/raw/ml-latest-small/README.txt`.

### TVmaze + catálogo curado de series

- **90 series** seleccionadas de la API pública de [TVmaze](https://www.tvmaze.com/api).
- Metadatos reales: título, año, géneros, temporadas, episodios, duración y valoración.
- Las sinopsis en español que se muestran en la aplicación fueron redactadas para este proyecto.
- Los identificadores de series son `900000 + tvmaze_id` para no colisionar con los `movieId` de MovieLens.

### Catálogo unificado

`data/processed/catalog.csv` contiene **9.832 ítems**: 9.742 películas + 90 series. Se normalizan títulos, años, géneros y URLs. Las películas enlazan a The Movie Database mediante el `tmdbId` de MovieLens; las series enlazan a TVmaze.

Limitación conocida: MovieLens no incluye sinopsis de películas. La app informa cuando no dispone de ella y, aun así, ofrece la ficha técnica y recomendaciones parecidas.

---

## 5. Base de datos

SQLite 3 con un esquema relacional normalizado. El archivo `database/schema.sql` incluye:

- **12 tablas**: `users`, `items`, `genres`, `item_genres`, `ratings`, `feedback`, `interactions`, `conversations`, `messages`, `recommendations`, `training_runs`, `model_metrics`.
- **3FN** en el catálogo: géneros en tabla propia y tabla puente `item_genres` (muchos a muchos).
- Clave primaria compuesta `(user_id, item_id, source)` en `ratings` para separar las notas históricas de MovieLens de las notas creadas en la app.
- **Claves foráneas**, restricciones `CHECK`, índices y transacciones.
- **Vistas** para estadísticas de usuario, afinidad por género, ranking y recomendaciones recientes.
- **Triggers** que mantienen los agregados locales de cada ítem al insertar, actualizar o borrar valoraciones.
- **FTS5** con `unicode61 remove_diacritics 2` para búsqueda de texto completo en título, sinopsis y palabras clave.
- Tablas de **trazabilidad conversacional**: cada mensaje y cada recomendación mostrada queda registrada.

Consultas de ejemplo:

```sql
-- Afinidad por género de un usuario
SELECT genre, n_ratings, avg_rating, affinity
FROM v_user_genre_taste
WHERE user_id = 414
ORDER BY affinity DESC
LIMIT 5;

-- Últimas recomendaciones mostradas
SELECT username, title, media_type, rank, score, reason, shown_at
FROM v_recent_recommendations
LIMIT 10;
```

---

## 6. Machine learning aplicado

### 6.1 Recomendador de contenido — TF-IDF + coseno

Cada ítem se convierte en un documento con título (con mayor peso), géneros canónicos, equivalentes en español, sinopsis, palabras clave, década, tipo y etiquetas de ánimo. Se aplica TF-IDF con n-gramas (1,2), `min_df=2`, `max_df=0,85`, `sublinear_tf` y normalización L2. La similitud entre ítems se calcula con **similitud coseno** y se precomputan los 40 vecinos más cercanos por bloques para no materializar una matriz de 9.832 × 9.832.

### 6.2 Filtrado colaborativo — FunkSVD con sesgos

Se aprende:

```text
r̂(u,i) = μ + b_u + b_i + p_u · q_i
```

- 24 factores latentes, SGD con `lr = 0,008`, regularización L2 `0,08`, 35 épocas y *early stopping* (paciencia 6).
- Partición por usuario: **70 % train / 15 % validación / 15 % test**.
- Para usuarios nuevos se usa **fold-in**: se resuelve una regresión ridge para estimar `p_u` sin reentrenar todo el modelo.
- Las series no tienen factores colaborativos, por lo que el peso del CF se redistribuye entre contenido y calidad.

### 6.3 Recomendador híbrido

```text
score = w_c · contenido + w_f · colaborativo + w_q · calidad + w_p · popularidad
```

Los pesos se eligen con una búsqueda en grilla sobre el conjunto de validación. Además se aplica:

- filtros por tipo, género, año, nota y temporadas;
- exclusión de ítems vistos o rechazados;
- **MMR aproximado** con solapamiento Jaccard de géneros para diversificar;
- explicaciones basadas en géneros, títulos valorados y consenso colaborativo;
- *fallback* progresivo si los filtros no alcanzan: sin nota → sin época → sin género → sin tipo.

Pesos finales aprendidos: **contenido 0,40 · colaborativo 0,30 · calidad 0,20 · popularidad 0,10**.

### 6.4 Segmentación de usuarios — K-Means + PCA

Cada usuario se representa por su nota media suavizada en cada género (prior bayesiano de 5 valoraciones para evitar conclusiones con un solo título). Sobre ese perfil se aplica un **doble centrado**: se resta la media del propio usuario y la media global del género, de modo que el vector expresa preferencias relativas y no el nivel general de exigencia. Las desviaciones se ponderan por `log(1 + n)` para atenuar géneros con pocos datos, se estandariza la matriz, se prueban `k = 3…8` y se elige el `k` con mayor **coeficiente de silueta**. PCA en 2 componentes permite visualizar los grupos. El perfil se muestra en la barra lateral y se usa como contexto explicativo.

### 6.5 Clasificador de intenciones — TF-IDF + regresión logística

- Corpus sintético en español con **18 intenciones** (saludo, recomendar, similar, información, calificar, historial, resetear, etc.).
- Unión de TF-IDF de palabras (1-2 gramas) y de caracteres (3-5 gramas) para tolerar tildes y errores de escritura.
- Regresión logística multinomial con `class_weight="balanced"`, validación cruzada de 5 pliegues y matriz de confusión.
- Las entidades (géneros, ánimos, años, notas, cantidad y títulos) se extraen con reglas y diccionarios; luego se refina la intención con evidencia léxica. Esto hace el sistema explicable y evita depender de un LLM externo. Los artículos «un»/«una» no cuentan como cantidad («recomiéndame una película» devuelve el lote habitual de cinco); «solo una» o «una sola» sí piden un único resultado.

### 6.6 Métricas de evaluación

- **Regresión de notas:** RMSE, MAE, R² y comparación contra la media global.
- **Ranking top-10:** Precision@10, Recall@10, NDCG@10, Hit Rate@10, cobertura del catálogo y diversidad.
- **Clasificación:** accuracy, F1 macro, F1 ponderado y validación cruzada.
- **Clustering:** inercia, codo y silueta.

Resultados principales del entrenamiento incluido:

| Modelo / métrica | Valor |
|---|---:|
| RMSE baseline (media global) | 1,0355 |
| **RMSE FunkSVD** | **0,8503** |
| MAE FunkSVD | 0,6508 |
| R² FunkSVD | 0,3181 |
| Mejora de RMSE vs baseline | 17,88 % |
| NDCG@10 híbrido | 0,0411 |
| Precision@10 híbrido | 0,0184 |
| Hit Rate@10 híbrido | 0,1616 |
| Cobertura del catálogo | 8,75 % |
| Accuracy del clasificador | 0,5867 |
| F1 macro del clasificador | 0,5625 |
| K-Means (k / silueta) | 3 / 0,1253 |

> El baseline de popularidad obtiene un NDCG@10 alto porque el conjunto de test premia títulos populares. El híbrido mejora claramente al contenido y al colaborativo puro y mantiene una diversidad mucho mayor. El informe completo, con todas las comparaciones y figuras, está en `reports/evaluacion_modelos.md`.

---

## 7. Flujo conversacional

1. El usuario escribe en lenguaje natural.
2. El NLU predice la intención y extrae entidades.
3. El motor mantiene contexto (tipo de contenido, filtros y última recomendación).
4. El recomendador genera candidatos, los puntúa y diversifica.
5. El motor redacta una respuesta explicable y la persiste en SQLite.
6. La interfaz muestra el mensaje, las tarjetas y sugerencias de seguimiento.

Ejemplos:

```text
> series de ciencia ficción cortas
> algo para reír
> de qué trata Breaking Bad
> algo parecido a Inception
> califica Inception con 4
> top 10 series de drama
> borra mis valoraciones
```

---

## 8. Pruebas

```powershell
cd ia
python -m unittest discover -s tests -v
```

Cubren base de datos, triggers, FTS, normalización de títulos, NLU, modelos, recomendador y flujos completos del motor (incluido el flujo de calificación en dos pasos).

Comprobación en vivo de los modelos:

```powershell
python scripts/evaluate_models.py --live
```

---

## 9. Documento del trabajo

En `docs/` se incluyen:

- `Documento_IA_CineBot.doc` — versión Word-compatible (RTF) para entregar.
- `Documento_IA_CineBot.docx` — versión Word moderna con figuras.
- `Documento_IA_CineBot.md` — fuente editable.

Para regenerarlos con las métricas actuales:

```powershell
python scripts/make_document.py
```

---

## 10. Limitaciones y trabajo futuro

- MovieLens es un conjunto académico pequeño (610 usuarios); las métricas de ranking tienen alta varianza.
- Las películas no traen sinopsis; se podría enriquecer con TMDB si se dispone de una clave.
- El catálogo de series es curado (90 títulos), no exhaustivo.
- El clasificador de intenciones se apoya en un corpus sintético; más datos reales etiquetados mejorarían su robustez.
- Próximos pasos: embeddings semánticos, modelo neuronal de factorización, explicaciones contrafactuales, API REST y despliegue web multiusuario.

---

## 11. Créditos y licencias

- **MovieLens**: GroupLens Research, University of Minnesota. Ver `data/raw/ml-latest-small/README.txt`.
- **TVmaze**: datos públicos de la API de TVmaze; respetar sus términos de uso.
- **The Movie Database (TMDB)**: solo se usan los identificadores públicos de MovieLens para enlazar fichas; no se requiere API key.
- Código y sinopsis originales de las series: trabajo del proyecto.
