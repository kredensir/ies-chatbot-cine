# CineBot

**Chatbot inteligente de recomendación de películas y series con Python, Flet, SQLite y aprendizaje automático**

---

Trabajo práctico integrador

Programación · Bases de Datos · Inteligencia Artificial

Autor/a: ______________________________________

Docente: ______________________________________

Fecha: 25/10/2026

Carpeta del proyecto: ia/

---

## Resumen ejecutivo

CineBot es una aplicación de escritorio y web construida con Python y Flet que recomienda películas y series mediante lenguaje natural. El sistema integra tres áreas del trabajo: **programación** orientada a objetos y por capas, **bases de datos** relacionales con SQLite, y **machine learning** clásico con modelos entrenados localmente. No utiliza un LLM externo ni depende de claves de API: todas las respuestas se generan a partir de datos, reglas y modelos que se ejecutan en el equipo del usuario.

El catálogo reúne 9.832 títulos (9.742 películas y 90 series). Las películas y las 100.836 valoraciones históricas provienen de MovieLens latest-small; las series se enriquecieron con metadatos públicos de TVmaze y sinopsis originales en español. Sobre esos datos se entrenaron un modelo de contenido TF-IDF, un modelo colaborativo FunkSVD con sesgos, un recomendador híbrido con pesos ajustados por validación, una segmentación de usuarios con K-Means y un clasificador de intenciones con regresión logística.

El recomendador colaborativo alcanzó un RMSE de **0.8503** frente a **1.0355** de la media global, una mejora del **17.88 %**. El modelo híbrido obtuvo un NDCG@10 de **0.0411**, superando al contenido puro (0.0179) y al colaborativo puro (0.0218). El clasificador de intenciones logró una accuracy de **58.67%** sobre un corpus sintético de 18 intenciones, y el clustering de usuarios eligió k = **3** con una silueta de **0.1253**.

El resultado es una aplicación funcional, reproducible y explicable: cada recomendación muestra sus motivos, cada conversación queda registrada en la base de datos y cada métrica se puede regenerar con un comando. El documento describe las decisiones de diseño, los fundamentos conceptuales, la implementación, los experimentos y las conclusiones del trabajo.

## 1. Introducción

### 1.1. Contexto y motivación

La oferta de cine y series creció hasta volverse difícil de explorar. Los sistemas de recomendación reducen esa sobrecarga al estimar qué contenido puede interesar a cada persona. Detrás de una recomendación aparentemente simple conviven problemas de representación de datos, recuperación de información, predicción de preferencias, evaluación experimental e interacción con el usuario.

CineBot propone una solución completa y de escala académica: un chatbot que entiende pedidos en español, consulta un catálogo heterogéneo, combina señales de contenido y de comportamiento, y presenta los resultados en una interfaz gráfica. El objetivo no es competir con un servicio industrial, sino demostrar de forma integrada los conceptos de las materias mediante una aplicación real, ejecutable y medible.

### 1.2. Problema abordado

- Un usuario nuevo no tiene historial: el sistema debe resolver el **arranque en frío**.
- Un usuario con historial necesita recomendaciones personalizadas y no repetitivas.
- El catálogo mezcla películas y series con metadatos incompletos y de distinta naturaleza.
- El lenguaje natural es ambiguo: “algo para reír” combina un ánimo con un género.
- Toda recomendación debe poder explicarse y quedar registrada para auditoría.

### 1.3. Objetivos

**Objetivo general:** construir una aplicación de chatbot que recomiende películas y series, aplicando conceptos de programación, bases de datos y machine learning sobre datos reales.

**Objetivos específicos:**

- Integrar dos fuentes de datos reales en un catálogo unificado y reproducible.
- Diseñar una base SQLite normalizada con integridad, vistas, triggers y búsqueda full-text.
- Entrenar y evaluar un recomendador híbrido de contenido y filtrado colaborativo.
- Implementar un clasificador de intenciones en español y un motor de diálogo con estado.
- Desarrollar una interfaz Flet funcional, responsive y con explicaciones visibles.
- Documentar el proceso, las métricas y las limitaciones de forma verificable.

### 1.4. Alcance

El sistema funciona de forma local y monousuario para el perfil de invitado, con perfiles de demostración de MovieLens para probar el filtrado colaborativo. No incluye autenticación, recomendación en tiempo real a gran escala ni sinopsis de películas. La arquitectura, sin embargo, deja preparadas las tablas y los puntos de extensión para una versión multiusuario.

## 2. Marco conceptual

### 2.1. Programación

La aplicación se organiza en capas: presentación (Flet), diálogo (ChatEngine), comprensión del lenguaje (NLU), dominio, acceso a datos y machine learning. Las dependencias apuntan hacia el dominio, no al revés. Esto permite probar el motor por consola sin abrir una ventana y cambiar la interfaz sin reentrenar modelos.

- **POO y dataclasses:** `MediaItem`, `UserProfile`, `QueryFilters`, `NLUResult`, `Recommendation` y `ChatResponse` forman un contrato explícito entre capas.
- **Módulos y paquetes:** `cinebot.data`, `cinebot.db`, `cinebot.ml`, `cinebot.chatbot` y `cinebot.ui` separan responsabilidades.
- **Type hints y `from __future__ import annotations`:** documentan las firmas y facilitan el análisis estático.
- **Manejo de errores:** el motor captura fallos de una consulta y devuelve una respuesta comprensible en lugar de cerrar la aplicación.
- **Concurrencia:** Flet ejecuta el procesamiento del mensaje en un hilo aparte para no bloquear la interfaz.
- **Persistencia:** repositorios especializados encapsulan SQL y evitan dispersar consultas por la UI.

### 2.2. Bases de datos

Se usa SQLite por ser un motor relacional completo, transaccional y sin servidor, ideal para un trabajo reproducible. El esquema aplica el modelo relacional, la normalización hasta tercera forma normal para el catálogo, claves primarias y foráneas, restricciones de dominio, índices y transacciones ACID. Además se usan objetos avanzados: vistas, triggers y una tabla virtual FTS5.

- **Normalización:** los géneros no se guardan como texto repetido; se almacenan en `genres` y se relacionan mediante `item_genres`.
- **Integridad referencial:** `ON DELETE CASCADE` evita registros huérfanos y `CHECK` limita tipos, notas y eventos.
- **Vistas:** encapsulan consultas frecuentes y simplifican la UI (`v_user_stats`, `v_user_genre_taste`, `v_top_rated`).
- **Triggers:** recalculan los agregados locales de un ítem cuando el usuario califica o borra una nota.
- **FTS5:** permite búsquedas por texto completo con tokenización Unicode y eliminación de diacríticos.
- **Trazabilidad:** conversaciones, mensajes, recomendaciones e interacciones quedan persistidos.

### 2.3. Machine learning

Se combinan técnicas supervisadas y no supervisadas. El filtrado colaborativo y el clasificador de intenciones son supervisados; K-Means es no supervisado. El recomendador de contenido es una técnica de recuperación de información basada en representaciones vectoriales. La evaluación distingue predicción de notas (regresión) de ordenamiento de recomendaciones (ranking).

- **TF-IDF:** convierte texto en vectores ponderados por frecuencia e inversa de frecuencia documental.
- **Similitud coseno:** mide el ángulo entre vectores y es apropiada para textos de longitud variable.
- **FunkSVD:** factorización de matrices con sesgos y SGD, orientada a predecir notas.
- **K-Means:** agrupa usuarios con perfiles de género parecidos.
- **Regresión logística:** clasifica intenciones a partir de n-gramas de palabras y caracteres.
- **Validación:** partición train/validación/test, validación cruzada y early stopping para controlar el sobreajuste.

### 2.4. Procesamiento de lenguaje natural

El NLU separa dos problemas. Primero, un clasificador aprende patrones generales de intención. Segundo, un conjunto de reglas y diccionarios extrae entidades concretas: géneros, ánimos, años, títulos, notas y cantidades. Esta división es deliberada: el modelo generaliza y las reglas garantizan precisión sobre el catálogo. Un paso final de refinamiento corrige la intención cuando hay evidencia léxica fuerte, por ejemplo “de qué trata X” o “califica X con 4”.

## 3. Arquitectura del sistema

La arquitectura sigue un flujo descendente desde la interfaz hasta los datos, con dependencias explícitas y bajo acoplamiento. El siguiente esquema resume las capas y sus responsabilidades.

```text
┌──────────────────────────────────────────────────────────────┐
│                    Interfaz Flet (app.py)                     │
│  perfiles · chat · tarjetas · sugerencias · métricas          │
└───────────────────────────────┬──────────────────────────────┘
                                │ ChatResponse / eventos
┌───────────────────────────────▼──────────────────────────────┐
│               ChatEngine (motor de diálogo)                   │
│  contexto · intenciones · calificaciones · persistencia      │
└───────────────┬───────────────────────────────┬──────────────┘
                │                               │
┌───────────────▼──────────────┐ ┌──────────────▼─────────────┐
│ NLU: intención + entidades   │ │ Recomendador híbrido       │
│ TF-IDF + LogisticRegression  │ │ contenido + FunkSVD +      │
│ reglas y refinamiento        │ │ calidad + popularidad      │
└───────────────┬──────────────┘ └──────────────┬─────────────┘
                │                               │
┌───────────────▼───────────────────────────────▼─────────────┐
│               SQLite: catálogo, usuarios, ratings,           │
│               conversaciones, recomendaciones, métricas      │
└──────────────────────────────────────────────────────────────┘
```

| Capa | Módulos | Responsabilidad |
|---|---|---|
| Presentación | app.py, cinebot/ui | Ventana, chat, tarjetas, eventos y estado visual. |
| Diálogo | chatbot/engine.py | Orquestar NLU, recomendador, acciones y persistencia. |
| NLU | chatbot/nlu.py, ml/intent.py | Predecir intención y extraer entidades. |
| Recomendación | ml/recommender.py | Generar, puntuar, diversificar y explicar candidatos. |
| Modelos | ml/content.py, collaborative.py, clustering.py | Entrenamiento, inferencia y persistencia de artefactos. |
| Datos | db/repository.py, db/connection.py | Consultas parametrizadas, transacciones y mapeo al dominio. |
| Datos fuente | scripts/, data/ | Descarga, limpieza, catálogo y construcción de SQLite. |

*Tabla 1. Capas de la arquitectura y responsabilidades.*

### 3.1. Flujo de una consulta

1. El usuario escribe un mensaje en el campo de texto o pulsa una sugerencia.
1. La interfaz lo envía al `ChatEngine` en un hilo de trabajo.
1. El NLU normaliza el texto, predice la intención y extrae entidades y filtros.
1. El motor decide la acción: recomendar, buscar, informar, calificar, listar historial o resetear.
1. El recomendador consulta candidatos en SQLite, puntúa con los modelos y aplica diversidad.
1. El motor redacta una respuesta, la guarda en `messages` y registra las recomendaciones.
1. La interfaz reemplaza el indicador de escritura por el mensaje, las tarjetas y nuevas sugerencias.

### 3.2. Decisiones de diseño

- El motor no importa Flet: puede probarse por consola y reutilizarse en web o escritorio.
- Los artefactos se serializan con `joblib`; la base y los modelos se versionan mediante `metadata.json`.
- La base incluida permite abrir la app sin descargas; los scripts permiten reconstruirla.
- Las series usan IDs propios (900000 + tvmaze_id) para convivir con los IDs de MovieLens.
- Las notas de la app se separan de las de MovieLens mediante la columna `source`, sin sobrescribir el histórico.

## 4. Datos y preparación

### 4.1. Fuentes

| Fuente | Contenido | Uso en el proyecto |
|---|---|---|
| MovieLens latest-small | 100,836 valoraciones, 9,742 películas, 610 usuarios | Entrenamiento y evaluación del CF; perfiles de demostración. |
| TVmaze API | Metadatos de 90 series seleccionadas | Catálogo de series, temporadas, episodios y géneros. |
| Sinopsis originales | 90 reseñas breves en español | Documentos del modelo de contenido y fichas de series. |
| MovieLens links | tmdbId e imdbId | URL de ficha externa de cada película. |

*Tabla 2. Origen y uso de los datos.*

MovieLens es un conjunto académico ampliamente utilizado. Su versión latest-small contiene valoraciones reales con timestamps y permite experimentar con filtrado colaborativo sin infraestructura pesada. TVmaze aporta metadatos de series, un tipo de contenido que MovieLens no cubre. La combinación obliga a resolver identificadores, géneros y campos faltantes de forma explícita.

### 4.2. Limpieza y unificación

- Normalización de títulos: minúsculas, sin tildes, sin año y reubicación de artículos finales (“Godfather, The” → “godfather”).
- Parseo del año embebido en el título de MovieLens y separación de géneros por el separador `|`.
- Normalización de series: `MINDHUNTER` → `Mindhunter`, `Stargate SG·1` → `Stargate SG-1`, `Shōgun` → `Shogun`.
- Asignación de IDs de series en el rango 900000 para evitar colisiones.
- Eliminación de cadenas `nan` y campos vacíos en sinopsis, URLs y pósters.
- Cálculo de la nota bayesiana y de la popularidad para desempatar títulos con pocos votos.
- Construcción de un único `catalog.csv` con 9.832 filas y columnas homogéneas.

| Indicador | Valor |
|---|---|
| Ítems del catálogo | 9,832 |
| Películas | 9,742 |
| Series | 90 |
| Valoraciones MovieLens | 100,836 |
| Usuarios históricos | 610 |
| Géneros distintos | 25 |
| Valoraciones creadas en la app | 0 |

*Tabla 3. Estadísticas del catálogo y de la base incluida.*

![Figura 1. Composición del catálogo: películas y series.](../reports/figures/catalog_composition.png)

*Figura 1. Composición del catálogo: películas y series.*

![Figura 2. Géneros con más títulos en el catálogo.](../reports/figures/genre_distribution.png)

*Figura 2. Géneros con más títulos en el catálogo.*

![Figura 3. Distribución de las valoraciones de MovieLens.](../reports/figures/ratings_distribution.png)

*Figura 3. Distribución de las valoraciones de MovieLens.*

## 5. Diseño de la base de datos

El esquema se implementó en SQLite 3 y se versiona en `database/schema.sql`. La base `database/cinebot.db` se genera con un script idempotente que crea el esquema, importa el catálogo y carga las valoraciones. El diseño busca integridad, consultas eficientes y trazabilidad del comportamiento del chatbot.

### 5.1. Tablas principales

| Tabla | Función | Claves y restricciones |
|---|---|---|
| users | Perfiles locales y de MovieLens | PK user_id; source e is_demo con CHECK. |
| items | Catálogo unificado de películas y series | PK item_id; media_type con CHECK; índices por año, tipo y nota. |
| genres | Catálogo de géneros | PK genre_id; nombre único. |
| item_genres | Relación muchos a muchos ítem-género | PK compuesta; FKs con ON DELETE CASCADE. |
| ratings | Valoraciones históricas y de la app | PK (user_id, item_id, source); rating entre 0,5 y 5. |
| feedback | Like, dislike, visto, no interesado | UNIQUE por usuario, ítem y acción. |
| interactions | Eventos analíticos | Eventos validados con CHECK. |
| conversations | Sesiones de chat | FK a users; fecha de inicio y fin. |
| messages | Mensajes de usuario y asistente | FK a conversations; intención, confianza y entidades JSON. |
| recommendations | Recomendaciones mostradas | FK a message, user e item; rank, score y motivo. |
| training_runs | Ejecuciones de entrenamiento | Parámetros y totales del run. |
| model_metrics | Métricas por modelo y run | UNIQUE (model, metric, run). |

*Tabla 4. Tablas del esquema relacional.*

### 5.2. Relaciones y normalización

La relación entre ítems y géneros es muchos a muchos y se resuelve con `item_genres`. Las valoraciones relacionan usuarios e ítems y añaden `source` a la clave para que una nota de la app pueda convivir con la nota histórica de MovieLens sin violar la unicidad. Las conversaciones agrupan mensajes y cada mensaje puede generar varias recomendaciones, lo que produce una jerarquía uno-a-muchos y muchos-a-uno bien definida.

```text
-- Integridad y trazabilidad de una recomendación
CREATE TABLE recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id        INTEGER NOT NULL,
    user_id           INTEGER NOT NULL,
    item_id           INTEGER NOT NULL,
    rank              INTEGER NOT NULL,
    score             REAL    NOT NULL,
    reason            TEXT    NOT NULL DEFAULT '',
    FOREIGN KEY (message_id) REFERENCES messages(message_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id)    REFERENCES users(user_id)    ON DELETE CASCADE,
    FOREIGN KEY (item_id)    REFERENCES items(item_id)    ON DELETE CASCADE
);
```

### 5.3. Triggers, vistas y FTS5

Los triggers mantienen `local_rating_sum`, `local_rating_count` y `local_rating_avg` en `items` cada vez que se inserta, actualiza o elimina una nota de la app. Así la lectura de una tarjeta no necesita agregar toda la tabla de valoraciones. Las vistas encapsulan estadísticas y rankings. FTS5 indexa título, título normalizado, sinopsis y palabras clave con eliminación de diacríticos, lo que permite buscar “parasitos” y encontrar “Parasite”.

```text
-- Búsqueda full-text tolerante a tildes
SELECT i.item_id, i.title, i.media_type
FROM items_fts f
JOIN items i ON i.item_id = f.rowid
WHERE items_fts MATCH ?
ORDER BY bm25(items_fts)
LIMIT 10;
```

La conexión usa `PRAGMA foreign_keys = ON`, `journal_mode = WAL` y transacciones explícitas para las operaciones que modifican varias tablas. Las consultas se construyen con parámetros, no con concatenación de cadenas, para evitar inyección SQL.

## 6. Recomendador de contenido

### 6.1. Representación de los ítems

Cada película o serie se transforma en un documento textual. El título se repite tres veces para darle peso; se agregan los géneros canónicos y sus equivalentes en español, las etiquetas de ánimo derivadas del género, las palabras clave, la sinopsis, la década y el tipo de contenido. Esta representación permite que una consulta como “ciencia ficción espacial” se acerque a títulos con esos términos aunque el título no los contenga.

```text
documento = tipo + título×3 + géneros_es + ánimos
          + palabras_clave×2 + sinopsis + década
```

### 6.2. TF-IDF y similitud coseno

Se ajusta un `TfidfVectorizer` con n-gramas de una y dos palabras, `min_df=2`, `max_df=0,85`, `sublinear_tf` y normalización L2. El resultado es una matriz dispersa de **9,832 × 12,715** y un vocabulario de **12,715 términos**. Para cada ítem se precomputan sus 40 vecinos más similares procesando la matriz por bloques, evitando materializar una matriz densa de 9.832 × 9.832.

```text
tf(t,d) = 1 + log(frecuencia(t,d))
idf(t)   = log((1 + N) / (1 + df(t))) + 1
w(t,d)   = tf(t,d) · idf(t) / ||vector(d)||

sim(A,B) = (A · B) / (||A|| · ||B||)
```

El perfil de contenido del usuario se construye como una suma ponderada de los vectores de los ítems que valoró. Las notas altas pesan más; las bajas aportan una señal negativa pequeña y se descartan si no superan el umbral. La consulta combina el perfil del usuario con un vector de géneros, ánimos y palabras clave, y con el ítem de referencia cuando se pide “algo parecido a”.

La principal ventaja es que resuelve el arranque en frío y produce explicaciones directas. La limitación es que solo captura similitud textual: dos películas pueden compartir género y sinopsis sin gustar al mismo público. Por eso se combina con el filtrado colaborativo.

## 7. Filtrado colaborativo con FunkSVD

### 7.1. Modelo

FunkSVD descompone la matriz de valoraciones en factores latentes de usuario y de ítem. El modelo incluye la media global y sesgos por usuario y por ítem, de modo que una película universalmente mejor valorada reciba un sesgo positivo y un usuario exigente reciba uno negativo.

```text
r̂(u,i) = μ + b_u + b_i + p_u · q_i

min Σ (r(u,i) - r̂(u,i))² + λ(||p_u||² + ||q_i||² + b_u² + b_i²)
```

El aprendizaje se realiza con descenso de gradiente estocástico. Para cada valoración se actualizan el error y los parámetros:

```text
e = r(u,i) - r̂(u,i)
b_u ← b_u + α (e - λ b_u)
b_i ← b_i + α (e - λ b_i)
p_u ← p_u + α (e q_i - λ p_u)
q_i ← q_i + α (e p_u - λ q_i)
```

| Hiperparámetro | Valor |
|---|---|
| Factores latentes | 24 |
| Épocas | 35 |
| Tasa de aprendizaje | 0.008 |
| Regularización L2 | 0.08 |
| Paciencia (early stopping) | 6 |
| Usuarios en train | 610 |
| Ítems en train | 8,584 |
| Mejor época | 35 |
| Mejor RMSE de validación | 0.8577 |

*Tabla 5. Configuración del modelo FunkSVD.*

### 7.2. Partición y evaluación

Las valoraciones se dividen por usuario en entrenamiento (70,614), validación (15,111) y prueba (15,111). La partición por usuario garantiza que todos los perfiles aporten a las tres partes y evita que un usuario aparezca solo en test. El modelo se evalúa con RMSE, MAE y R²; la media global se usa como baseline.

| Métrica | Resultado |
|---|---|
| RMSE FunkSVD | 0.8503 |
| MAE FunkSVD | 0.6508 |
| R² FunkSVD | 0.3181 |
| RMSE baseline | 1.0355 |
| Mejora vs baseline | 17.88 % |
| Valoraciones evaluadas | 14,433 |

*Tabla 6. Predicción de notas en el conjunto de test.*

![Figura 4. Curva de aprendizaje de FunkSVD: RMSE de entrenamiento y validación por época.](../reports/figures/svd_learning_curve.png)

*Figura 4. Curva de aprendizaje de FunkSVD: RMSE de entrenamiento y validación por época.*

### 7.3. Fold-in para usuarios nuevos

Un usuario nuevo no tiene fila en la matriz de factores. En lugar de reentrenar todo el modelo, se estima su vector resolviendo una regresión ridge sobre los factores de los ítems que ya valoró:

```text
p_u = (Qᵀ Q + λI)⁻¹ Qᵀ (r - μ - b_u - b_i)
```

Si el usuario tiene al menos dos valoraciones, la predicción mezcla el factor histórico con el factor fold-in. Esto permite que el perfil de invitado mejore a medida que califica títulos sin reentrenar los 610 perfiles de MovieLens.

## 8. Recomendador híbrido

### 8.1. Función de puntuación

El puntaje final combina cuatro señales normalizadas entre 0 y 1. La calidad usa la nota bayesiana de las películas y la nota media de las series; la popularidad se normaliza por el máximo del catálogo. Los pesos se aprendieron con una búsqueda en grilla sobre el conjunto de validación, maximizando NDCG@10.

```text
score(u,i) = w_c · contenido(u,i)
           + w_f · colaborativo(u,i)
           + w_q · calidad(i)
           + w_p · popularidad(i)

sujeto a  w_c + w_f + w_q + w_p = 1
```

| Peso | Valor aprendido |
|---|---|
| Contenido | 0.40 |
| Colaborativo | 0.30 |
| Calidad | 0.20 |
| Popularidad | 0.10 |

*Tabla 7. Pesos del recomendador híbrido elegidos por validación.*

### 8.2. Diversidad, filtros y fallback

- **Filtros:** tipo, géneros, ánimos, rango de años, nota mínima y máximo de temporadas.
- **Exclusión:** no se recomiendan ítems ya vistos, rechazados o valorados.
- **Diversidad:** se resta una penalización proporcional al máximo solapamiento Jaccard de géneros con los ítems ya seleccionados.
- **Fallback progresivo:** si no hay suficientes candidatos, se relajan en orden la nota mínima, la época, el género y el tipo de contenido; la respuesta informa qué se relajó.
- **Series:** al no tener factores colaborativos, el peso del CF se redistribuye un 60 % a contenido y un 40 % a calidad.
- **Cold start:** un usuario con menos de tres valoraciones usa pesos específicos sin colaborativo.

### 8.3. Resultados de ranking

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 | Cobertura |
|---|---|---|---|---|---|
| Popularidad (baseline) | 0.0813 | 0.1024 | 0.1206 | 0.4830 | 0.49% |
| Contenido | 0.0083 | 0.0188 | 0.0179 | 0.0782 | 9.96% |
| Colaborativo | 0.0170 | 0.0200 | 0.0218 | 0.1361 | 4.37% |
| Híbrido | 0.0184 | 0.0386 | 0.0411 | 0.1616 | 8.75% |

*Tabla 8. Métricas de ranking top-10 sobre el conjunto de test.*

![Figura 5. Comparación de modelos de recomendación en test.](../reports/figures/model_comparison.png)

*Figura 5. Comparación de modelos de recomendación en test.*

El baseline de popularidad obtiene un NDCG alto porque el conjunto de test premia títulos populares: muchos usuarios valoran bien las películas más conocidas. El híbrido no supera a ese baseline en precisión, pero sí mejora de forma clara al contenido puro y al colaborativo puro, y mantiene una cobertura y diversidad mayores que popularidad. El resultado refleja una lección habitual en sistemas de recomendación: un baseline simple puede ser muy competitivo cuando los datos son dispersos.

## 9. Segmentación de usuarios con K-Means

### 9.1. Construcción de características

Cada usuario se representa por su nota media en cada género. Para evitar que una sola valoración en un género minoritario domine el perfil, se aplica un suavizado bayesiano con un prior de cinco valoraciones y la media global del género. Después se aplica un **doble centrado**: se resta la media del propio usuario (que captura su severidad o generosidad general) y la media global del género (que captura el nivel de valoración habitual de ese género). El resultado expresa qué géneros prefiere cada persona *por encima de lo esperado*. Las desviaciones se ponderan por log(1 + número de valoraciones) para reducir el ruido de los géneros con pocos datos y luego se estandarizan las columnas antes de aplicar K-Means.

```text
perfil(u,g) = (Σ ratings(u,g) + μ_g · m) / (n(u,g) + m),  m = 5
residuo(u,g) = perfil(u,g) − media_u(perfil) − μ_g + media_global
```

### 9.2. Elección de k y resultados

Se probaron valores de k entre 3 y 8. Para cada uno se calculó la inercia y el coeficiente de silueta; se eligió el k con mayor silueta. PCA en dos componentes se usa solo para visualizar, no para entrenar. Los centroides se interpretan con la media del perfil residual de los miembros, de modo que los géneros mostrados son los que el grupo prefiere por encima de su propia media y de la media del catálogo. La silueta es moderada porque los gustos cinematográficos son continuos y los grupos se solapan; su valor está en comparar alternativas de k y en producir segmentos interpretables, no en demostrar una separación perfecta.

| Parámetro | Valor |
|---|---|
| k elegido | 3 |
| Silueta | 0.1253 |
| Usuarios agrupados | 610 |
| Varianza explicada por PCA | 36.73% |

*Tabla 9. Resultados de la segmentación.*

| Grupo | Nombre interpretable | Géneros afines y nota media |
|---|---|---|
| Grupo 0 | Espectador familiar | Children, Fantasy, Animation. Nota media del grupo: 3.67/5. |
| Grupo 1 | Buscador de rarezas | Film-Noir, Documentary, Musical. Nota media del grupo: 3.42/5. |
| Grupo 2 | Explorador del suspenso | Thriller, Action, Sci-Fi. Nota media del grupo: 3.72/5. |

*Tabla 10. Interpretación de los clusters de usuarios.*

![Figura 6. Segmentación de usuarios con K-Means y proyección PCA en dos dimensiones.](../reports/figures/user_clusters.png)

*Figura 6. Segmentación de usuarios con K-Means y proyección PCA en dos dimensiones.*

El perfil se muestra en la barra lateral de la aplicación y se actualiza al cambiar de usuario. Para un usuario nuevo, `predict_profile` construye su vector con las mismas reglas y lo asigna al centroide más cercano. Esta información se usa como contexto explicativo, no como filtro rígido: el recomendador sigue combinando contenido y comportamiento.

## 10. Comprensión del lenguaje natural

### 10.1. Intenciones y entidades

El clasificador distingue 18 intenciones. Las entidades se extraen con diccionarios y expresiones regulares: géneros (incluidos sinónimos coloquiales), ánimos, años o décadas, cantidades, temporadas, notas y títulos. El matcher de títulos combina alias manuales, coincidencia exacta y difflib para tolerar errores de escritura. En la extracción de cantidades los artículos «un» y «una» no se interpretan como cantidad —«recomiéndame una película» pide el lote habitual de cinco—; para solicitar un único resultado se exige una fórmula explícita como «solo una» o «una sola».

| Familia | Intenciones |
|---|---|
| Social | saludo, despedida, agradecimiento, ayuda, afirmación, rechazo |
| Recomendación | recomendar_pelicula, recomendar_serie, recomendar_genero, recomendar_animo, similar, otra_recomendacion |
| Consulta | informacion, historial, top_valoradas |
| Acción | calificar, resetear |
| Fuera de dominio | desconocido |

*Tabla 11. Familias de intenciones del chatbot.*

### 10.2. Clasificador

Se entrenó una regresión logística multinomial sobre una unión de TF-IDF de palabras (1-2 gramas) y de caracteres (3-5 gramas). Los caracteres ayudan con tildes, abreviaturas y errores de tipeo; las palabras aportan interpretabilidad. Se usó `class_weight='balanced'`, 2.000 iteraciones y validación cruzada de 5 pliegues.

| Métrica | Valor |
|---|---|
| Accuracy | 0.5867 |
| F1 macro | 0.5625 |
| F1 ponderado | 0.5667 |
| Validación cruzada (media) | 0.5764 |
| Desvío de la validación cruzada | 0.0941 |
| Ejemplos etiquetados | 373 |
| Intenciones | 18 |
| Características TF-IDF | 3,921 |

*Tabla 12. Evaluación del clasificador de intenciones.*

![Figura 7. Matriz de confusión del clasificador de intenciones.](../reports/figures/intent_confusion.png)

*Figura 7. Matriz de confusión del clasificador de intenciones.*

La accuracy del clasificador es moderada porque el corpus es sintético, pequeño y contiene intenciones semánticamente cercanas (por ejemplo, recomendar una película genérica frente a recomendar por género). Esa limitación se compensa con el refinamiento por reglas: si el texto contiene un título y una fórmula de calificación, la intención se corrige a `calificar`; si contiene “de qué trata” y un título, se corrige a `informacion`. Así, el sistema final es más robusto que el clasificador aislado.

### 10.3. Estado del diálogo

El motor mantiene el tipo de contenido, los últimos filtros, la última recomendación y una acción pendiente. Gracias a ese estado, “dame otra” recupera el contexto anterior y “5” completa una calificación que había quedado a la espera del título o de la nota. Las respuestas incluyen sugerencias de seguimiento para hacer visible el estado y reducir la ambigüedad.

## 11. Aplicación e interfaz

### 11.1. Interfaz Flet

La interfaz es una ventana oscura con estética cinematográfica. La barra lateral contiene el selector de perfil, las estadísticas del usuario, los atajos de género y el panel de métricas de los modelos. El panel principal muestra el chat, las tarjetas de recomendación, las sugerencias y el compositor. En pantallas angostas la barra lateral se oculta y el chat ocupa el ancho disponible.

- **Tarjetas:** póster generado con degradado e iniciales, año, géneros, duración, nota, predicción, motivo, estrellas, like/dislike, ficha y enlace externo.
- **Chat:** burbujas alineadas, Markdown, desplazamiento automático y un indicador animado mientras el motor procesa.
- **Perfiles:** invitado local y 20 perfiles de MovieLens con historial real.
- **Sugerencias:** chips contextuales que cambian según la última respuesta.
- **Acciones:** nueva conversación, ayuda, borrar valoraciones y cambio de perfil.
- **Métricas:** RMSE, R², NDCG@10, Precision@10, accuracy del NLU y número de clusters.

### 11.2. Ejecución y experiencia de uso

```text
python app.py          # escritorio
python app.py --web    # navegador en http://127.0.0.1:8550
```

En escritorio, Flet abre una ventana nativa; en modo web, FastAPI y Uvicorn sirven la aplicación en localhost. El procesamiento del mensaje se ejecuta con `page.run_thread` para que la ventana siga respondiendo. Las tarjetas y el panel lateral se actualizan al cambiar el perfil o al registrar una nota.

| Acción del usuario | Resultado |
|---|---|
| “recomiéndame una película de ciencia ficción” | Filtra por tipo y género, recomienda y explica. |
| “series de misterio cortas” | Aplica género, tipo y máximo de temporadas. |
| “de qué trata Breaking Bad” | Devuelve ficha, géneros, valoración y sinopsis. |
| “algo parecido a Inception” | Usa similitud de contenido y excluye el título de origen. |
| “califica Inception con 4” | Guarda la nota, actualiza triggers y muestra el perfil. |
| “dame otra” | Recupera filtros y excluye las recomendaciones ya mostradas. |
| “borra mis valoraciones” | Elimina notas de la app y feedback, conserva MovieLens. |

*Tabla 13. Diálogos de ejemplo y comportamiento esperado.*

## 12. Evaluación global y discusión

| Componente | Métrica principal | Resultado |
|---|---|---|
| Predicción de notas | RMSE | 0.8503 |
| Predicción de notas | Mejora vs baseline | 17.88 % |
| Ranking híbrido | NDCG@10 | 0.0411 |
| Ranking híbrido | Hit Rate@10 | 0.1616 |
| Ranking híbrido | Cobertura | 8.75% |
| NLU | Accuracy | 58.67% |
| NLU | F1 macro | 0.5625 |
| Clustering | Silueta | 0.1253 |

*Tabla 14. Resumen de métricas del sistema.*

**Fortalezas observadas:**

- El sistema es completamente local, reproducible y no depende de servicios de pago.
- La separación por capas permitió probar el motor, los modelos y la base de datos de forma independiente.
- El híbrido mejora al contenido y al colaborativo puro en ranking, y conserva explicaciones útiles.
- La base de datos demuestra un uso real de integridad, triggers, vistas, FTS5 y trazabilidad.
- La interfaz traduce概念s técnicos en una experiencia comprensible para un usuario no técnico.

**Limitaciones:**

- MovieLens latest-small tiene 610 usuarios; las métricas de ranking tienen alta varianza y no representan a toda la población.
- Las películas no incluyen sinopsis; la ficha informa esa ausencia en lugar de inventar contenido.
- El catálogo de series es curado y limitado a 90 títulos.
- El corpus de intenciones es sintético; ampliarlo con mensajes reales mejoraría la generalización.
- El baseline de popularidad es muy competitivo en NDCG; el híbrido prioriza diversidad y explicabilidad, no solo precisión.
- La evaluación offline no mide satisfacción real ni retención; sería necesario un estudio con usuarios.

## 13. Conclusiones

CineBot demuestra que es posible integrar programación, bases de datos y machine learning en una aplicación coherente y funcional sin depender de un LLM externo. El proyecto obligó a tomar decisiones en todas las capas: normalizar datos heterogéneos, diseñar un esquema con integridad y trazabilidad, entrenar modelos con métricas adecuadas, manejar el arranque en frío y traducir los resultados a una interfaz conversacional.

Los resultados son razonables para un conjunto académico: el filtrado colaborativo reduce el RMSE un 17.9 % respecto de la media global; el híbrido supera a sus componentes puros en NDCG@10 y mantiene una cobertura del 8.7%; y el chatbot resuelve consultas de recomendación, información, calificación e historial con estado persistente. La evaluación también mostró límites reales, especialmente la dispersión de los datos y la dificultad de superar un baseline de popularidad.

Como trabajo integrador, el valor principal está en la trazabilidad del proceso: cada dato tiene un origen, cada modelo tiene una métrica, cada recomendación tiene una explicación y cada conversación queda registrada. Esa cadena es la que convierte una demo en un sistema defendible.

## 14. Trabajo futuro

- Enriquecer sinopsis de películas con TMDB o Wikipedia manteniendo el modo offline mediante caché.
- Incorporar embeddings semánticos (Sentence Transformers) y comparar con TF-IDF.
- Probar factorización con sesgos temporales y modelos neuronales (NCF).
- Añadir evaluación online con clics, tiempo de visualización y encuestas de satisfacción.
- Convertir la app en multiusuario con autenticación y API REST.
- Internacionalizar la interfaz y el NLU (español/inglés).
- Explicaciones contrafactuales: “si te gustó X, Y es una alternativa más ligera”.

## 15. Mapa de conceptos aplicados

| Área | Concepto | Evidencia en el proyecto |
|---|---|---|
| Programación | POO y dataclasses | domain.py, modelos tipados y contratos entre capas. |
| Programación | Modularidad y bajo acoplamiento | Paquetes data, db, ml, chatbot, ui. |
| Programación | Manejo de errores y concurrencia | Captura en ChatEngine y page.run_thread. |
| Bases de datos | Modelo relacional y 3FN | items, genres e item_genres. |
| Bases de datos | Integridad referencial | FOREIGN KEY, CHECK y ON DELETE CASCADE. |
| Bases de datos | Transacciones | db.transaction() en operaciones compuestas. |
| Bases de datos | Vistas y triggers | v_user_stats, trg_ratings_ai/au/ad. |
| Bases de datos | Full-text search | items_fts con FTS5 y unicode61. |
| Machine learning | TF-IDF y coseno | content.py. |
| Machine learning | Filtrado colaborativo | collaborative.py (FunkSVD). |
| Machine learning | Sistema híbrido | recommender.py y pesos por validación. |
| Machine learning | Clustering | clustering.py (K-Means + PCA + silueta). |
| Machine learning | Clasificación | intent.py (regresión logística). |
| Machine learning | Evaluación | RMSE, MAE, R², Precision, Recall, NDCG, Hit Rate. |
| NLP | Intenciones y entidades | nlu.py y corpus etiquetado. |
| NLP | Estado del diálogo | ChatEngine y acciones pendientes. |

*Tabla 15. Correspondencia entre conceptos de las materias y su implementación.*

## 16. Reproducibilidad

Todos los pasos se ejecutan desde la carpeta `ia`:

```text
python -m pip install -r requirements.txt

python scripts/download_data.py --series
python scripts/build_series_catalog.py
python -m cinebot.data.catalog
python scripts/build_database.py --force --rebuild-catalog
python scripts/train_models.py
python scripts/evaluate_models.py --live
python -m unittest discover -s tests -v
python scripts/make_document.py
```

Los artefactos principales son:

- `database/cinebot.db`: base SQLite lista para usar.
- `models/*.joblib`: modelos de contenido, colaborativo, intenciones y clustering.
- `models/metadata.json`: métricas, hiperparámetros y pesos del híbrido.
- `reports/evaluacion_modelos.md`: informe automático de evaluación.
- `reports/figures/*.png`: figuras del documento.
- `docs/Documento_IA_CineBot.doc` y `.docx`: documento final.

## 17. Bibliografía y fuentes

- GroupLens Research. MovieLens Latest Datasets. https://grouplens.org/datasets/movielens/
- TVmaze. Public API. https://www.tvmaze.com/api
- The Movie Database (TMDB). https://www.themoviedb.org/
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. IEEE Computer.
- Rendle, S. (2010). Factorization Machines. ICDM.
- Järvelin, K., & Kekäläinen, J. (2002). Cumulated Gain-based Evaluation of IR Techniques. ACM TOIS.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.
- Flet. Documentación oficial. https://flet.dev/
- SQLite. SQLite Documentation: FTS5, triggers and views. https://www.sqlite.org/docs.html

### Declaración sobre el uso de datos

MovieLens se utiliza con fines académicos y de investigación, respetando las condiciones del conjunto original. Los metadatos de series provienen de la API pública de TVmaze y las sinopsis en español fueron redactadas para este proyecto. La aplicación no redistribuye imágenes ni sinopsis protegidas de TMDB: solo usa identificadores públicos para enlazar la ficha externa.
