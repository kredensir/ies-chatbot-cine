-- =====================================================================
-- CineBot - Esquema relacional (SQLite 3)
-- =====================================================================
-- Decisiones de diseño:
--  * `items` unifica películas y series con una clave primaria común.
--  * Los géneros se normalizan en `genres` + tabla puente `item_genres`
--    (relación muchos-a-muchos, 3FN).
--  * `ratings` guarda tanto las valoraciones históricas de MovieLens
--    (source='movielens') como las del usuario de la app (source='app').
--  * `conversations`, `messages`, `recommendations` e `interactions`
--    registran la trazabilidad del chatbot.
--  * `model_metrics` deja evidencia de la evaluación de cada modelo.
--  * Se usan FOREIGN KEY, CHECK, índices, vistas, triggers y FTS5.
-- =====================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

DROP VIEW IF EXISTS v_recent_recommendations;
DROP VIEW IF EXISTS v_top_rated;
DROP VIEW IF EXISTS v_user_genre_taste;
DROP VIEW IF EXISTS v_user_stats;
DROP VIEW IF EXISTS v_catalog;

DROP TABLE IF EXISTS items_fts;
DROP TABLE IF EXISTS interactions;
DROP TABLE IF EXISTS recommendations;
DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS conversations;
DROP TABLE IF EXISTS feedback;
DROP TABLE IF EXISTS ratings;
DROP TABLE IF EXISTS item_genres;
DROP TABLE IF EXISTS genres;
DROP TABLE IF EXISTS model_metrics;
DROP TABLE IF EXISTS training_runs;
DROP TABLE IF EXISTS items;
DROP TABLE IF EXISTS users;

-- ---------------------------------------------------------------------
-- Usuarios
-- ---------------------------------------------------------------------
CREATE TABLE users (
    user_id         INTEGER PRIMARY KEY,
    username        TEXT    NOT NULL UNIQUE,
    source          TEXT    NOT NULL DEFAULT 'local'
                    CHECK (source IN ('local', 'movielens')),
    is_demo         INTEGER NOT NULL DEFAULT 0 CHECK (is_demo IN (0, 1)),
    preferred_genres TEXT   NOT NULL DEFAULT '',
    cluster_id      INTEGER,
    cluster_name    TEXT    NOT NULL DEFAULT '',
    cluster_description TEXT NOT NULL DEFAULT '',
    avatar_color    TEXT    NOT NULL DEFAULT '#7C4DFF',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------------
-- Catálogo unificado de películas y series
-- ---------------------------------------------------------------------
CREATE TABLE items (
    item_id             INTEGER PRIMARY KEY,
    tmdb_id             INTEGER,
    imdb_id             TEXT    NOT NULL DEFAULT '',
    tvmaze_id           INTEGER,
    media_type          TEXT    NOT NULL CHECK (media_type IN ('movie', 'series')),
    title               TEXT    NOT NULL,
    title_norm          TEXT    NOT NULL,
    year                INTEGER CHECK (year IS NULL OR year BETWEEN 1880 AND 2100),
    end_year            INTEGER,
    overview            TEXT    NOT NULL DEFAULT '',
    keywords            TEXT    NOT NULL DEFAULT '',
    runtime             INTEGER,
    seasons             INTEGER,
    episodes            INTEGER,
    rating_avg          REAL    NOT NULL DEFAULT 0 CHECK (rating_avg BETWEEN 0 AND 5),
    rating_count        INTEGER NOT NULL DEFAULT 0 CHECK (rating_count >= 0),
    bayes_rating        REAL    NOT NULL DEFAULT 0 CHECK (bayes_rating BETWEEN 0 AND 5),
    popularity          REAL    NOT NULL DEFAULT 0,
    source              TEXT    NOT NULL DEFAULT 'MovieLens',
    source_url          TEXT    NOT NULL DEFAULT '',
    poster_url          TEXT    NOT NULL DEFAULT '',
    local_rating_sum    REAL    NOT NULL DEFAULT 0,
    local_rating_count  INTEGER NOT NULL DEFAULT 0,
    local_rating_avg    REAL    NOT NULL DEFAULT 0,
    updated_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_items_type       ON items(media_type);
CREATE INDEX idx_items_year       ON items(year);
CREATE INDEX idx_items_title_norm ON items(title_norm);
CREATE INDEX idx_items_bayes      ON items(bayes_rating DESC);
CREATE INDEX idx_items_tmdb       ON items(tmdb_id);

-- ---------------------------------------------------------------------
-- Géneros (muchos a muchos)
-- ---------------------------------------------------------------------
CREATE TABLE genres (
    genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT NOT NULL UNIQUE
);

CREATE TABLE item_genres (
    item_id  INTEGER NOT NULL,
    genre_id INTEGER NOT NULL,
    PRIMARY KEY (item_id, genre_id),
    FOREIGN KEY (item_id)  REFERENCES items(item_id)  ON DELETE CASCADE,
    FOREIGN KEY (genre_id) REFERENCES genres(genre_id) ON DELETE CASCADE
);

CREATE INDEX idx_item_genres_genre ON item_genres(genre_id);

-- ---------------------------------------------------------------------
-- Valoraciones
-- ---------------------------------------------------------------------
CREATE TABLE ratings (
    user_id  INTEGER NOT NULL,
    item_id  INTEGER NOT NULL,
    rating   REAL    NOT NULL CHECK (rating BETWEEN 0.5 AND 5.0),
    rated_at TEXT    NOT NULL DEFAULT (datetime('now')),
    source   TEXT    NOT NULL DEFAULT 'app'
             CHECK (source IN ('app', 'movielens')),
    PRIMARY KEY (user_id, item_id, source),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE
);

CREATE INDEX idx_ratings_user   ON ratings(user_id);
CREATE INDEX idx_ratings_item   ON ratings(item_id);
CREATE INDEX idx_ratings_source ON ratings(source);

-- ---------------------------------------------------------------------
-- Feedback explícito / eventos de interacción (analítica)
-- ---------------------------------------------------------------------
CREATE TABLE feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    item_id     INTEGER NOT NULL,
    action      TEXT    NOT NULL CHECK (action IN ('like', 'dislike', 'seen', 'not_interested')),
    value       REAL    NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (user_id, item_id, action),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE
);

CREATE TABLE interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL,
    item_id        INTEGER,
    event_type     TEXT    NOT NULL
                   CHECK (event_type IN (
                       'view', 'click', 'rate', 'like', 'dislike',
                       'seen', 'not_interested', 'recommendation_shown'
                   )),
    value          REAL,
    metadata_json  TEXT    NOT NULL DEFAULT '{}',
    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE SET NULL
);

CREATE INDEX idx_interactions_user ON interactions(user_id, created_at DESC);

-- ---------------------------------------------------------------------
-- Conversaciones del chatbot
-- ---------------------------------------------------------------------
CREATE TABLE conversations (
    conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    title           TEXT    NOT NULL DEFAULT 'Nueva conversación',
    model_version   TEXT    NOT NULL DEFAULT '1.0.0',
    started_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    ended_at        TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX idx_conversations_user ON conversations(user_id, started_at DESC);

CREATE TABLE messages (
    message_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role            TEXT    NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content         TEXT    NOT NULL,
    intent          TEXT,
    confidence      REAL,
    entities_json   TEXT    NOT NULL DEFAULT '{}',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);

CREATE INDEX idx_messages_conversation ON messages(conversation_id, message_id);

CREATE TABLE recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id        INTEGER NOT NULL,
    user_id           INTEGER NOT NULL,
    item_id           INTEGER NOT NULL,
    rank              INTEGER NOT NULL,
    score             REAL    NOT NULL,
    content_score     REAL    NOT NULL DEFAULT 0,
    cf_score          REAL    NOT NULL DEFAULT 0,
    quality_score     REAL    NOT NULL DEFAULT 0,
    reason            TEXT    NOT NULL DEFAULT '',
    shown_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    clicked           INTEGER NOT NULL DEFAULT 0 CHECK (clicked IN (0, 1)),
    FOREIGN KEY (message_id) REFERENCES messages(message_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id)    REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (item_id)    REFERENCES items(item_id) ON DELETE CASCADE
);

CREATE INDEX idx_recommendations_user ON recommendations(user_id, shown_at DESC);
CREATE INDEX idx_recommendations_item ON recommendations(item_id);

-- ---------------------------------------------------------------------
-- Evidencia de entrenamiento y evaluación
-- ---------------------------------------------------------------------
CREATE TABLE training_runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name  TEXT NOT NULL,
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    n_users     INTEGER,
    n_items     INTEGER,
    n_ratings   INTEGER,
    params_json TEXT NOT NULL DEFAULT '{}',
    notes       TEXT NOT NULL DEFAULT ''
);

CREATE TABLE model_metrics (
    metric_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER,
    model_name  TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value       REAL NOT NULL,
    params_json TEXT NOT NULL DEFAULT '{}',
    evaluated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE,
    UNIQUE (model_name, metric_name, run_id)
);

-- ---------------------------------------------------------------------
-- Vistas
-- ---------------------------------------------------------------------
CREATE VIEW v_catalog AS
SELECT
    i.item_id,
    i.media_type,
    i.title,
    i.title_norm,
    i.year,
    i.end_year,
    i.overview,
    i.keywords,
    i.runtime,
    i.seasons,
    i.episodes,
    i.rating_avg,
    i.rating_count,
    i.bayes_rating,
    i.popularity,
    i.source,
    i.source_url,
    i.poster_url,
    i.tmdb_id,
    i.tvmaze_id,
    i.local_rating_count,
    i.local_rating_avg,
    COALESCE((
        SELECT group_concat(g.name, '|')
        FROM item_genres ig
        JOIN genres g ON g.genre_id = ig.genre_id
        WHERE ig.item_id = i.item_id
    ), '') AS genres
FROM items i;

CREATE VIEW v_user_stats AS
WITH effective AS (
    SELECT
        user_id,
        item_id,
        rating,
        source,
        rated_at,
        ROW_NUMBER() OVER (
            PARTITION BY user_id, item_id
            ORDER BY CASE source WHEN 'app' THEN 0 ELSE 1 END
        ) AS rn
    FROM ratings
)
SELECT
    u.user_id,
    u.username,
    u.source,
    u.is_demo,
    COUNT(e.item_id)                                      AS n_ratings,
    ROUND(COALESCE(AVG(e.rating), 0), 2)                  AS avg_rating,
    COALESCE(SUM(CASE WHEN e.rating >= 4.0 THEN 1 ELSE 0 END), 0) AS n_likes,
    MAX(e.rated_at)                                       AS last_rating_at
FROM users u
LEFT JOIN effective e
       ON e.user_id = u.user_id
      AND e.rn = 1
GROUP BY u.user_id, u.username, u.source, u.is_demo;

CREATE VIEW v_user_genre_taste AS
WITH effective AS (
    SELECT
        user_id,
        item_id,
        rating,
        source,
        ROW_NUMBER() OVER (
            PARTITION BY user_id, item_id
            ORDER BY CASE source WHEN 'app' THEN 0 ELSE 1 END
        ) AS rn
    FROM ratings
)
SELECT
    e.user_id,
    g.name                                AS genre,
    COUNT(*)                              AS n_ratings,
    ROUND(AVG(e.rating), 3)               AS avg_rating,
    ROUND(SUM(e.rating - 3.0), 3)         AS affinity
FROM effective e
JOIN item_genres ig ON ig.item_id = e.item_id
JOIN genres g      ON g.genre_id = ig.genre_id
WHERE e.rn = 1
GROUP BY e.user_id, g.name;

CREATE VIEW v_top_rated AS
SELECT *
FROM (
    SELECT
        i.item_id,
        i.media_type,
        i.title,
        i.year,
        i.rating_avg,
        i.rating_count,
        i.bayes_rating,
        ROW_NUMBER() OVER (
            PARTITION BY i.media_type
            ORDER BY i.bayes_rating DESC, i.rating_count DESC
        ) AS position,
        (
            SELECT group_concat(g.name, '|')
            FROM item_genres ig
            JOIN genres g ON g.genre_id = ig.genre_id
            WHERE ig.item_id = i.item_id
        ) AS genres
    FROM items i
    WHERE i.media_type = 'series' OR i.rating_count >= 25
) ranked
WHERE position <= 100;

CREATE VIEW v_recent_recommendations AS
SELECT
    rec.recommendation_id,
    rec.user_id,
    u.username,
    rec.item_id,
    i.title,
    i.media_type,
    rec.rank,
    ROUND(rec.score, 4) AS score,
    rec.reason,
    rec.shown_at,
    rec.clicked
FROM recommendations rec
JOIN users u ON u.user_id = rec.user_id
JOIN items i ON i.item_id = rec.item_id
ORDER BY rec.shown_at DESC;

-- ---------------------------------------------------------------------
-- Triggers: mantienen agregados locales y el índice FTS5 sincronizado
-- ---------------------------------------------------------------------
CREATE TRIGGER trg_ratings_ai
AFTER INSERT ON ratings
WHEN NEW.source = 'app'
BEGIN
    UPDATE items
    SET local_rating_sum = (
            SELECT COALESCE(SUM(rating), 0) FROM ratings
            WHERE item_id = NEW.item_id AND source = 'app'
        ),
        local_rating_count = (
            SELECT COUNT(*) FROM ratings
            WHERE item_id = NEW.item_id AND source = 'app'
        ),
        local_rating_avg = (
            SELECT COALESCE(AVG(rating), 0) FROM ratings
            WHERE item_id = NEW.item_id AND source = 'app'
        ),
        updated_at = datetime('now')
    WHERE item_id = NEW.item_id;
END;

CREATE TRIGGER trg_ratings_au
AFTER UPDATE ON ratings
WHEN NEW.source = 'app' OR OLD.source = 'app'
BEGIN
    UPDATE items
    SET local_rating_sum = (
            SELECT COALESCE(SUM(rating), 0) FROM ratings
            WHERE item_id = NEW.item_id AND source = 'app'
        ),
        local_rating_count = (
            SELECT COUNT(*) FROM ratings
            WHERE item_id = NEW.item_id AND source = 'app'
        ),
        local_rating_avg = (
            SELECT COALESCE(AVG(rating), 0) FROM ratings
            WHERE item_id = NEW.item_id AND source = 'app'
        ),
        updated_at = datetime('now')
    WHERE item_id = NEW.item_id;
END;

CREATE TRIGGER trg_ratings_ad
AFTER DELETE ON ratings
WHEN OLD.source = 'app'
BEGIN
    UPDATE items
    SET local_rating_sum = (
            SELECT COALESCE(SUM(rating), 0) FROM ratings
            WHERE item_id = OLD.item_id AND source = 'app'
        ),
        local_rating_count = (
            SELECT COUNT(*) FROM ratings
            WHERE item_id = OLD.item_id AND source = 'app'
        ),
        local_rating_avg = (
            SELECT COALESCE(AVG(rating), 0) FROM ratings
            WHERE item_id = OLD.item_id AND source = 'app'
        ),
        updated_at = datetime('now')
    WHERE item_id = OLD.item_id;
END;

-- Búsqueda de texto completo (FTS5) sobre título, sinopsis y palabras clave.
CREATE VIRTUAL TABLE items_fts USING fts5(
    title,
    title_norm,
    overview,
    keywords,
    content='items',
    content_rowid='item_id',
    tokenize="unicode61 remove_diacritics 2"
);

CREATE TRIGGER trg_items_ai AFTER INSERT ON items BEGIN
    INSERT INTO items_fts(rowid, title, title_norm, overview, keywords)
    VALUES (NEW.item_id, NEW.title, NEW.title_norm, NEW.overview, NEW.keywords);
END;

CREATE TRIGGER trg_items_ad AFTER DELETE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, title, title_norm, overview, keywords)
    VALUES ('delete', OLD.item_id, OLD.title, OLD.title_norm, OLD.overview, OLD.keywords);
END;

CREATE TRIGGER trg_items_au AFTER UPDATE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, title, title_norm, overview, keywords)
    VALUES ('delete', OLD.item_id, OLD.title, OLD.title_norm, OLD.overview, OLD.keywords);
    INSERT INTO items_fts(rowid, title, title_norm, overview, keywords)
    VALUES (NEW.item_id, NEW.title, NEW.title_norm, NEW.overview, NEW.keywords);
END;
