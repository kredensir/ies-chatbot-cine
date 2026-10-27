"""Paleta y estilos visuales de CineBot."""

from __future__ import annotations

from typing import Any

from cinebot.domain import MediaItem

LIGHT = {
    "BG": "#F2F2F7",
    "SURFACE": "#FFFFFF",
    "SURFACE_ALT": "#F2F2F7",
    "CARD": "#FFFFFF",
    "CARD_HOVER": "#F7F7FA",
    "BORDER": "#E5E5EA",
    "BORDER_LIGHT": "#D1D1D6",
    "TEXT": "#1C1C1E",
    "TEXT_MUTED": "#6C6C70",
    "TEXT_DIM": "#8E8E93",
    "PRIMARY": "#E50914",
    "PRIMARY_SOFT": "#FDECEE",
    "ACCENT": "#34C759",
    "ACCENT_SOFT": "#EAF9EE",
    "ACCENT_TEXT": "#1B7F3A",
    "YELLOW": "#FF9F0A",
    "YELLOW_SOFT": "#FFF4E5",
    "YELLOW_TEXT": "#B25E00",
    "CORAL": "#FF3B30",
    "GREEN": "#34C759",
    "BLUE": "#E50914",
    "PINK": "#FF2D55",
    "PURPLE": "#AF52DE",
    "USER_BUBBLE": "#E50914",
    "BOT_BUBBLE": "#FFFFFF",
    "BOT_BUBBLE_BORDER": "#E5E5EA",
    "TOAST_BG": "#1C1C1E",
    "TOAST_TEXT": "#FFFFFF",
    "GENRE_PALETTES": [
        ("#FFE9EA", "#FFF7F7"),
        ("#FFF0F3", "#FFF8FA"),
        ("#EAF9EE", "#F7FDF8"),
        ("#FFF4E5", "#FFFBF5"),
        ("#F0EEFF", "#FAF9FF"),
        ("#E8F7FA", "#F6FCFD"),
        ("#FFECEB", "#FFF7F6"),
        ("#F5F5F7", "#FFFFFF"),
    ],
}

DARK = {
    "BG": "#0B0B0F",
    "SURFACE": "#141419",
    "SURFACE_ALT": "#1D1D24",
    "CARD": "#18181F",
    "CARD_HOVER": "#22222B",
    "BORDER": "#2A2A33",
    "BORDER_LIGHT": "#3A3A45",
    "TEXT": "#F5F5F7",
    "TEXT_MUTED": "#A1A1AA",
    "TEXT_DIM": "#71717A",
    "PRIMARY": "#E50914",
    "PRIMARY_SOFT": "#2B1114",
    "ACCENT": "#30D158",
    "ACCENT_SOFT": "#11261A",
    "ACCENT_TEXT": "#4ADE80",
    "YELLOW": "#FF9F0A",
    "YELLOW_SOFT": "#2A1E0E",
    "YELLOW_TEXT": "#FFB340",
    "CORAL": "#FF453A",
    "GREEN": "#30D158",
    "BLUE": "#E50914",
    "PINK": "#FF375F",
    "PURPLE": "#BF5AF2",
    "USER_BUBBLE": "#E50914",
    "BOT_BUBBLE": "#18181F",
    "BOT_BUBBLE_BORDER": "#2A2A33",
    "TOAST_BG": "#F2F2F7",
    "TOAST_TEXT": "#1C1C1E",
    "GENRE_PALETTES": [
        ("#2B1114", "#18181F"),
        ("#2A1218", "#18181F"),
        ("#11261A", "#18181F"),
        ("#2A1E0E", "#18181F"),
        ("#201633", "#18181F"),
        ("#13252A", "#18181F"),
        ("#2A1414", "#18181F"),
        ("#1D1D24", "#18181F"),
    ],
}

_MODE = "light"
_PALETTES: dict[str, dict[str, Any]] = {"light": LIGHT, "dark": DARK}


def set_mode(mode: str) -> None:
    """Aplica la paleta clara u oscura a los controles nuevos."""
    global _MODE
    normalized = mode if mode in _PALETTES else "light"
    _MODE = normalized
    globals().update(_PALETTES[normalized])
    PAGE_THEME.clear()
    PAGE_THEME.update(
        {
            "bgcolor": BG,
            "text_color": TEXT,
            "divider_color": BORDER,
            "font_family": "Segoe UI",
        }
    )


def current_mode() -> str:
    return _MODE


def is_dark() -> bool:
    return _MODE == "dark"


def toggle_mode() -> str:
    set_mode("dark" if _MODE == "light" else "light")
    return _MODE


TYPE_ICONS = {"movie": "🎬", "series": "📺"}
PAGE_THEME: dict[str, str] = {}


def palette_for(item: MediaItem) -> tuple[str, str]:
    """Color estable por título para las portadas generadas localmente."""
    if not item.genres:
        return GENRE_PALETTES[item.item_id % len(GENRE_PALETTES)]
    seed = sum(ord(ch) for ch in item.genres[0]) + item.item_id
    return GENRE_PALETTES[seed % len(GENRE_PALETTES)]


def type_label(media_type: str) -> str:
    return "Película" if media_type == "movie" else "Serie"


def type_icon(media_type: str) -> str:
    return TYPE_ICONS.get(media_type, "🎬")


def rating_stars(value: float, total: int = 5) -> str:
    filled = int(round(max(0.0, min(5.0, value))))
    return "★" * filled + "☆" * (total - filled)


set_mode("light")
