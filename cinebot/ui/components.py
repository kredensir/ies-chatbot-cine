"""Componentes visuales reutilizables de la interfaz Flet."""

from __future__ import annotations

from typing import Callable

import flet as ft

from cinebot.domain import MediaItem, Recommendation
from cinebot.ui import theme

POSTER_WIDTH = 82
POSTER_HEIGHT = 123

def markdown_style() -> ft.MarkdownStyleSheet:
    return ft.MarkdownStyleSheet(
        p_text_style=ft.TextStyle(size=13, color=theme.TEXT),
        strong_text_style=ft.TextStyle(
            size=13, color=theme.TEXT, weight=ft.FontWeight.W_600
        ),
        em_text_style=ft.TextStyle(size=13, color=theme.TEXT_MUTED, italic=True),
        a_text_style=ft.TextStyle(size=13, color=theme.PRIMARY),
        code_text_style=ft.TextStyle(
            size=12, color=theme.TEXT, font_family="Consolas"
        ),
        block_spacing=6,
    )


def _text(value: str, size: int = 13, color: str | None = None, weight=None, **kwargs) -> ft.Text:
    return ft.Text(value=value, size=size, color=color or theme.TEXT, weight=weight, **kwargs)


def pill(text: str, color: str | None = None, bgcolor: str | None = None) -> ft.Container:
    return ft.Container(
        content=ft.Text(
            text,
            size=10,
            color=color or theme.TEXT_MUTED,
            weight=ft.FontWeight.W_600,
        ),
        padding=ft.Padding.symmetric(horizontal=8, vertical=3),
        bgcolor=bgcolor or theme.SURFACE_ALT,
        border_radius=ft.BorderRadius.all(20),
    )


def action_chip(
    label: str,
    on_click: Callable[[], None],
    icon: str | None = None,
    muted: bool = False,
) -> ft.Container:
    controls: list[ft.Control] = []
    if icon:
        controls.append(ft.Icon(icon, size=14, color=theme.TEXT_MUTED))
    controls.append(
        ft.Text(
            label,
            size=11,
            color=theme.TEXT_MUTED if muted else theme.TEXT,
            weight=ft.FontWeight.W_500,
        )
    )
    return ft.Container(
        content=ft.Row(controls=controls, spacing=6, tight=True),
        padding=ft.Padding.symmetric(horizontal=12, vertical=7),
        bgcolor=theme.SURFACE_ALT,
        border_radius=ft.BorderRadius.all(18),
        ink=True,
        ink_color=theme.PRIMARY_SOFT,
        on_click=lambda e: on_click(),
    )


def metric_row(label: str, value: str, color: str | None = None) -> ft.Row:
    return ft.Row(
        controls=[
            ft.Text(label, size=11, color=theme.TEXT_MUTED, expand=True),
            ft.Text(value, size=11, color=color or theme.TEXT, weight=ft.FontWeight.W_600),
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        spacing=8,
    )


def _poster_fallback(item: MediaItem, width: int, height: int) -> ft.Container:
    start, end = theme.palette_for(item)
    initials = "".join(part[0] for part in item.title.split()[:2]).upper() or "?"
    icon = ft.Icons.MOVIE_OUTLINED if item.media_type == "movie" else ft.Icons.TV_OUTLINED
    return ft.Container(
        width=width,
        height=height,
        border_radius=ft.BorderRadius.all(10),
        gradient=ft.LinearGradient(
            colors=[start, end],
            begin=ft.Alignment.TOP_LEFT,
            end=ft.Alignment.BOTTOM_RIGHT,
        ),
        border=ft.Border.all(1, theme.BORDER),
        content=ft.Column(
            controls=[
                ft.Icon(icon, size=17, color=theme.TEXT_MUTED),
                ft.Text(
                    initials,
                    size=20,
                    weight=ft.FontWeight.W_600,
                    color=theme.TEXT,
                ),
                ft.Text(theme.type_label(item.media_type), size=8, color=theme.TEXT_MUTED),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=2,
        ),
        alignment=ft.Alignment.CENTER,
    )


def poster(item: MediaItem, width: int = POSTER_WIDTH, height: int = POSTER_HEIGHT) -> ft.Container:
    fallback = _poster_fallback(item, width, height)
    if not item.poster_url:
        return fallback
    return ft.Container(
        width=width,
        height=height,
        border_radius=ft.BorderRadius.all(10),
        content=ft.Image(
            src=item.poster_url,
            width=width,
            height=height,
            fit=ft.BoxFit.COVER,
            border_radius=ft.BorderRadius.all(10),
            error_content=fallback,
            cache_width=int(width * 2),
            cache_height=int(height * 2),
            semantics_label=f"Portada de {item.title}",
        ),
    )


def recommendation_card(
    rec: Recommendation,
    on_rate: Callable[[int, int], None] | None = None,
    on_feedback: Callable[[int, str], None] | None = None,
    on_details: Callable[[int], None] | None = None,
    max_width: int = 760,
) -> ft.Container:
    item = rec.item
    stars = ft.Row(
        controls=[
            ft.IconButton(
                icon=ft.Icons.STAR_BORDER,
                icon_size=16,
                icon_color=theme.TEXT_DIM,
                tooltip=f"Valorar {value}/5",
                on_click=(lambda e, v=value: on_rate(item.item_id, v)) if on_rate else None,
            )
            for value in range(1, 6)
        ],
        spacing=0,
    )
    actions = ft.Row(
        controls=[
            stars,
            ft.Container(expand=True),
            ft.IconButton(
                icon=ft.Icons.THUMB_UP_OUTLINED,
                icon_size=17,
                icon_color=theme.TEXT_MUTED,
                tooltip="Me gusta",
                on_click=(lambda e: on_feedback(item.item_id, "like")) if on_feedback else None,
            ),
            ft.IconButton(
                icon=ft.Icons.THUMB_DOWN_OUTLINED,
                icon_size=17,
                icon_color=theme.TEXT_MUTED,
                tooltip="No me interesa",
                on_click=(lambda e: on_feedback(item.item_id, "not_interested")) if on_feedback else None,
            ),
            ft.IconButton(
                icon=ft.Icons.INFO_OUTLINE,
                icon_size=17,
                icon_color=theme.TEXT_MUTED,
                tooltip="Ver ficha",
                on_click=(lambda e: on_details(item.item_id)) if on_details else None,
            ),
            ft.IconButton(
                icon=ft.Icons.OPEN_IN_NEW,
                icon_size=16,
                icon_color=theme.PRIMARY if item.external_url else theme.TEXT_DIM,
                tooltip="Abrir ficha externa",
                url=item.external_url or None,
                disabled=not bool(item.external_url),
            ),
        ],
        spacing=0,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    return ft.Container(
        width=max_width,
        padding=13,
        bgcolor=theme.CARD,
        border=ft.Border.all(1, theme.BORDER),
        border_radius=ft.BorderRadius.all(18),
        shadow=[
            ft.BoxShadow(
                blur_radius=16,
                spread_radius=0,
                color=ft.Colors.with_opacity(
                    0.05 if not theme.is_dark() else 0.45, "#000000"
                ),
                offset=ft.Offset(0, 3),
            )
        ],
        content=ft.Row(
            controls=[
                poster(item),
                ft.Column(
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Text(
                                    item.title,
                                    size=15,
                                    weight=ft.FontWeight.W_600,
                                    color=theme.TEXT,
                                    expand=True,
                                    max_lines=2,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                ),
                                pill(theme.type_label(item.media_type), theme.PRIMARY, theme.PRIMARY_SOFT),
                            ],
                            spacing=8,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Text(
                            f"{item.year_label} · {' · '.join(item.genres[:3]) or 'Sin género'} · {item.duration_label}",
                            size=11,
                            color=theme.TEXT_MUTED,
                            max_lines=2,
                            overflow=ft.TextOverflow.ELLIPSIS,
                        ),
                        ft.Row(
                            controls=[
                                pill(f"★ {item.rating_label}", theme.YELLOW_TEXT, theme.YELLOW_SOFT),
                                pill(
                                    f"Predicción {rec.predicted_rating:.1f}/5",
                                    theme.ACCENT_TEXT,
                                    theme.ACCENT_SOFT,
                                ),
                                pill(f"#{rec.rank}", theme.TEXT_MUTED, theme.SURFACE_ALT),
                            ],
                            spacing=6,
                            wrap=True,
                            run_spacing=6,
                        ),
                        ft.Text(
                            rec.reason or "Recomendación personalizada",
                            size=11,
                            color=theme.TEXT_MUTED,
                            max_lines=2,
                            overflow=ft.TextOverflow.ELLIPSIS,
                        ),
                        actions,
                    ],
                    spacing=5,
                    expand=True,
                ),
            ],
            spacing=14,
            vertical_alignment=ft.CrossAxisAlignment.START,
        ),
    )


def chat_bubble(
    text: str,
    is_user: bool,
    recommendations: list[Recommendation] | None = None,
    on_rate: Callable[[int, int], None] | None = None,
    on_feedback: Callable[[int, str], None] | None = None,
    on_details: Callable[[int], None] | None = None,
    max_width: int = 820,
) -> ft.Row:
    if is_user:
        return ft.Row(
            controls=[
                ft.Container(
                    content=ft.Text(text, size=14, color="#FFFFFF", selectable=True),
                    padding=ft.Padding.symmetric(horizontal=16, vertical=11),
                    bgcolor=theme.USER_BUBBLE,
                    border_radius=ft.BorderRadius.only(
                        top_left=19,
                        top_right=19,
                        bottom_left=19,
                        bottom_right=6,
                    ),
                )
            ],
            alignment=ft.MainAxisAlignment.END,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )

    if recommendations:
        children: list[ft.Control] = []
        if text:
            children.append(
                ft.Container(
                    content=ft.Markdown(
                        value=text,
                        selectable=True,
                        auto_follow_links=True,
                        md_style_sheet=markdown_style(),
                    ),
                    padding=ft.Padding.symmetric(horizontal=2, vertical=2),
                )
            )
        for rec in recommendations:
            children.append(
                recommendation_card(
                    rec,
                    on_rate=on_rate,
                    on_feedback=on_feedback,
                    on_details=on_details,
                    max_width=max_width - 4,
                )
            )
        return ft.Row(
            controls=[
                ft.Container(
                    content=ft.Column(controls=children, spacing=10, tight=True),
                    width=max_width,
                )
            ],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )

    return ft.Row(
        controls=[
            ft.Container(
                width=max_width,
                content=ft.Markdown(
                    value=text,
                    selectable=True,
                    auto_follow_links=True,
                    md_style_sheet=markdown_style(),
                ),
                padding=ft.Padding.symmetric(horizontal=15, vertical=11),
                bgcolor=theme.BOT_BUBBLE,
                border=ft.Border.all(1, theme.BOT_BUBBLE_BORDER),
                border_radius=ft.BorderRadius.only(
                    top_left=19,
                    top_right=19,
                    bottom_left=6,
                    bottom_right=19,
                ),
            )
        ],
        alignment=ft.MainAxisAlignment.START,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )


def typing_indicator() -> ft.Row:
    return ft.Row(
        controls=[
            ft.Container(
                content=ft.Row(
                    controls=[
                        ft.ProgressRing(width=14, height=14, stroke_width=2, color=theme.PRIMARY),
                        ft.Text("CineBot está pensando…", size=12, color=theme.TEXT_MUTED),
                    ],
                    spacing=8,
                ),
                padding=ft.Padding.symmetric(horizontal=14, vertical=10),
                bgcolor=theme.BOT_BUBBLE,
                border=ft.Border.all(1, theme.BOT_BUBBLE_BORDER),
                border_radius=ft.BorderRadius.only(
                    top_left=19,
                    top_right=19,
                    bottom_left=6,
                    bottom_right=19,
                ),
            )
        ],
        alignment=ft.MainAxisAlignment.START,
    )


def section_title(text: str) -> ft.Text:
    return ft.Text(
        text.upper(),
        size=9,
        color=theme.TEXT_DIM,
        weight=ft.FontWeight.W_600,
    )


def profile_panel(profile, stats: dict) -> ft.Container:
    initial = (profile.username or "?")[0].upper()
    genres = " · ".join(g.genre for g in profile.top_genres[:4]) or "Sin datos todavía"
    cluster: list[ft.Control] = []
    if profile.cluster_name:
        cluster.append(
            ft.Container(
                content=ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.PSYCHOLOGY_OUTLINED, size=13, color=theme.PRIMARY),
                        ft.Text(profile.cluster_name, size=10, color=theme.PRIMARY),
                    ],
                    spacing=5,
                ),
                padding=ft.Padding.symmetric(horizontal=9, vertical=5),
                bgcolor=theme.PRIMARY_SOFT,
                border_radius=ft.BorderRadius.all(12),
            )
        )
    return ft.Container(
        padding=13,
        bgcolor=theme.SURFACE_ALT,
        border_radius=ft.BorderRadius.all(16),
        content=ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Container(
                            width=40,
                            height=40,
                            border_radius=ft.BorderRadius.all(20),
                            bgcolor=theme.PRIMARY_SOFT,
                            alignment=ft.Alignment.CENTER,
                            content=ft.Text(
                                initial,
                                size=17,
                                weight=ft.FontWeight.W_600,
                                color=theme.PRIMARY,
                            ),
                        ),
                        ft.Column(
                            controls=[
                                ft.Text(
                                    profile.username,
                                    size=13,
                                    weight=ft.FontWeight.W_600,
                                    color=theme.TEXT,
                                ),
                                ft.Text(
                                    "Perfil MovieLens" if profile.is_demo else "Tu perfil local",
                                    size=10,
                                    color=theme.TEXT_MUTED,
                                ),
                            ],
                            spacing=1,
                            expand=True,
                        ),
                    ],
                    spacing=10,
                ),
                ft.Divider(height=1, color=theme.BORDER),
                metric_row("Valoraciones", f"{profile.n_ratings:,}"),
                metric_row("Nota media", f"{profile.avg_rating:.2f}/5" if profile.n_ratings else "—"),
                metric_row("Favoritas", f"{profile.n_likes:,}"),
                ft.Text(
                    genres,
                    size=10,
                    color=theme.TEXT_MUTED,
                    max_lines=2,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
                *cluster,
            ],
            spacing=8,
        ),
    )


def model_panel(metadata: dict, stats: dict) -> ft.Container:
    collab = metadata.get("collaborative", {})
    ranking = metadata.get("ranking", {}).get("hybrid", {})
    intent = metadata.get("intent", {})
    clusters = metadata.get("clusters", {})
    weights = metadata.get("hybrid_weights", {})

    controls: list[ft.Control] = [
        section_title("Modelos entrenados"),
        metric_row("Catálogo", f"{stats.get('items', 0):,}"),
        metric_row("Valoraciones", f"{stats.get('ratings', 0):,}"),
        metric_row("RMSE FunkSVD", f"{collab.get('rmse', 0):.3f}"),
        metric_row("R²", f"{collab.get('r2', 0):.3f}"),
        metric_row("NDCG@10", f"{ranking.get('ndcg@10', 0):.3f}"),
        metric_row("Precision@10", f"{ranking.get('precision@10', 0):.3f}"),
        metric_row("Accuracy NLU", f"{intent.get('accuracy', 0) * 100:.1f}%"),
        metric_row("Clusters K-Means", f"{clusters.get('best_k', 0)}"),
        metric_row(
            "Silueta",
            f"{clusters.get('silhouette', 0):.3f}" if clusters.get("silhouette") else "—",
        ),
    ]
    if weights:
        controls.append(
            ft.Text(
                "Pesos: contenido {:.0%} · CF {:.0%} · calidad {:.0%} · popularidad {:.0%}".format(
                    weights.get("content", 0),
                    weights.get("collaborative", 0),
                    weights.get("quality", 0),
                    weights.get("popularity", 0),
                ),
                size=9,
                color=theme.TEXT_DIM,
            )
        )
    return ft.Container(
        padding=13,
        bgcolor=theme.SURFACE_ALT,
        border_radius=ft.BorderRadius.all(16),
        content=ft.Column(controls=controls, spacing=5),
    )
