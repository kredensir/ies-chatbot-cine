"""Punto de entrada de CineBot.

Ejecución de escritorio:
    python app.py

Ejecución en navegador (requiere el paquete web de Flet):
    python app.py --web
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import flet as ft

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cinebot import config  # noqa: E402
from cinebot.bootstrap import ensure_models  # noqa: E402
from cinebot.ui import theme  # noqa: E402
from cinebot.ui.app_view import CineBotView  # noqa: E402


def _loading_view() -> ft.Container:
    return ft.Container(
        expand=True,
        bgcolor=theme.BG,
        alignment=ft.Alignment.CENTER,
        content=ft.Column(
            controls=[
                ft.ProgressRing(width=42, height=42, stroke_width=3, color=theme.PRIMARY),
                ft.Text(
                    "Cargando CineBot…",
                    size=16,
                    weight=ft.FontWeight.W_600,
                    color=theme.TEXT,
                ),
                ft.Text(
                    "Cargando catálogo, modelos y perfiles de demostración",
                    size=11,
                    color=theme.TEXT_MUTED,
                ),
            ],
            spacing=12,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )


def _error_view(message: str, detail: str = "") -> ft.Container:
    controls: list[ft.Control] = [
        ft.Icon(ft.Icons.ERROR_OUTLINE, color=theme.CORAL, size=44),
        ft.Text(
            "No se pudo iniciar CineBot",
            size=22,
            weight=ft.FontWeight.W_600,
            color=theme.TEXT,
        ),
        ft.Text(message, size=13, color=theme.TEXT_MUTED, text_align=ft.TextAlign.CENTER),
        ft.Container(
            padding=14,
            bgcolor=theme.SURFACE_ALT,
            border=ft.Border.all(1, theme.BORDER),
            border_radius=ft.BorderRadius.all(12),
            content=ft.Text(
                "python scripts/download_data.py --series\n"
                "python scripts/build_series_catalog.py\n"
                "python scripts/build_database.py --force\n"
                "python scripts/train_models.py",
                size=11,
                color=theme.TEXT,
                font_family="Consolas",
                selectable=True,
            ),
        ),
    ]
    if detail:
        controls.append(
            ft.ExpansionTile(
                title=ft.Text("Detalle técnico", size=11, color=theme.TEXT_DIM),
                controls=[
                    ft.Text(detail[-4000:], size=9, color=theme.TEXT_DIM, font_family="Consolas", selectable=True)
                ],
            )
        )
    return ft.Container(
        expand=True,
        bgcolor=theme.BG,
        alignment=ft.Alignment.CENTER,
        padding=40,
        content=ft.Column(
            controls=controls,
            spacing=14,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            width=620,
        ),
    )


def main(page: ft.Page) -> None:
    page.title = f"{config.APP_NAME} · {config.APP_TAGLINE}"
    page.theme_mode = ft.ThemeMode.DARK if theme.is_dark() else ft.ThemeMode.LIGHT
    page.theme = ft.Theme(font_family="Segoe UI", color_scheme_seed=theme.PRIMARY)
    page.dark_theme = ft.Theme(font_family="Segoe UI", color_scheme_seed=theme.PRIMARY)
    page.bgcolor = theme.BG
    page.padding = 0
    page.spacing = 0
    page.window.width = 1360
    page.window.height = 700
    page.window.min_width = 900
    page.window.min_height = 600
    page.window.maximized = True
    page.add(_loading_view())
    page.update()

    try:
        ensure_models()
        view = CineBotView(page)
    except Exception as exc:  # noqa: BLE001
        page.controls.clear()
        page.add(
            _error_view(
                "Faltan datos o modelos entrenados. Ejecuta los comandos de preparación "
                "desde la carpeta `ia` y vuelve a abrir la aplicación.",
                traceback.format_exc(),
            )
        )
        page.update()
        print(f"[ERROR] {exc}", file=sys.stderr)
        return

    page.controls.clear()
    page.on_resize = view._on_resize
    page.add(view.root)
    page.update()
    view.mount()


def cli() -> int:
    parser = argparse.ArgumentParser(description="CineBot: chatbot de películas y series")
    parser.add_argument("--web", action="store_true", help="Abrir en el navegador")
    parser.add_argument("--port", type=int, default=8550, help="Puerto para el modo web")
    args = parser.parse_args()

    config.ensure_directories()
    if args.web:
        ft.run(
            main,
            view=ft.AppView.WEB_BROWSER,
            host="127.0.0.1",
            port=args.port,
            assets_dir=str(config.ASSETS_DIR),
        )
    else:
        ft.run(main, view=ft.AppView.FLET_APP, assets_dir=str(config.ASSETS_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
