"""Vista principal de CineBot construida con Flet 1.0."""

from __future__ import annotations

from typing import Any, Callable

import flet as ft

from cinebot import config
from cinebot.bootstrap import ensure_database
from cinebot.chatbot.engine import ChatEngine
from cinebot.chatbot.nlu import CineBotNLU
from cinebot.db.repository import CatalogRepository, UserRepository
from cinebot.domain import ChatResponse
from cinebot.ml.recommender import HybridRecommender
from cinebot.posters import PosterResolver
from cinebot.ui import components, theme

QUICK_PROMPTS: list[tuple[str, str, str]] = [
    ("Películas", "recomiéndame una película", ft.Icons.MOVIE_OUTLINED),
    ("Series", "recomiéndame una serie", ft.Icons.TV_OUTLINED),
    ("Comedia", "una película de comedia", ft.Icons.SENTIMENT_VERY_SATISFIED_OUTLINED),
    ("Terror", "algo de terror", ft.Icons.NIGHTLIGHT_OUTLINED),
    ("Ciencia ficción", "series de ciencia ficción", ft.Icons.ROCKET_LAUNCH_OUTLINED),
    ("Romance", "algo romántico", ft.Icons.FAVORITE_BORDER),
    ("Familiar", "algo para ver en familia", ft.Icons.FAMILY_RESTROOM),
    ("Sorpréndeme", "sorpréndeme con una película", ft.Icons.AUTO_AWESOME),
]


class CineBotView:
    """Orquesta la interfaz: barra lateral, chat y tarjetas de recomendación."""

    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self.db = ensure_database()
        self.catalog_repo = CatalogRepository(self.db)
        self.user_repo = UserRepository(self.db)
        self.recommender = HybridRecommender(self.db)
        self.poster_resolver = PosterResolver(self.db, self.catalog_repo)
        self.nlu = CineBotNLU(self.catalog_repo.all_titles())

        self.demo_users = self.user_repo.list_demo_users(limit=20)
        self.guest_id = self.user_repo.get_or_create_guest()
        self.user_id = self.guest_id
        self.engine = self._make_engine(self.user_id)
        self.metadata = self.engine.model_metrics()
        self.stats = self.catalog_repo.stats()
        self.busy = False
        self.typing_row: ft.Row | None = None
        self.history: list[dict[str, Any]] = []
        self._suggestions: list[str] = []

        self.root = self._build()
        self._apply_responsive()

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------
    def _make_engine(self, user_id: int) -> ChatEngine:
        return ChatEngine(
            self.db,
            user_id=user_id,
            recommender=self.recommender,
            nlu=self.nlu,
        )

    def mount(self) -> None:
        """Inicia la conversación una vez que la vista ya está en la página."""
        self.page.update()
        self._show_response(self.engine.start())

    def _build(self) -> ft.Row:
        self.profile_holder = ft.Container()
        self.model_holder = ft.Container()
        self.sidebar = self._build_sidebar()
        self.chat_list = ft.ListView(
            expand=True,
            spacing=16,
            padding=ft.Padding.symmetric(horizontal=28, vertical=24),
            auto_scroll=True,
        )
        self.suggestions_row = ft.Row(
            controls=[],
            wrap=True,
            spacing=6,
            run_spacing=6,
        )
        self.input = ft.TextField(
            hint_text="Escribe lo que quieres ver…  ej. «series de misterio cortas»",
            expand=True,
            filled=True,
            bgcolor=theme.CARD,
            fill_color=theme.CARD,
            color=theme.TEXT,
            cursor_color=theme.PRIMARY,
            border_color=theme.BORDER,
            focused_border_color=theme.PRIMARY,
            border_radius=ft.BorderRadius.all(22),
            content_padding=ft.Padding.symmetric(horizontal=16, vertical=13),
            text_size=13,
            prefix_icon=ft.Icons.SEARCH,
            shift_enter=False,
            on_submit=self._on_submit,
        )
        self.send_button = ft.IconButton(
            icon=ft.Icons.SEND,
            icon_color="#FFFFFF",
            bgcolor=theme.PRIMARY,
            width=42,
            height=42,
            tooltip="Enviar",
            on_click=self._on_submit,
        )
        chat_panel = ft.Container(
            expand=True,
            content=ft.Column(
                controls=[
                    self._build_header(),
                    self.chat_list,
                    self._build_composer(),
                ],
                spacing=0,
                expand=True,
            ),
        )
        self._refresh_sidebar()
        return ft.Row(
            controls=[self.sidebar, ft.VerticalDivider(width=1, color=theme.BORDER), chat_panel],
            spacing=0,
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    def _build_header(self) -> ft.Container:
        self.header_stats = ft.Row(
            controls=[
                components.pill(
                    f"{self.stats.get('items', 0):,} títulos",
                    theme.PRIMARY,
                    theme.PRIMARY_SOFT,
                ),
                components.pill(
                    f"{self.stats.get('movielens_ratings', 0):,} valoraciones",
                    theme.YELLOW_TEXT,
                    theme.YELLOW_SOFT,
                ),
            ],
            spacing=6,
        )
        return ft.Container(
            height=78,
            padding=ft.Padding.symmetric(horizontal=24, vertical=12),
            bgcolor=theme.SURFACE,
            border=ft.Border.only(bottom=ft.BorderSide(width=1, color=theme.BORDER)),
            content=ft.Row(
                controls=[
                    ft.Column(
                        controls=[
                            ft.Row(
                                controls=[
                                    ft.Text(
                                        "CineBot",
                                        size=20,
                                        weight=ft.FontWeight.W_600,
                                        color=theme.TEXT,
                                    ),
                                    ft.Container(
                                        content=ft.Row(
                                            controls=[
                                                ft.Container(
                                                    width=6,
                                                    height=6,
                                                    bgcolor=theme.GREEN,
                                                    border_radius=ft.BorderRadius.all(3),
                                                ),
                                                ft.Text("En línea", size=9, color=theme.ACCENT_TEXT),
                                            ],
                                            spacing=5,
                                        ),
                                        padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                                        bgcolor=theme.ACCENT_SOFT,
                                        border_radius=ft.BorderRadius.all(10),
                                    ),
                                ],
                                spacing=8,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                            ft.Text(
                                "Recomendaciones explicables · modelo híbrido local",
                                size=10,
                                color=theme.TEXT_MUTED,
                            ),
                        ],
                        spacing=1,
                    ),
                    ft.Container(expand=True),
                    self.header_stats,
                    ft.IconButton(
                        icon=(
                            ft.Icons.LIGHT_MODE_OUTLINED
                            if theme.is_dark()
                            else ft.Icons.DARK_MODE_OUTLINED
                        ),
                        tooltip=(
                            "Cambiar a tema claro"
                            if theme.is_dark()
                            else "Cambiar a tema oscuro"
                        ),
                        icon_color=theme.TEXT_MUTED,
                        on_click=self._on_toggle_theme,
                    ),
                    ft.IconButton(
                        icon=ft.Icons.AUTO_AWESOME,
                        tooltip="Ayuda",
                        icon_color=theme.TEXT_MUTED,
                        on_click=lambda e: self._send_text("ayuda"),
                    ),
                    ft.IconButton(
                        icon=ft.Icons.ADD_COMMENT_OUTLINED,
                        tooltip="Nueva conversación",
                        icon_color=theme.TEXT_MUTED,
                        on_click=self._on_new_conversation,
                    ),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    def _build_sidebar(self) -> ft.Container:
        brand = ft.Column(
            controls=[
                ft.Text("CineBot", size=19, weight=ft.FontWeight.W_600, color=theme.TEXT),
                ft.Text(config.APP_TAGLINE, size=9, color=theme.TEXT_MUTED),
            ],
            spacing=0,
        )

        self.profile_dropdown = self._build_profile_dropdown()
        quick_chips = ft.Row(
            controls=[
                components.action_chip(
                    label,
                    lambda prompt=prompt: self._send_text(prompt),
                    icon=icon,
                )
                for label, prompt, icon in QUICK_PROMPTS
            ],
            wrap=True,
            spacing=6,
            run_spacing=6,
        )

        return ft.Container(
            width=310,
            bgcolor=theme.SURFACE,
            padding=ft.Padding.symmetric(horizontal=18, vertical=20),
            content=ft.Column(
                controls=[
                    brand,
                    ft.Divider(height=1, color=theme.BORDER),
                    components.section_title("Perfil"),
                    self.profile_dropdown,
                    self.profile_holder,
                    components.section_title("Atajos"),
                    quick_chips,
                    self.model_holder,
                    ft.Divider(height=1, color=theme.BORDER),
                    ft.Row(
                        controls=[
                            ft.Text(
                                "TF-IDF · FunkSVD · K-Means · Regresión logística",
                                size=8,
                                color=theme.TEXT_DIM,
                                expand=True,
                            ),
                            ft.IconButton(
                                icon=ft.Icons.DELETE_SWEEP_OUTLINED,
                                icon_size=16,
                                tooltip="Borrar mis valoraciones",
                                icon_color=theme.TEXT_DIM,
                                on_click=lambda e: self._send_text("borra mis valoraciones"),
                            ),
                        ],
                        spacing=4,
                    ),
                    ft.Text(
                        "Datos: MovieLens + TVmaze · Portadas con caché local",
                        size=8,
                        color=theme.TEXT_DIM,
                    ),
                ],
                spacing=11,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            ),
        )

    def _build_profile_dropdown(self) -> ft.Dropdown:
        options: list[ft.DropdownOption] = [
            ft.DropdownOption(
                key=str(self.guest_id),
                text=f"Invitado (tu perfil) · {self.user_repo.profile(self.guest_id).n_ratings} notas",
            )
        ]
        for profile in self.demo_users:
            options.append(
                ft.DropdownOption(
                    key=str(profile.user_id),
                    text=f"{profile.username} · {profile.n_ratings:,} notas",
                )
            )
        return ft.Dropdown(
            value=str(self.user_id),
            options=options,
            on_select=self._on_profile_change,
            dense=True,
            text_size=11,
            bgcolor=theme.SURFACE_ALT,
            fill_color=theme.SURFACE_ALT,
            border_color=theme.BORDER,
            focused_border_color=theme.PRIMARY,
            border_radius=ft.BorderRadius.all(12),
            content_padding=ft.Padding.symmetric(horizontal=12, vertical=10),
            label="Perfil de demostración",
            label_style=ft.TextStyle(size=10, color=theme.TEXT_MUTED),
            color=theme.TEXT,
        )

    def _build_composer(self) -> ft.Container:
        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=24, vertical=14),
            bgcolor=theme.SURFACE,
            border=ft.Border.only(top=ft.BorderSide(width=1, color=theme.BORDER)),
            content=ft.Column(
                controls=[
                    self.suggestions_row,
                    ft.Row(
                        controls=[self.input, self.send_button],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Text(
                        "CineBot puede equivocarse: las notas y sinopsis provienen de MovieLens y TVmaze.",
                        size=8,
                        color=theme.TEXT_DIM,
                    ),
                ],
                spacing=8,
            ),
        )

    # ------------------------------------------------------------------
    # Acciones
    # ------------------------------------------------------------------
    def _on_submit(self, e: Any = None) -> None:
        if self.busy:
            return
        text = (self.input.value or "").strip()
        if not text:
            return
        self.input.value = ""
        self._send_text(text)

    def _send_text(self, text: str) -> None:
        if self.busy or not text.strip():
            return
        self.busy = True
        self.send_button.disabled = True
        self.history.append({"role": "user", "text": text})
        self._append(components.chat_bubble(text, is_user=True, max_width=self._bubble_width()))
        self.suggestions_row.controls = []
        self._suggestions = []
        self.typing_row = components.typing_indicator()
        self.chat_list.controls.append(self.typing_row)
        self.page.update()
        self.page.run_thread(self._process_message, text)

    def _process_message(self, text: str) -> None:
        try:
            response = self.engine.handle(text)
        except Exception as exc:  # noqa: BLE001
            response = ChatResponse(
                text=(
                    "Ocurrió un error al procesar el mensaje. "
                    f"Detalle: `{type(exc).__name__}: {exc}`. "
                    "Puedes reintentar o iniciar una conversación nueva."
                ),
                intent="error",
            )
        self._finish_message(response)

    def _attach_posters(self, response: ChatResponse) -> None:
        items = [rec.item for rec in response.recommendations]
        if not items:
            return
        try:
            self.poster_resolver.enrich(items)
        except Exception:  # noqa: BLE001
            pass

    def _finish_message(self, response: ChatResponse) -> None:
        try:
            self._attach_posters(response)
            if self.typing_row is not None and self.typing_row in self.chat_list.controls:
                self.chat_list.controls.remove(self.typing_row)
            self.typing_row = None
            self._show_response(response)
            self._refresh_sidebar()
        finally:
            self.busy = False
            self.send_button.disabled = False
            try:
                self.input.focus()
            except Exception:
                pass
            self.page.update()

    def _show_response(self, response: ChatResponse, remember: bool = True) -> None:
        if remember:
            self.history.append({"role": "assistant", "response": response})
        bubble = components.chat_bubble(
            response.text,
            is_user=False,
            recommendations=response.recommendations,
            on_rate=self._on_rate,
            on_feedback=self._on_feedback,
            on_details=self._on_details,
            max_width=self._bubble_width(),
        )
        self._append(bubble)
        self._set_suggestions(response.followups)
        self.page.update()

    def _append(self, control: ft.Control) -> None:
        self.chat_list.controls.append(control)
        self.suggestions_row.controls = []
        if self.chat_list.page:
            self.chat_list.update()

    def _set_suggestions(self, followups: list[str]) -> None:
        self._suggestions = followups[:5]
        self.suggestions_row.controls = [
            components.action_chip(
                text,
                lambda prompt=text: self._send_text(prompt),
                muted=True,
            )
            for text in followups[:5]
        ]

    def _on_rate(self, item_id: int, rating: int) -> None:
        if self.busy:
            return
        self.busy = True
        self.send_button.disabled = True
        self.page.run_thread(self._rate_thread, item_id, rating)

    def _rate_thread(self, item_id: int, rating: int) -> None:
        try:
            response = self.engine.rate(item_id, float(rating))
        except Exception as exc:  # noqa: BLE001
            response = ChatResponse(text=f"No pude guardar la nota: `{exc}`")
        self._finish_message(response)

    def _on_feedback(self, item_id: int, action: str) -> None:
        try:
            message = self.engine.feedback(item_id, action)
        except Exception as exc:  # noqa: BLE001
            message = f"No pude registrar el feedback: {exc}"
        self._snack(message)

    def _on_details(self, item_id: int) -> None:
        item = self.catalog_repo.get_item(item_id)
        if item:
            self._send_text(f"de qué trata {item.title}")

    def _on_profile_change(self, e: Any = None) -> None:
        if self.busy:
            return
        # Flet 1.x entrega el valor seleccionado en `e.data`; se usa como
        # fuente principal y se conserva el valor del control como respaldo.
        raw_data = getattr(e, "data", None)
        value = raw_data if raw_data not in (None, "") else self._event_value(e, self.profile_dropdown)
        try:
            user_id = int(value)
        except (TypeError, ValueError):
            return
        if user_id == self.user_id:
            return
        self.user_id = user_id
        self.engine = self._make_engine(user_id)
        self.history.clear()
        self._suggestions = []
        self.chat_list.controls.clear()
        self.suggestions_row.controls = []
        self._refresh_sidebar()
        response = self.engine.start()
        self._show_response(response)
        self.page.update()

    def _on_new_conversation(self, e: Any = None) -> None:
        if self.busy:
            return
        self.history.clear()
        self._suggestions = []
        self.chat_list.controls.clear()
        self.suggestions_row.controls = []
        response = self.engine.new_conversation()
        self._show_response(response)
        self.page.update()

    def _on_toggle_theme(self, e: Any = None) -> None:
        if self.busy:
            return
        previous = theme.current_mode()
        theme.toggle_mode()
        self._apply_page_theme()
        try:
            self._rebuild_ui()
        except Exception:
            theme.set_mode(previous)
            self._apply_page_theme()
            self._snack("No se pudo cambiar el tema.")
            self.page.update()

    def _apply_page_theme(self) -> None:
        self.page.theme_mode = ft.ThemeMode.DARK if theme.is_dark() else ft.ThemeMode.LIGHT
        self.page.bgcolor = theme.BG
        page_theme = ft.Theme(font_family="Segoe UI", color_scheme_seed=theme.PRIMARY)
        self.page.theme = page_theme
        self.page.dark_theme = page_theme

    def _rebuild_ui(self) -> None:
        input_value = self.input.value or ""
        self.root = self._build()
        self.input.value = input_value
        self._render_history()
        self._set_suggestions(self._suggestions)
        self._apply_responsive()
        self.page.clean()
        self.page.add(self.root)
        self.page.update()

    def _render_history(self) -> None:
        max_width = self._bubble_width()
        for entry in self.history:
            if entry["role"] == "user":
                self.chat_list.controls.append(
                    components.chat_bubble(
                        entry["text"],
                        is_user=True,
                        max_width=max_width,
                    )
                )
                continue
            response = entry["response"]
            self.chat_list.controls.append(
                components.chat_bubble(
                    response.text,
                    is_user=False,
                    recommendations=response.recommendations,
                    on_rate=self._on_rate,
                    on_feedback=self._on_feedback,
                    on_details=self._on_details,
                    max_width=max_width,
                )
            )

    def _on_resize(self, e: Any = None) -> None:
        self._apply_responsive()
        try:
            self.page.update()
        except Exception:
            pass

    def _apply_responsive(self) -> None:
        width = self.page.width or 1280
        self.sidebar.visible = width >= 1040
        self.header_stats.visible = width >= 960

    def _bubble_width(self) -> int:
        width = self.page.width or 1280
        sidebar = 310 if width >= 1040 else 0
        return max(320, min(880, width - sidebar - 72))

    def _event_value(self, e: Any, control: ft.Control) -> Any:
        control_value = getattr(getattr(e, "control", None), "value", None)
        return control_value if control_value is not None else getattr(control, "value", None)

    def _snack(self, message: str) -> None:
        try:
            self.page.show_dialog(
                ft.SnackBar(
                    content=ft.Text(message, size=12, color=theme.TOAST_TEXT),
                    bgcolor=theme.TOAST_BG,
                )
            )
        except Exception:
            pass

    def _refresh_sidebar(self) -> None:
        try:
            profile = self.engine.profile
            self.profile_holder.content = components.profile_panel(profile, self.stats)
            self.model_holder.content = components.model_panel(self.metadata, self.stats)
            self.profile_dropdown.value = str(self.user_id)
        except Exception:
            pass
