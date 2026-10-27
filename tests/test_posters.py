"""Pruebas de la resolución y presentación de portadas."""

from __future__ import annotations

import unittest

import flet as ft

from cinebot.domain import MediaItem
from cinebot.posters import parse_letterboxd_poster, parse_letterboxd_year, slugify, title_variants
from cinebot.ui import components, theme


class PosterHelperTests(unittest.TestCase):
    def test_slugify_handles_accents_and_ampersands(self) -> None:
        self.assertEqual(slugify("Cronocrímenes, Los"), "cronocrimenes-los")
        self.assertEqual(slugify("Fast & Furious"), "fast-and-furious")

    def test_title_variants_move_articles_and_drop_foreign_title(self) -> None:
        self.assertIn("The Shining", title_variants("Shining, The"))
        self.assertIn("Departures", title_variants("Departures (Okuribito)"))
        self.assertIn("A Prophet", title_variants("Prophet, A (Un Prophète)"))

    def test_letterboxd_parser(self) -> None:
        html = (
            '<div data-item-name="Waitress (2007)"></div>'
            '<img src="https://a.ltrbxd.com/resized/film-poster/4/5/9/2/4/'
            '45924-waitress-0-600-0-900-crop.jpg?v=8a807fa038">'
        )
        self.assertEqual(
            parse_letterboxd_poster(html),
            "https://a.ltrbxd.com/resized/film-poster/4/5/9/2/4/"
            "45924-waitress-0-600-0-900-crop.jpg?v=8a807fa038",
        )
        self.assertEqual(parse_letterboxd_year(html), 2007)

    def test_poster_uses_image_when_url_exists(self) -> None:
        item = MediaItem(
            item_id=1,
            media_type="movie",
            title="Toy Story",
            year=1995,
            genres=["Animation"],
            poster_url="https://example.com/toy-story.jpg",
        )
        control = components.poster(item)
        self.assertIsNotNone(control.content)
        self.assertIsInstance(control.content, ft.Image)

    def test_poster_falls_back_without_url(self) -> None:
        item = MediaItem(
            item_id=2,
            media_type="movie",
            title="Jumanji",
            year=1995,
            genres=["Adventure"],
        )
        control = components.poster(item)
        self.assertIsInstance(control.content, ft.Column)

    def test_theme_toggle_changes_palette_and_markdown(self) -> None:
        previous = theme.current_mode()
        self.addCleanup(theme.set_mode, previous)
        theme.set_mode("light")
        light_background = theme.BG
        theme.set_mode("dark")
        self.assertTrue(theme.is_dark())
        self.assertNotEqual(theme.BG, light_background)
        self.assertEqual(components.markdown_style().p_text_style.color, theme.TEXT)
        metric = components.metric_row("RMSE", "0.850")
        self.assertEqual(metric.controls[1].color, theme.TEXT)
        theme.set_mode("light")
        self.assertFalse(theme.is_dark())


if __name__ == "__main__":
    unittest.main()
