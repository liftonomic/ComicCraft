"""Export service.

Builds a professional comic PDF using ReportLab. Each panel gets a clean,
comic-like page layout: panel number + title, generated image, narration and
dialogue. The PDF is written to ``generated/pdfs/``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from app.config import Settings
from app.database.database import ComicStore
from app.models.comic import ComicRecord
from app.utils.errors import ExportError
from app.utils.files import pdfs_dir

logger = logging.getLogger(__name__)

A4_W, A4_H = A4


class ExportService:
    """Turn a comic record into a downloadable PDF file."""

    def __init__(self, *, settings: Settings, store: ComicStore) -> None:
        self._settings = settings
        self._store = store

    async def build_pdf(self, comic: ComicRecord) -> str:
        """Generate the PDF and return its absolute file path."""
        directory = pdfs_dir()
        filename = f"comic-{comic.id}-{int(time.time())}.pdf"
        path: Path = directory / filename
        try:
            self._draw(path, comic)
        except Exception as exc:  # noqa: BLE001
            logger.exception("PDF export failed for comic %s", comic.id)
            raise ExportError(str(exc)) from exc
        return str(path)

    def _draw(self, path: Path, comic: ComicRecord) -> None:
        c = canvas.Canvas(str(path), pagesize=A4)
        c.setTitle(f"{comic.title or 'Comic'} — ComicCraft")
        margin = 18 * mm
        page_w = A4_W - 2 * margin

        self._page_background(c)
        c.setFillColor(colors.HexColor("#ffd93d"))
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, A4_H - 22, "COMICCRAFT  /  STORY EDITION")
        c.setFillColor(colors.HexColor("#f8fafc"))
        c.setFont("Helvetica-Bold", 22)
        c.drawString(margin, A4_H - 42, _truncate(comic.title or "Untitled Comic", 60))
        c.setFillColor(colors.HexColor("#a9b8cf"))
        c.setFont("Helvetica", 9)
        c.drawString(
            margin,
            A4_H - 54,
            f"{comic.art_style or 'mixed'}  |  {len(comic.panels)} panels",
        )

        y = A4_H - 72
        for panel in comic.panels:
            need = self._panel_height(panel)
            if y - need < margin:
                self._new_page(c)
                y = A4_H - margin
            y = self._draw_panel(c, panel, margin, page_w, y)
        c.save()

    def _page_background(self, c: canvas.Canvas) -> None:
        c.setFillColor(colors.HexColor("#0f172a"))
        c.rect(0, 0, A4_W, A4_H, stroke=0, fill=1)

    def _new_page(self, c: canvas.Canvas) -> None:
        c.showPage()
        self._page_background(c)

    def _panel_height(self, panel) -> int:
        text = len(panel.scene or "") + len(panel.narration or "") + len(
            " ".join(d.text for d in panel.dialogue)
        )
        return 40 * mm + int(text * 0.25) * mm

    def _draw_panel(self, c: canvas.Canvas, panel, margin, page_w, y) -> int:
        c.setFillColor(colors.HexColor("#ffd93d"))
        c.setFont("Helvetica-Bold", 11)
        label = f"PANEL {panel.panel_number}  /  {panel.title or 'Scene'}"
        c.drawString(margin, y, _truncate(label, 60))
        y -= 7 * mm

        # --- Image (if available) ---
        image_path = Path(panel.image_path) if panel.image_path else None
        if image_path and image_path.exists():
            try:
                c.drawImage(
                    str(image_path),
                    margin,
                    max(y - 60 * mm, 20),
                    width=page_w,
                    height=60 * mm,
                    preserveAspectRatio=True,
                    anchor="s",
                )
                y -= 64 * mm
            except Exception:  # noqa: BLE001 - missing image not fatal
                y -= 8 * mm
        else:
            c.setFillColor(colors.HexColor("#334155"))
            c.roundRect(margin, y - 40 * mm, page_w, 40 * mm, 4 * mm, stroke=0, fill=1)
            c.setFillColor(colors.HexColor("#a9b8cf"))
            c.setFont("Helvetica-Oblique", 10)
            c.drawString(margin + 8 * mm, y - 18 * mm, "Image unavailable")
            y -= 46 * mm

        if panel.scene:
            c.setFillColor(colors.HexColor("#f8fafc"))
            c.setFont("Helvetica", 10)
            y = self._wrap(c, panel.scene, margin, page_w, y)

        dialogue_lines = [d.text for d in panel.dialogue if d.text]
        if dialogue_lines:
            c.setFillColor(colors.HexColor("#ffafaf"))
            c.setFont("Helvetica-Bold", 10)
            y = self._wrap(c, '"' + " ".join(dialogue_lines) + '"', margin, page_w, y)

        if panel.narration:
            c.setFillColor(colors.HexColor("#a9b8cf"))
            c.setFont("Helvetica-Oblique", 10)
            y = self._wrap(c, panel.narration, margin, page_w, y)

        return y - 4 * mm

    def _wrap(self, c: canvas.Canvas, text: str, x: float, w: float, y: float) -> float:
        """Draw wrapped text and return the new baseline y position."""
        for line in _split_lines(text, w, c):
            if y < 22 * mm:
                self._new_page(c)
                y = A4_H - 22 * mm
            c.drawString(x, y, line)
            y -= 5 * mm
        return y


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _split_lines(text: str, width: float, c: canvas.Canvas) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split(" "):
            probe = f"{current} {word}".strip()
            if c.stringWidth(probe, "Helvetica", 10) <= width:
                current = probe
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines