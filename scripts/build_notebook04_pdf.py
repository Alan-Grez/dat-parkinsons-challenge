"""Build the method-only PDF documentation for notebook 04."""

from __future__ import annotations

from pathlib import Path

import build_notebook03_pdf as base
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "docs" / "notebook_04_registro_biomarcadores_radiomica.md"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "documentacion_notebook_04_registro_biomarcadores_radiomica.pdf"
base.TEMP_DIR = PROJECT_ROOT / "tmp" / "pdfs" / "notebook_04"


def page_decoration(canvas, document) -> None:
    canvas.saveState()
    canvas.setTitle("Notebook 04 v3 - Registro, biomarcadores candidatos y radiomics 3D")
    canvas.setAuthor("Proyecto DaT Parkinson's Challenge")
    if document.page > 1:
        canvas.setStrokeColor(base.LIGHT_RULE)
        canvas.line(22 * mm, 282 * mm, 188 * mm, 282 * mm)
        canvas.setFont(base.FONT, 7.3)
        canvas.setFillColor(base.MID_GRAY)
        canvas.drawString(22 * mm, 286 * mm, "NOTEBOOK 04 · DOCUMENTACIÓN METODOLÓGICA")
    canvas.setStrokeColor(base.LIGHT_RULE)
    canvas.line(22 * mm, 15 * mm, 188 * mm, 15 * mm)
    canvas.setFont(base.FONT, 7.3)
    canvas.setFillColor(base.MID_GRAY)
    canvas.drawString(22 * mm, 10 * mm, "DaT Parkinson's Challenge · Documento metodológico")
    canvas.drawRightString(188 * mm, 10 * mm, f"Página {document.page}")
    canvas.restoreState()


def cover_story() -> list:
    badge = Table(
        [[Paragraph("DOCUMENTACIÓN EXCLUSIVA DEL NOTEBOOK 04", base.STYLES["small"])]],
        colWidths=[80 * mm],
    )
    badge.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), base.PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.6, base.BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    scope = Table(
        [[
            Paragraph(
                "<b>Alcance</b><br/>Registro físico, fondo robusto con QC, cuantificación "
                "relativa, forma, textura 3D, estabilidad y mapas descriptivos para la cohorte.",
                base.STYLES["body"],
            )
        ]],
        colWidths=[163 * mm],
    )
    scope.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), base.PALE_GRAY),
                ("BOX", (0, 0), (-1, -1), 0.7, base.TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    contents = [
        "1. Propósito y alcance",
        "2. Terminología fundamental",
        "3. Recorrido celda por celda",
        "4. Interpretación de features",
        "5. Supuestos y limitaciones",
        "6. Glosario breve",
    ]
    return [
        Spacer(1, 23 * mm),
        badge,
        Spacer(1, 15 * mm),
        Paragraph("Notebook 04", base.STYLES["cover_title"]),
        Paragraph(
            "Registro, biomarcadores candidatos<br/>y radiomics 3D",
            base.STYLES["cover_title"],
        ),
        Spacer(1, 5 * mm),
        Paragraph(
            "Definiciones, supuestos, fórmulas y justificación metodológica",
            base.STYLES["cover_subtitle"],
        ),
        Spacer(1, 18 * mm),
        scope,
        Spacer(1, 18 * mm),
        Paragraph("Contenido", base.STYLES["h2"]),
        *[Paragraph(item, base.STYLES["toc"]) for item in contents],
        Spacer(1, 15 * mm),
        Paragraph("Versión metodológica v3 · 26 de agosto de 2026", base.STYLES["small"]),
        PageBreak(),
    ]


def main() -> None:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(SOURCE_PATH)
    base.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=22 * mm,
        leftMargin=22 * mm,
        topMargin=23 * mm,
        bottomMargin=20 * mm,
        title="Notebook 04 v3 - Registro, biomarcadores candidatos y radiomics 3D",
        author="Proyecto DaT Parkinson's Challenge",
        subject="Documentación metodológica exclusiva del notebook 04",
    )
    available_width = A4[0] - document.leftMargin - document.rightMargin
    source = SOURCE_PATH.read_text(encoding="utf-8")
    story = cover_story() + base.parse_markdown(source, available_width)
    document.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
