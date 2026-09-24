"""Build the private, method-only PDF documentation for notebook 03."""

from __future__ import annotations

import html
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    LongTable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "docs" / "notebook_03_eda_cohorte_3d.md"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "documentacion_notebook_03_eda_cohorte_3d.pdf"
TEMP_DIR = PROJECT_ROOT / "tmp" / "pdfs" / "notebook_03"

INK = colors.HexColor("#17324D")
BLUE = colors.HexColor("#246B9E")
TEAL = colors.HexColor("#268C8C")
MAGENTA = colors.HexColor("#C33D72")
PALE_BLUE = colors.HexColor("#EAF3F8")
PALE_GRAY = colors.HexColor("#F4F6F7")
MID_GRAY = colors.HexColor("#607080")
LIGHT_RULE = colors.HexColor("#CBD7DF")


def register_fonts() -> tuple[str, str, str]:
    """Register Windows fonts with a safe built-in fallback."""
    regular = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    mono = Path("C:/Windows/Fonts/consola.ttf")
    if regular.exists() and bold.exists() and mono.exists():
        pdfmetrics.registerFont(TTFont("DocSans", regular))
        pdfmetrics.registerFont(TTFont("DocSans-Bold", bold))
        pdfmetrics.registerFont(TTFont("DocMono", mono))
        return "DocSans", "DocSans-Bold", "DocMono"
    return "Helvetica", "Helvetica-Bold", "Courier"


FONT, FONT_BOLD, FONT_MONO = register_fonts()


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.4,
            leading=13.2,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=5.5,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.7,
            leading=10.2,
            textColor=MID_GRAY,
        ),
        "body_lead": ParagraphStyle(
            "BodyLead",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.4,
            leading=13.2,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=5.5,
            keepWithNext=True,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=17,
            leading=20,
            textColor=INK,
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=13,
            leading=16,
            textColor=BLUE,
            spaceBefore=9,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName=FONT_BOLD,
            fontSize=10.5,
            leading=13,
            textColor=TEAL,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName=FONT_MONO,
            fontSize=7.7,
            leading=10.1,
            leftIndent=7,
            rightIndent=7,
            textColor=colors.HexColor("#21303C"),
            backColor=PALE_GRAY,
            borderColor=LIGHT_RULE,
            borderWidth=0.5,
            borderPadding=7,
            spaceBefore=4,
            spaceAfter=7,
        ),
        "table": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.7,
            leading=10,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName=FONT_BOLD,
            fontSize=7.8,
            leading=10,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName=FONT_BOLD,
            fontSize=27,
            leading=32,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=13,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=13,
            leading=18,
            textColor=BLUE,
            alignment=TA_LEFT,
        ),
        "toc": ParagraphStyle(
            "Contents",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=10,
            leading=15,
            textColor=INK,
            leftIndent=7,
        ),
    }


STYLES = build_styles()


def inline_markup(value: str) -> str:
    """Convert the small inline Markdown subset used by the source document."""
    escaped = html.escape(value.strip())
    escaped = re.sub(r"`([^`]+)`", rf'<font name="{FONT_MONO}">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def equation_flowable(latex: str, index: int):
    """Render compatible LaTeX with MathText and retain source on fallback."""
    expression = " ".join(line.strip() for line in latex.splitlines()).strip()
    if "\\begin{" in expression:
        return Preformatted(expression, STYLES["code"], maxLineLength=88)
    image_path = TEMP_DIR / f"equation_{index:02d}.png"
    try:
        figure = plt.figure(figsize=(8.2, 0.62), dpi=180)
        figure.patch.set_alpha(0)
        figure.text(
            0.5,
            0.5,
            f"${expression}$",
            ha="center",
            va="center",
            fontsize=15,
            color="#17324D",
        )
        figure.savefig(image_path, transparent=True, bbox_inches="tight", pad_inches=0.06)
        plt.close(figure)
        rendered = Image(str(image_path))
        max_width = 165 * mm
        scale = min(1.0, max_width / rendered.imageWidth)
        rendered.drawWidth = rendered.imageWidth * scale
        rendered.drawHeight = rendered.imageHeight * scale
        rendered.hAlign = "CENTER"
        return KeepTogether([Spacer(1, 2 * mm), rendered, Spacer(1, 2 * mm)])
    except (RuntimeError, ValueError):
        plt.close("all")
        return Preformatted(expression, STYLES["code"], maxLineLength=88)


def table_flowable(rows: list[list[str]], available_width: float) -> LongTable:
    """Create a compact, repeating-header table with adaptive column widths."""
    column_count = max(len(row) for row in rows)
    normalized = [row + [""] * (column_count - len(row)) for row in rows]
    weights = []
    for column in range(column_count):
        longest = max(len(row[column]) for row in normalized)
        weights.append(min(max(longest, 12), 42))
    total = sum(weights)
    widths = [available_width * weight / total for weight in weights]
    cells = []
    for row_index, row in enumerate(normalized):
        style = STYLES["table_header"] if row_index == 0 else STYLES["table"]
        cells.append([Paragraph(inline_markup(cell), style) for cell in row])
    table = LongTable(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("BOX", (0, 0), (-1, -1), 0.5, LIGHT_RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, LIGHT_RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_index in range(1, len(cells)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE_BLUE))
    table.setStyle(TableStyle(commands))
    return table


def parse_markdown(source: str, available_width: float) -> list:
    """Convert the controlled Markdown source into ReportLab flowables."""
    lines = source.splitlines()
    story: list = []
    paragraph_lines: list[str] = []
    equation_index = 0
    index = 0

    def flush_paragraph() -> None:
        if paragraph_lines:
            paragraph = " ".join(paragraph_lines)
            style = STYLES["body_lead"] if paragraph.rstrip().endswith(":") else STYLES["body"]
            story.append(Paragraph(inline_markup(paragraph), style))
            paragraph_lines.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if index == 0 and stripped.startswith("# "):
            index += 1
            continue
        if stripped.startswith("```"):
            flush_paragraph()
            language = stripped[3:].strip()
            index += 1
            code_lines = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            caption = "Extracto de código" if language == "python" else "Esquema"
            story.append(Paragraph(f"<b>{caption}</b>", STYLES["small"]))
            story.append(
                Preformatted("\n".join(code_lines), STYLES["code"], maxLineLength=88)
            )
            index += 1
            continue
        if stripped == "$$":
            flush_paragraph()
            index += 1
            equation_lines = []
            while index < len(lines) and lines[index].strip() != "$$":
                equation_lines.append(lines[index])
                index += 1
            equation_index += 1
            story.append(equation_flowable("\n".join(equation_lines), equation_index))
            index += 1
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            table_lines = []
            while index < len(lines):
                candidate = lines[index].strip()
                if not (candidate.startswith("|") and candidate.endswith("|")):
                    break
                table_lines.append(candidate)
                index += 1
            rows = [
                [cell.strip() for cell in row.strip("|").split("|")]
                for row in table_lines
            ]
            rows = [
                row
                for row in rows
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in row)
            ]
            story.extend([table_flowable(rows, available_width), Spacer(1, 3 * mm)])
            continue
        heading = re.match(r"^(#{2,4})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            story.append(Paragraph(inline_markup(heading.group(2)), STYLES[f"h{level - 1}"]))
            index += 1
            continue
        if re.match(r"^-\s+", stripped):
            flush_paragraph()
            items = []
            while index < len(lines) and re.match(r"^-\s+", lines[index].strip()):
                item = re.sub(r"^-\s+", "", lines[index].strip())
                items.append(ListItem(Paragraph(inline_markup(item), STYLES["body"])))
                index += 1
            story.append(
                KeepTogether(
                    [
                        ListFlowable(
                            items,
                            bulletType="bullet",
                            bulletColor=TEAL,
                            leftIndent=15,
                        ),
                        Spacer(1, 2 * mm),
                    ]
                )
            )
            continue
        if re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            items = []
            while index < len(lines) and re.match(r"^\d+\.\s+", lines[index].strip()):
                item = re.sub(r"^\d+\.\s+", "", lines[index].strip())
                items.append(ListItem(Paragraph(inline_markup(item), STYLES["body"])))
                index += 1
            story.append(
                KeepTogether(
                    [
                        ListFlowable(items, bulletType="1", leftIndent=18),
                        Spacer(1, 2 * mm),
                    ]
                )
            )
            continue
        if not stripped:
            flush_paragraph()
        else:
            paragraph_lines.append(stripped)
        index += 1
    flush_paragraph()
    return story


def page_decoration(canvas, document) -> None:
    """Draw page furniture and PDF metadata."""
    canvas.saveState()
    canvas.setTitle("Notebook 03 - EDA de cohorte completa y control de calidad físico")
    canvas.setAuthor("Proyecto DaT Parkinson's Challenge")
    if document.page > 1:
        canvas.setStrokeColor(LIGHT_RULE)
        canvas.line(22 * mm, 282 * mm, 188 * mm, 282 * mm)
        canvas.setFont(FONT, 7.3)
        canvas.setFillColor(MID_GRAY)
        canvas.drawString(22 * mm, 286 * mm, "NOTEBOOK 03 · DOCUMENTACIÓN METODOLÓGICA")
    canvas.setStrokeColor(LIGHT_RULE)
    canvas.line(22 * mm, 15 * mm, 188 * mm, 15 * mm)
    canvas.setFont(FONT, 7.3)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(22 * mm, 10 * mm, "DaT Parkinson's Challenge · Documento metodológico")
    canvas.drawRightString(188 * mm, 10 * mm, f"Página {document.page}")
    canvas.restoreState()


def cover_story() -> list:
    """Create a restrained cover and a compact contents page."""
    badge = Table(
        [[Paragraph("DOCUMENTACIÓN EXCLUSIVA DEL NOTEBOOK 03", STYLES["small"])]],
        colWidths=[80 * mm],
    )
    badge.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.6, BLUE),
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
                "<b>Alcance</b><br/>Auditoría física y técnica de los 1.362 estudios. "
                "No registra cerebros, no diagnostica y no excluye casos automáticamente.",
                STYLES["body"],
            )
        ]],
        colWidths=[163 * mm],
    )
    scope.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_GRAY),
                ("BOX", (0, 0), (-1, -1), 0.7, TEAL),
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
        "4. Interpretación correcta",
        "5. Supuestos y limitaciones",
        "6. Glosario breve",
    ]
    return [
        Spacer(1, 23 * mm),
        badge,
        Spacer(1, 15 * mm),
        Paragraph("Notebook 03", STYLES["cover_title"]),
        Paragraph(
            "EDA de cohorte completa y<br/>control de calidad físico",
            STYLES["cover_title"],
        ),
        Spacer(1, 5 * mm),
        Paragraph(
            "Definiciones, supuestos, fórmulas y justificación metodológica",
            STYLES["cover_subtitle"],
        ),
        Spacer(1, 18 * mm),
        scope,
        Spacer(1, 18 * mm),
        Paragraph("Contenido", STYLES["h2"]),
        *[Paragraph(item, STYLES["toc"]) for item in contents],
        Spacer(1, 15 * mm),
        Paragraph(
            "Versión metodológica · 22 de agosto de 2026",
            STYLES["small"],
        ),
        PageBreak(),
    ]


def main() -> None:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(SOURCE_PATH)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=22 * mm,
        leftMargin=22 * mm,
        topMargin=23 * mm,
        bottomMargin=20 * mm,
        title="Notebook 03 - EDA de cohorte completa y control de calidad físico",
        author="Proyecto DaT Parkinson's Challenge",
        subject="Documentación metodológica exclusiva del notebook 03",
    )
    available_width = A4[0] - document.leftMargin - document.rightMargin
    source = SOURCE_PATH.read_text(encoding="utf-8")
    story = cover_story() + parse_markdown(source, available_width)
    document.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    try:
        main()
    finally:
        plt.close("all")
