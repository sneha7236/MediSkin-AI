"""PDF report generation (ReportLab). Reports are written to reports/generated/."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import config, constants
from database import crud
from modules.disease_info import get_disease_info
from prediction import confidence as confidence_rules
from utils import security
from utils.logger import get_logger

logger = get_logger("reports")

NAVY = colors.HexColor("#0F3D5E")
TEAL = colors.HexColor("#0E7C86")
LIGHT = colors.HexColor("#EEF5F7")


class ReportError(RuntimeError):
    """Report could not be generated (message is safe to show to the user)."""


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], textColor=NAVY, fontSize=22, spaceAfter=2, alignment=0),
        "subtitle": ParagraphStyle("s", parent=base["Normal"], textColor=TEAL, fontSize=10, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], textColor=NAVY, fontSize=13, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("b", parent=base["BodyText"], fontSize=10, leading=14),
        "small": ParagraphStyle("sm", parent=base["BodyText"], fontSize=8.5, leading=11, textColor=colors.HexColor("#334155")),
        "disclaimer": ParagraphStyle("d", parent=base["BodyText"], fontSize=9, leading=12,
                                     textColor=colors.HexColor("#7F1D1D"), backColor=colors.HexColor("#FEF2F2"),
                                     borderPadding=6, spaceBefore=8),
    }


def _p(text: str, style) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def _bullets(items, style) -> ListFlowable:
    return ListFlowable([ListItem(_p(i, style), leftIndent=12) for i in items], bulletType="bullet", start="•")


def _image_flowable(image: Image.Image, max_mm: float = 60) -> RLImage:
    thumb = image.convert("RGB")
    thumb.thumbnail((600, 600))
    buffer = io.BytesIO()
    thumb.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    width, height = thumb.size
    scale = (max_mm * mm) / max(width, height)
    return RLImage(buffer, width=width * scale, height=height * scale)


def _page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(20 * mm, 10 * mm, f"{constants.APP_NAME} — informational report, not a medical diagnosis")
    canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_report_pdf(path: Path, user_name: str, condition_id: str, condition_name: str,
                     confidence: float, risk_level: str, probabilities: dict[str, float],
                     created_at: datetime, image: Image.Image | None = None) -> None:
    styles = _styles()
    info = get_disease_info(condition_id)
    band = confidence_rules.confidence_band(confidence)
    top = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[: constants.TOP_K_PREDICTIONS]

    summary = Table(
        [["Condition (model suggestion)", condition_name],
         ["Model confidence", f"{confidence * 100:.1f}% ({band})"],
         ["Informational risk category", risk_level],
         ["Analysis date/time", created_at.strftime("%d %b %Y, %H:%M UTC")],
         ["Prepared for", user_name]],
        colWidths=[55 * mm, 80 * mm], hAlign="LEFT",
    )
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    summary_cells = [[summary, _image_flowable(image) if image else _p("Image not retained (privacy).", styles["small"])]]
    header = Table(summary_cells, colWidths=[140 * mm, 40 * mm], hAlign="LEFT")
    header.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))

    top_table = Table([["Rank", "Category", "Model probability"]] +
                      [[str(i), name, f"{p * 100:.1f}%"] for i, (name, p) in enumerate(top, 1)],
                      colWidths=[18 * mm, 100 * mm, 40 * mm], hAlign="LEFT")
    top_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
    ]))

    story = [
        _p(constants.APP_NAME, styles["title"]), _p("AI-assisted skin analysis report", styles["subtitle"]),
        header, Spacer(1, 6),
        _p("Top predictions", styles["h2"]), top_table,
        _p("About this category", styles["h2"]), _p(info.description, styles["body"]),
        _p("Common signs", styles["h2"]), _bullets(info.common_signs, styles["body"]),
        _p("General care and prevention", styles["h2"]), _bullets(info.care_and_prevention, styles["body"]),
        _p("When to seek professional advice", styles["h2"]), _bullets(info.seek_care_when, styles["body"]),
        _p("General recommendation", styles["h2"]),
        _p(confidence_rules.next_steps_guidance(risk_level, band), styles["body"]),
        _p(confidence_rules.confidence_explanation(confidence), styles["small"]),
        _p("Medical disclaimer: " + constants.MEDICAL_DISCLAIMER, styles["disclaimer"]),
    ]
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm, title=f"{constants.APP_NAME} Report",
                            author=constants.APP_NAME)
    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)


def generate_report(user_name: str, prediction: crud.PredictionRecord, image: Image.Image | None = None,
                    output_dir: Path | None = None) -> Path:
    """Write a PDF for ``prediction`` with a unique filename; returns its path."""
    output_dir = Path(output_dir or config.REPORTS_DIR)
    path = output_dir / security.unique_filename("report", "pdf")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not security.is_within_directory(path, output_dir):
            raise ReportError("Invalid report location.")
        from modules.disease_info import display_name

        # prediction.condition stores the class slug; probabilities are keyed by display name.
        build_report_pdf(path, user_name, prediction.condition, display_name(prediction.condition),
                         prediction.confidence, prediction.risk_level, prediction.probabilities,
                         prediction.created_at, image)
    except ReportError:
        raise
    except Exception:
        logger.exception("Report generation failed")
        path.unlink(missing_ok=True)
        raise ReportError("The PDF report could not be generated. Please try again.") from None
    return path


def create_and_store_report(user_id: int, user_name: str, prediction: crud.PredictionRecord,
                            image: Image.Image | None = None) -> crud.ReportRecord:
    """Generate the PDF and record it in the database (file is removed if the DB write fails)."""
    path = generate_report(user_name, prediction, image)
    try:
        return crud.create_report(user_id, prediction.id, str(path))
    except Exception:
        path.unlink(missing_ok=True)
        raise
