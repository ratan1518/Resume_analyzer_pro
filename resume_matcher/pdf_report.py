from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = 50
TOP_MARGIN = PAGE_HEIGHT - 50
BOTTOM_MARGIN = 50
LINE_HEIGHT = 16
FONT_NAME = "Helvetica"
FONT_NAME_BOLD = "Helvetica-Bold"
FONT_SIZE = 11
TITLE_SIZE = 16
MAX_WIDTH = PAGE_WIDTH - (LEFT_MARGIN * 2)


def wrap_line(text, font_name=FONT_NAME, font_size=FONT_SIZE):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        if stringWidth(candidate, font_name, font_size) <= MAX_WIDTH:
            current = candidate
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def build_pdf_report(report_text):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    y = TOP_MARGIN
    pdf.setTitle("Resume Match Report")

    for raw_line in report_text.splitlines():
        if y <= BOTTOM_MARGIN:
            pdf.showPage()
            y = TOP_MARGIN

        if raw_line == "RESUME MATCH REPORT":
            pdf.setFont(FONT_NAME_BOLD, TITLE_SIZE)
            pdf.drawString(LEFT_MARGIN, y, raw_line)
            y -= LINE_HEIGHT + 6
            continue

        if raw_line.endswith(":") or raw_line == "===================":
            pdf.setFont(FONT_NAME_BOLD, FONT_SIZE)
        else:
            pdf.setFont(FONT_NAME, FONT_SIZE)

        wrapped_lines = wrap_line(raw_line, FONT_NAME_BOLD if raw_line.endswith(":") else FONT_NAME, FONT_SIZE)
        for line in wrapped_lines:
            if y <= BOTTOM_MARGIN:
                pdf.showPage()
                y = TOP_MARGIN
                pdf.setFont(FONT_NAME, FONT_SIZE)
            pdf.drawString(LEFT_MARGIN, y, line)
            y -= LINE_HEIGHT

        if raw_line == "":
            y -= 2

    pdf.save()
    buffer.seek(0)
    return buffer
