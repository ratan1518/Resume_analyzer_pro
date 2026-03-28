from io import BytesIO

from pypdf import PdfReader


def extract_pdf_text(file_storage):
    pdf_bytes = file_storage.read()
    file_storage.seek(0)
    reader = PdfReader(BytesIO(pdf_bytes))
    chunks = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks).strip()
