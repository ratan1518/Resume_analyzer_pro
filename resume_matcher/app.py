from pathlib import Path

from flask import Flask, redirect, render_template, request, send_file, url_for

from matcher import analyze_match
from parser import extract_pdf_text
from pdf_report import build_pdf_report


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))


def extract_submission():
    resume_text = request.form.get("resume_text", "").strip()
    job_description = request.form.get("job_description", "").strip()
    source_label = request.form.get("source_label", "Pasted resume text").strip() or "Pasted resume text"
    error_message = None

    resume_file = request.files.get("resume_file")
    if resume_file and resume_file.filename:
        if resume_file.filename.lower().endswith(".pdf"):
            try:
                resume_text = extract_pdf_text(resume_file)
                source_label = f"Extracted from PDF: {resume_file.filename}"
            except Exception:
                error_message = "Could not read that PDF. Try a text-based PDF or paste the resume manually."
        else:
            error_message = "Please upload a PDF resume or paste the resume text manually."

    return {
        "resume_text": resume_text,
        "job_description": job_description,
        "source_label": source_label,
        "error_message": error_message,
    }


def build_result_context(page_name):
    submission = extract_submission()
    if submission["error_message"]:
        return None, submission
    if not submission["resume_text"] or not submission["job_description"]:
        return None, submission

    result = analyze_match(submission["resume_text"], submission["job_description"])
    return result, submission


@app.route("/", methods=["GET", "POST"])
def index():
    resume_text = ""
    job_description = ""
    source_label = "Pasted resume text"
    error_message = None

    if request.method == "POST":
        submission = extract_submission()
        resume_text = submission["resume_text"]
        job_description = submission["job_description"]
        source_label = submission["source_label"]
        error_message = submission["error_message"]

        if resume_text and job_description and not error_message:
            result = analyze_match(resume_text, job_description)
            return render_template(
                "results.html",
                result=result,
                page_name="summary",
                resume_text=resume_text,
                job_description=job_description,
                source_label=source_label,
            )

    return render_template(
        "index.html",
        error_message=error_message,
        has_result=False,
        resume_text=resume_text,
        job_description=job_description,
        source_label=source_label,
    )


def render_results_page(page_name):
    if request.method != "POST":
        return redirect(url_for("index"))

    result, submission = build_result_context(page_name)
    if submission["error_message"]:
        return render_template(
            "index.html",
            error_message=submission["error_message"],
            has_result=False,
            resume_text=submission["resume_text"],
            job_description=submission["job_description"],
            source_label=submission["source_label"],
        )
    if not result:
        return redirect(url_for("index"))

    return render_template(
        "results.html",
        result=result,
        page_name=page_name,
        resume_text=submission["resume_text"],
        job_description=submission["job_description"],
        source_label=submission["source_label"],
    )


@app.route("/results", methods=["POST"])
@app.route("/results/summary", methods=["POST"])
def results_summary():
    return render_results_page("summary")


@app.route("/results/skills", methods=["POST"])
def results_skills():
    return render_results_page("skills")


@app.route("/results/strategy", methods=["POST"])
def results_strategy():
    return render_results_page("strategy")


@app.route("/download-report", methods=["POST"])
def download_report():
    result, submission = build_result_context("summary")
    if submission["error_message"] or not result:
        return ("No report available yet. Run an analysis first.", 400)

    report_bytes = build_pdf_report(result["report_text"])
    return send_file(
        report_bytes,
        as_attachment=True,
        download_name="resume_match_report.pdf",
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    app.run(debug=True)
