from pathlib import Path

from flask import Flask, redirect, render_template, request, send_file, url_for

from matcher import analyze_match, compare_resumes
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


def extract_resume_input(text_key, file_key, label_key, default_label):
    resume_text = request.form.get(text_key, "").strip()
    source_label = request.form.get(label_key, default_label).strip() or default_label
    error_message = None

    resume_file = request.files.get(file_key)
    if resume_file and resume_file.filename:
        if resume_file.filename.lower().endswith(".pdf"):
            try:
                resume_text = extract_pdf_text(resume_file)
                source_label = f"Extracted from PDF: {resume_file.filename}"
            except Exception:
                error_message = "Could not read one of the PDF resumes. Try a text-based PDF or paste the resume manually."
        else:
            error_message = "Please upload PDF resumes or paste the resume text manually."

    return {
        "resume_text": resume_text,
        "source_label": source_label,
        "error_message": error_message,
    }


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


@app.route("/compare", methods=["GET", "POST"])
def compare():
    resume_a_text = ""
    resume_b_text = ""
    source_label_a = "Pasted resume text"
    source_label_b = "Pasted resume text"
    label_a = "Resume A"
    label_b = "Resume B"
    error_message = None
    comparison = None

    if request.method == "POST":
        label_a = request.form.get("label_a", "Resume A").strip() or "Resume A"
        label_b = request.form.get("label_b", "Resume B").strip() or "Resume B"
        submission_a = extract_resume_input("resume_a_text", "resume_a_file", "source_label_a", "Pasted resume text")
        submission_b = extract_resume_input("resume_b_text", "resume_b_file", "source_label_b", "Pasted resume text")

        resume_a_text = submission_a["resume_text"]
        resume_b_text = submission_b["resume_text"]
        source_label_a = submission_a["source_label"]
        source_label_b = submission_b["source_label"]
        error_message = submission_a["error_message"] or submission_b["error_message"]

        if resume_a_text and resume_b_text and not error_message:
            comparison = compare_resumes(resume_a_text, resume_b_text, label_a=label_a, label_b=label_b)

    return render_template(
        "compare.html",
        comparison=comparison,
        error_message=error_message,
        resume_a_text=resume_a_text,
        resume_b_text=resume_b_text,
        source_label_a=source_label_a,
        source_label_b=source_label_b,
        label_a=label_a,
        label_b=label_b,
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
