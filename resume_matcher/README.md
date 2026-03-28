# Resume Matcher

A lightweight NLP/ML web app for internship applications. It compares a resume against a target job description and highlights:

- overall match percentage
- matching skills
- missing skills
- extra skills already present in the resume
- practical suggestions to improve alignment
- ATS-style tips and recruiter-friendly feedback
- PDF resume upload support
- downloadable analysis report
- downloadable PDF analysis report
- role-specific recommendations for ML, data analyst, and software internships

## Features

- skill extraction using curated technical keywords
- semantic similarity scoring using TF-IDF vectorization with scikit-learn
- weighted skill scoring with stronger emphasis on important technical skills
- resume evidence scoring using action verbs and measurable proof
- combined match score for quick screening
- recruiter-friendly web interface
- easy demo workflow for interviews and portfolio reviews
- exportable text report for sharing or saving analysis output
- exportable PDF report for portfolio demos and application review

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Demo input

Use the examples in [sample_inputs.md](C:/Users/Lenovo/OneDrive/Desktop/driver-project/resume_matcher/sample_inputs.md) for a quick demo.

## Resume-ready description

Built an NLP/ML-based Resume Matcher web app that compares resumes with job descriptions using TF-IDF semantic similarity, extracts overlapping and missing skills, computes weighted match scores, parses PDF resumes, and generates ATS-oriented improvement suggestions through an interactive Flask interface.

## Future upgrades

- PDF resume upload
- embedding-based semantic search
- JD-specific bullet rewriting suggestions
- recruiter dashboard for multi-resume comparison
