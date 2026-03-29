import re
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer


SKILL_PATTERNS = {
    "python": [r"\bpython\b"],
    "sql": [r"\bsql\b", r"\bstructured query language\b"],
    "mysql": [r"\bmysql\b"],
    "postgresql": [r"\bpostgresql\b", r"\bpostgres\b"],
    "excel": [r"\bexcel\b", r"\bms excel\b", r"\bmicrosoft excel\b"],
    "power bi": [r"\bpower\s*bi\b"],
    "tableau": [r"\btableau\b"],
    "pandas": [r"\bpandas\b"],
    "numpy": [r"\bnumpy\b"],
    "scikit-learn": [r"\bscikit[-\s]?learn\b", r"\bsklearn\b"],
    "tensorflow": [r"\btensorflow\b", r"\btf\b"],
    "keras": [r"\bkeras\b"],
    "pytorch": [r"\bpytorch\b", r"\btorch\b"],
    "nlp": [r"\bnlp\b", r"\bnatural language processing\b"],
    "machine learning": [r"\bmachine learning\b", r"\bml\b"],
    "deep learning": [r"\bdeep learning\b"],
    "computer vision": [r"\bcomputer vision\b"],
    "data analysis": [r"\bdata analysis\b", r"\bdata analytics\b"],
    "data visualization": [r"\bdata visualization\b", r"\bdata visualisation\b"],
    "statistics": [r"\bstatistics\b", r"\bstatistical\b"],
    "linear regression": [r"\blinear regression\b"],
    "logistic regression": [r"\blogistic regression\b"],
    "classification": [r"\bclassification\b", r"\bclassifier\b"],
    "clustering": [r"\bclustering\b"],
    "feature engineering": [r"\bfeature engineering\b"],
    "model deployment": [r"\bmodel deployment\b", r"\bdeploy(ed|ment)? models?\b"],
    "flask": [r"\bflask\b"],
    "fastapi": [r"\bfastapi\b", r"\bfast api\b"],
    "streamlit": [r"\bstreamlit\b"],
    "django": [r"\bdjango\b"],
    "php": [r"\bphp\b"],
    "jquery": [r"\bjquery\b", r"\bj query\b"],
    "wordpress": [r"\bwordpress\b", r"\bword press\b"],
    "bootstrap": [r"\bbootstrap\b"],
    "node.js": [r"\bnode\.?js\b", r"\bnode js\b"],
    "api integration": [r"\bapi integration\b", r"\brest api\b", r"\bapis?\b"],
    "java": [r"\bjava\b"],
    "flutter": [r"\bflutter\b"],
    "ui/ux": [r"\bui\/ux\b", r"\bui ux\b", r"\buser experience\b", r"\buser interface\b"],
    "testing": [r"\btesting\b", r"\btest automation\b", r"\bqa\b"],
    "debugging": [r"\bdebugging\b", r"\bdebug\b"],
    "html": [r"\bhtml\b", r"\bhtml5\b"],
    "css": [r"\bcss\b", r"\bcss3\b"],
    "javascript": [r"\bjavascript\b", r"\bjs\b"],
    "react": [r"\breact\b", r"\breactjs\b", r"\breact\.js\b"],
    "git": [r"\bgit\b"],
    "github": [r"\bgithub\b"],
    "docker": [r"\bdocker\b"],
    "aws": [r"\baws\b", r"\bamazon web services\b"],
    "azure": [r"\bazure\b"],
    "gcp": [r"\bgcp\b", r"\bgoogle cloud\b", r"\bgoogle cloud platform\b"],
    "cloud computing": [r"\bcloud computing\b", r"\bcloud\b"],
    "devops": [r"\bdevops\b", r"\bdev ops\b"],
    "ci/cd": [r"\bci\/cd\b", r"\bcicd\b", r"\bcontinuous integration\b", r"\bcontinuous delivery\b", r"\bcontinuous deployment\b"],
    "automation": [r"\bautomation\b", r"\bautomate\b", r"\bautomated\b"],
    "infrastructure": [r"\binfrastructure\b"],
    "monitoring": [r"\bmonitoring\b", r"\bobservability\b"],
    "troubleshooting": [r"\btroubleshooting\b", r"\btroubleshoot\b"],
    "reliability": [r"\breliability\b", r"\buptime\b"],
    "linux": [r"\blinux\b"],
    "shell scripting": [r"\bshell scripting\b", r"\bbash\b", r"\bshell scripts?\b"],
    "mlops": [r"\bmlops\b", r"\bml ops\b"],
    "rag": [r"\brag\b", r"\bretrieval[-\s]augmented generation\b"],
    "llm": [r"\bllm\b", r"\bllms\b", r"\blarge language model(s)?\b"],
    "openai": [r"\bopenai\b", r"\bgpt[-\s]?[34o5]+\b"],
    "prompt engineering": [r"\bprompt engineering\b", r"\bprompt design\b"],
    "solidworks": [r"\bsolidworks\b"],
    "catia": [r"\bcatia\b"],
    "ansys": [r"\bansys\b"],
    "autodesk fusion 360": [r"\bautodesk fusion 360\b", r"\bfusion 360\b"],
    "engineering drawing": [r"\bengineering drawing\b"],
    "rendering": [r"\brendering\b"],
    "design thinking": [r"\bdesign thinking\b"],
    "cad": [r"\bcad\b", r"\bcad models?\b"],
    "simulation": [r"\bsimulation\b", r"\bsimulations\b", r"\bfea\b", r"\bcfd\b"],
    "prototyping": [r"\bprototyping\b", r"\bprototype\b"],
}

IMPORTANT_SKILLS = {
    "python",
    "sql",
    "mysql",
    "postgresql",
    "machine learning",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "nlp",
    "flask",
    "fastapi",
    "php",
    "bootstrap",
    "node.js",
    "docker",
    "aws",
    "cloud computing",
    "devops",
    "ci/cd",
    "automation",
    "infrastructure",
    "monitoring",
    "git",
    "solidworks",
    "ansys",
    "catia",
    "autodesk fusion 360",
    "cad",
    "simulation",
}

ACTION_VERBS = {
    "built",
    "developed",
    "implemented",
    "designed",
    "deployed",
    "created",
    "optimized",
    "analyzed",
    "trained",
    "evaluated",
    "improved",
    "automated",
}

ROLE_PROFILES = {
    "machine learning intern": [
        "python",
        "sql",
        "machine learning",
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "pytorch",
        "nlp",
        "model deployment",
    ],
    "data analyst intern": [
        "sql",
        "excel",
        "power bi",
        "tableau",
        "python",
        "pandas",
        "data analysis",
        "statistics",
        "data visualization",
    ],
    "software engineering intern": [
        "python",
        "javascript",
        "react",
        "html",
        "css",
        "git",
        "github",
        "flask",
        "fastapi",
        "docker",
    ],
    "web development intern": [
        "html",
        "css",
        "javascript",
        "bootstrap",
        "jquery",
        "php",
        "mysql",
        "wordpress",
        "node.js",
        "git",
    ],
    "devops intern": [
        "devops",
        "ci/cd",
        "automation",
        "cloud computing",
        "monitoring",
        "infrastructure",
        "troubleshooting",
        "reliability",
        "linux",
        "git",
        "github",
        "python",
    ],
    "engineering design intern": [
        "solidworks",
        "ansys",
        "catia",
        "autodesk fusion 360",
        "engineering drawing",
        "cad",
        "simulation",
        "prototyping",
        "rendering",
        "design thinking",
        "testing",
    ],
}

GENERIC_PHRASE_STOPWORDS = {
    "skills required",
    "skill required",
    "required skills",
    "selected intern",
    "day to day responsibilities",
    "work from home",
    "who can apply",
    "responsibilities",
    "requirements",
    "qualification",
    "qualifications",
    "job description",
    "client requirements",
    "new ideas",
    "team members",
    "cutting edge",
    "latest web",
    "earn certifications",
    "can apply",
    "implementation",
    "monitor",
    "node",
    "relevant skills",
    "skills and interests",
    "relevant skills and interests",
    "have relevant skills",
    "interests",
    "who can apply",
    "other requirements",
    "women wanting to start",
    "women wanting to restart their career",
}

DYNAMIC_PHRASE_HINTS = {
    "cloud",
    "pipeline",
    "pipelines",
    "infrastructure",
    "monitoring",
    "reliability",
    "linux",
    "kubernetes",
    "terraform",
    "ansible",
    "jenkins",
    "docker",
    "aws",
    "azure",
    "gcp",
    "mysql",
    "php",
    "wordpress",
    "bootstrap",
    "javascript",
    "react",
    "python",
    "sql",
    "devops",
    "automation",
    "scripting",
    "testing",
    "debugging",
    "deployment",
    "ci/cd",
}

NOISE_SECTION_PATTERNS = [
    r"(earn certifications.*?)(who can apply|about the internship|about the work|other requirements|$)",
    r"(learn .*?)(who can apply|about the internship|about the work|other requirements|$)",
    r"(who can apply.*?)(other requirements|$)",
    r"(\*?\s*women wanting to start\/restart their career.*?$)",
]

TITLE_PATTERNS = [
    r"are you an?\s+(.*?)\s+looking",
    r"we are seeking an?\s+(.*?)\s+to join",
    r"we are looking for an?\s+(.*?)\s+to join",
    r"about the work.*?(job|internship)\s*(.*?)(selected intern|responsibilities|skill\(s\)\s+required|who can apply|$)",
    r"about the internship\s*(.*?)(selected intern|responsibilities|skill\(s\)\s+required|who can apply|$)",
    r"we are seeking an?\s*(.*?)(selected intern|responsibilities|skill\(s\)\s+required|who can apply|$)",
    r"we are looking for an?\s*(.*?)(selected intern|responsibilities|skill\(s\)\s+required|who can apply|$)",
]

TITLE_PREFIX_WORDS = {
    "skilled",
    "talented",
    "motivated",
    "passionate",
    "proactive",
    "dynamic",
    "enthusiastic",
    "experienced",
}

JD_SECTION_RULES = [
    ("required", r"(skill\(s\)\s+required)(.*?)(who can apply|about the internship|about the work|other requirements|$)", 3.0),
    ("requirements", r"(requirements?)(.*?)(responsibilities|who can apply|preferred|other requirements|$)", 2.4),
    ("proficiency", r"(proficient in)(.*?)(selected intern|responsibilities|\.|$)", 2.2),
    ("experience", r"(experience with)(.*?)(selected intern|responsibilities|\.|$)", 2.0),
    ("responsibilities", r"(responsibilities include|day-to-day responsibilities include|selected intern's day-to-day responsibilities include)(.*?)(who can apply|skill\(s\)\s+required|$)", 1.4),
]

RESUME_SECTION_RULES = [
    ("skills", r"(skills?)(.*?)(education|experience|projects|certifications|accomplishments|hobbies|$)", 2.6),
    ("frameworks", r"(languages?\s*&\s*frameworks?)(.*?)(project management|education|$)", 2.6),
    ("projects", r"(projects?)(.*?)(hobbies|certifications|accomplishments|education|$)", 1.6),
]


def normalize_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text):
    normalized = normalize_text(text)
    return re.findall(r"[a-zA-Z0-9\-\+\.#%]+", normalized)


def remove_noise_sections(text):
    cleaned = text
    for pattern in NOISE_SECTION_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned


def extract_known_skills(text):
    normalized = normalize_text(text)
    found = []
    for skill, patterns in SKILL_PATTERNS.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            found.append(skill)
    return sorted(set(found))


def normalize_phrase(phrase):
    cleaned = normalize_text(phrase)
    cleaned = re.sub(r"^[\-\d\.\)\(,:;]+", "", cleaned).strip()
    cleaned = re.sub(r"[\-\d\.\)\(,:;]+$", "", cleaned).strip()
    cleaned = re.sub(r"\b[a-z]{0,2}nd may\b", "", cleaned).strip()
    cleaned = re.sub(r"\b\d{1,2}(st|nd|rd|th)?\s+[a-z]{3,9}'?\d{0,2}\b", "", cleaned).strip()
    cleaned = re.sub(r"\b(can|have|are|is|be|with|for|to|of|and|or)\b\s*$", "", cleaned).strip()
    cleaned = cleaned.replace("node js", "node.js").replace("react js", "react")
    cleaned = cleaned.replace("git & github", "git github")
    cleaned = cleaned.replace("autodesk fusion", "autodesk fusion 360")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


def canonicalize_phrase(phrase):
    normalized = normalize_phrase(phrase)
    for skill, patterns in SKILL_PATTERNS.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            return skill
    return normalized


def is_valid_dynamic_phrase(phrase, allow_unknown=False):
    phrase = normalize_phrase(phrase)
    if not phrase or phrase in GENERIC_PHRASE_STOPWORDS:
        return False
    if re.search(r"\b(apply|available|duration|months?|start|career|women|internship|office|full time)\b", phrase):
        return False
    if re.search(r"\b\d", phrase):
        return False
    if len(phrase) > 40:
        return False
    tokens = phrase.split()
    if len(tokens) > 4:
        return False
    if len(tokens) == 1 and len(tokens[0]) <= 2:
        return False
    if all(token in {"and", "or", "the", "with", "for", "using", "to"} for token in tokens):
        return False
    if not re.search(r"[a-z]", phrase):
        return False
    if not allow_unknown and phrase not in SKILL_PATTERNS:
        if not any(token in DYNAMIC_PHRASE_HINTS for token in tokens):
            return False
    return True


def split_candidate_fragments(text):
    return re.split(r"[\n,;|]+|\band\b", text)


def extract_candidate_phrases(text, allow_unknown=False):
    candidates = []
    for raw in split_candidate_fragments(text):
        cleaned = normalize_phrase(raw)
        if is_valid_dynamic_phrase(cleaned, allow_unknown=allow_unknown):
            candidates.append(canonicalize_phrase(cleaned))
    return sorted(set(candidates))


def extract_job_title(job_description):
    normalized = normalize_text(job_description)
    for pattern in TITLE_PATTERNS:
        match = re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue

        title = normalize_phrase(match.group(2) if len(match.groups()) >= 2 and match.group(2) else match.group(1))
        title = re.sub(r"^(are you an?|are you a|we are seeking an?|we are looking for an?)\s+", "", title).strip()
        title = re.sub(r"\s+", " ", title).strip(" -:")
        title_tokens = title.split()
        while title_tokens and title_tokens[0] in TITLE_PREFIX_WORDS:
            title_tokens = title_tokens[1:]
        title = " ".join(title_tokens)
        if title and len(title.split()) <= 6 and title not in {"selected intern", "work from home internship", "job internship"}:
            return title
    return ""


def extract_section_items(text, rules):
    weighted_items = defaultdict(float)

    for rule in rules:
        if len(rule) == 3:
            _, pattern, weight = rule
            allow_unknown = weight >= 2.8
        elif len(rule) == 4:
            _, pattern, weight, allow_unknown = rule
        else:
            _, pattern = rule
            weight = 1.0
            allow_unknown = False
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            body = match.group(2)
            for item in extract_known_skills(body):
                weighted_items[item] = max(weighted_items[item], weight)
            if weight >= 2.0:
                for item in extract_candidate_phrases(body, allow_unknown=allow_unknown):
                    weighted_items[item] = max(weighted_items[item], weight)

    return dict(weighted_items)


def extract_resume_skills(text):
    known = set(extract_known_skills(text))
    weighted = extract_section_items(text, RESUME_SECTION_RULES)
    return sorted(known | set(weighted))


def extract_job_skill_weights(text):
    cleaned_text = remove_noise_sections(text)
    weighted = defaultdict(float)

    for skill in extract_known_skills(cleaned_text):
        weighted[skill] = max(weighted[skill], 1.0)

    for skill, weight in extract_section_items(cleaned_text, JD_SECTION_RULES).items():
        weighted[skill] = max(weighted[skill], weight)

    return dict(weighted)


def tfidf_similarity(text_a, text_b):
    cleaned_a = normalize_text(text_a)
    cleaned_b = normalize_text(text_b)
    if not cleaned_a or not cleaned_b:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform([cleaned_a, cleaned_b])
    similarity_matrix = matrix * matrix.T
    return float(similarity_matrix[0, 1])


def phrase_similarity(phrase_a, phrase_b):
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    matrix = vectorizer.fit_transform([phrase_a, phrase_b])
    similarity_matrix = matrix * matrix.T
    return float(similarity_matrix[0, 1])


def token_overlap_score(phrase_a, phrase_b):
    tokens_a = set(tokenize(phrase_a))
    tokens_b = set(tokenize(phrase_b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def phrases_match(phrase_a, phrase_b):
    normalized_a = canonicalize_phrase(phrase_a)
    normalized_b = canonicalize_phrase(phrase_b)
    if normalized_a == normalized_b:
        return True

    char_score = phrase_similarity(normalized_a, normalized_b)
    token_score = token_overlap_score(normalized_a, normalized_b)
    contains_match = (
        normalized_a in normalized_b
        or normalized_b in normalized_a
        or token_score >= 0.75
    )

    return char_score >= 0.9 or (char_score >= 0.72 and contains_match)


def match_skills(resume_skills, job_skills):
    resume_set = set(resume_skills)
    matched = set()

    for job_skill in job_skills:
        if job_skill in resume_set:
            matched.add(job_skill)
            continue

        for resume_skill in resume_skills:
            if phrases_match(job_skill, resume_skill):
                matched.add(job_skill)
                break

    return sorted(matched)


def weighted_skill_score(matched_skills, job_skill_weights):
    if not job_skill_weights:
        return 0.0

    earned = 0.0
    total = 0.0
    matched_set = set(matched_skills)

    for skill, base_weight in job_skill_weights.items():
        importance_boost = 1.0 if skill in IMPORTANT_SKILLS else 0.0
        weight = base_weight + importance_boost
        total += weight
        if skill in matched_set:
            earned += weight

    return earned / total * 100 if total else 0.0


def evidence_score(resume_text):
    normalized = normalize_text(resume_text)
    tokens = set(tokenize(resume_text))
    numbers = len(re.findall(r"\b\d+[%+]?\b", normalized))
    verbs = len(ACTION_VERBS & tokens)
    score = min(numbers * 6 + verbs * 7, 100)
    return float(score)


def readiness_band(overall_score):
    if overall_score >= 80:
        return "Strong Match"
    if overall_score >= 65:
        return "Promising Match"
    if overall_score >= 50:
        return "Moderate Match"
    return "Needs Improvement"


def infer_role(job_skill_weights, job_description):
    normalized = normalize_text(job_description)
    extracted_title = extract_job_title(job_description)
    if extracted_title:
        if "intern" in extracted_title.split():
            return extracted_title
        if any(token in extracted_title.split() for token in {"engineer", "developer", "analyst", "specialist", "associate"}):
            return f"{extracted_title} intern"

    if not job_skill_weights:
        return "software engineering intern"

    role_scores = {}
    weighted_skills = set(job_skill_weights)
    for role_name, profile in ROLE_PROFILES.items():
        overlap = weighted_skills & set(profile)
        score = 0.0
        for skill in overlap:
            score += job_skill_weights.get(skill, 0.0)
        role_scores[role_name] = score

    best_role = max(role_scores, key=role_scores.get)
    if role_scores[best_role] > 0:
        return best_role

    if "devops" in normalized:
        return "devops intern"
    if "data analyst" in normalized or "business analyst" in normalized:
        return "data analyst intern"
    if "web development" in normalized:
        return "web development intern"
    return "software engineering intern"


def role_recommendations(role_name, resume_skills, job_skills):
    profile = ROLE_PROFILES.get(role_name, [])
    gaps = [skill for skill in profile if skill in job_skills and skill not in resume_skills]

    recommendations = {
        "machine learning intern": [
            "Highlight one project with dataset cleaning, model training, evaluation metrics, and deployment.",
            "Add tools like pandas, scikit-learn, TensorFlow, and SQL in a dedicated Skills section.",
            "Quantify project results with accuracy, F1 score, latency, or user impact where possible.",
        ],
        "data analyst intern": [
            "Add dashboard, reporting, and SQL analysis projects near the top of the resume.",
            "Use measurable analysis outcomes such as revenue lift, efficiency gains, or forecasting accuracy.",
            "Group Excel, Power BI, Tableau, SQL, and statistics together for faster ATS scanning.",
        ],
        "software engineering intern": [
            "Show full-stack or backend projects with APIs, databases, and deployment steps.",
            "Use bullets that emphasize implementation, debugging, testing, and feature delivery.",
            "Highlight Git, React, Flask/FastAPI, and deployment tooling if you have used them.",
        ],
        "web development intern": [
            "Move your strongest HTML, CSS, JavaScript, and React projects to the top of the resume.",
            "If you have used Bootstrap, PHP, MySQL, WordPress, or Node.js, mention them explicitly in projects or skills.",
            "Add one bullet that highlights debugging, testing, and website feature implementation.",
        ],
        "devops intern": [
            "Add any cloud, Linux, automation, CI/CD, or deployment experience near the top of the resume.",
            "Highlight scripting, infrastructure work, system reliability, or monitoring if you have done it.",
            "If you have only software projects, add bullets about deployment, uptime, debugging, or automation to make the fit clearer.",
        ],
        "engineering design intern": [
            "Bring CAD, prototyping, simulation, and product-design experience to the top of the resume.",
            "Explicitly list tools such as SolidWorks, CATIA, ANSYS, Fusion 360, or simulation software if you have used them.",
            "Add hands-on build, testing, manufacturing, or documentation experience to show practical engineering ownership.",
        ],
    }.get(role_name, [])

    if not recommendations:
        recommendations = [
            "Move the most role-relevant projects and technical skills higher in the resume.",
            "Mirror the strongest tool and responsibility phrases from the job description where they are genuinely true.",
            "Add measurable outcomes so the recruiter can quickly see impact, ownership, and relevance.",
        ]

    return {
        "role_name": role_name,
        "priority_gaps": gaps[:5],
        "recommendations": recommendations,
    }


def infer_resume_role(resume_skills):
    if not resume_skills:
        return "generalist candidate"

    role_scores = {}
    skill_set = set(resume_skills)
    for role_name, profile in ROLE_PROFILES.items():
        role_scores[role_name] = len(skill_set & set(profile))

    best_role = max(role_scores, key=role_scores.get)
    if role_scores[best_role] == 0:
        return "generalist candidate"
    return best_role


def section_presence_score(resume_text):
    normalized = normalize_text(resume_text)
    score = 0
    for pattern in (
        r"\bskills?\b",
        r"\bprojects?\b",
        r"\beducation\b",
        r"\bexperience\b",
        r"\bcertifications?\b",
    ):
        if re.search(pattern, normalized):
            score += 18
    return float(min(score, 100))


def skill_depth_score(resume_skills):
    if not resume_skills:
        return 0.0
    important_overlap = len(set(resume_skills) & IMPORTANT_SKILLS)
    base = min(len(resume_skills) * 6, 70)
    boost = min(important_overlap * 6, 30)
    return float(min(base + boost, 100))


def analyze_resume_profile(resume_text, label="Resume"):
    resume_skills = extract_resume_skills(resume_text)
    impact_score = evidence_score(resume_text)
    structure_score = section_presence_score(resume_text)
    skill_score = skill_depth_score(resume_skills)
    overall_score = round(skill_score * 0.45 + impact_score * 0.35 + structure_score * 0.20)
    inferred_role = infer_resume_role(resume_skills)

    strengths = []
    if resume_skills:
        strengths.append(f"Shows {len(resume_skills)} detected technical skills.")
    if any(skill in IMPORTANT_SKILLS for skill in resume_skills):
        strengths.append("Covers important industry-relevant tools and technical signals.")
    if impact_score >= 55:
        strengths.append("Uses action-oriented language and measurable evidence well.")
    if structure_score >= 54:
        strengths.append("Includes clear resume sections that improve readability and ATS scanning.")

    improvements = []
    if skill_score < 45:
        improvements.append("Add a clearer technical skills section with role-relevant tools and frameworks.")
    if impact_score < 45:
        improvements.append("Strengthen project bullets with measurable outcomes, ownership, and delivery impact.")
    if structure_score < 45:
        improvements.append("Use clearer sections such as Skills, Projects, Experience, and Certifications.")
    if len(resume_skills) <= 4:
        improvements.append("Expand the resume with stronger project tooling, libraries, and platform keywords.")

    role_tips = role_recommendations(inferred_role, resume_skills, ROLE_PROFILES.get(inferred_role, []))

    return {
        "label": label,
        "resume_skills": resume_skills,
        "overall_score": overall_score,
        "skill_score": round(skill_score, 1),
        "impact_score": round(impact_score, 1),
        "structure_score": round(structure_score, 1),
        "inferred_role": inferred_role,
        "strengths": strengths or ["Has a usable starting foundation but needs stronger positioning."],
        "improvements": improvements or ["Keep refining project depth and role-specific phrasing for stronger competitiveness."],
        "recommendations": role_tips["recommendations"],
    }


def winner_label(score_a, score_b, label_a, label_b):
    if score_a > score_b:
        return label_a
    if score_b > score_a:
        return label_b
    return "Tie"


def compare_resumes(resume_a_text, resume_b_text, label_a="Resume A", label_b="Resume B"):
    profile_a = analyze_resume_profile(resume_a_text, label=label_a)
    profile_b = analyze_resume_profile(resume_b_text, label=label_b)

    skills_a = set(profile_a["resume_skills"])
    skills_b = set(profile_b["resume_skills"])
    shared_skills = sorted(skills_a & skills_b)
    only_a = sorted(skills_a - skills_b)
    only_b = sorted(skills_b - skills_a)

    category_winners = {
        "overall": winner_label(profile_a["overall_score"], profile_b["overall_score"], label_a, label_b),
        "skills": winner_label(profile_a["skill_score"], profile_b["skill_score"], label_a, label_b),
        "impact": winner_label(profile_a["impact_score"], profile_b["impact_score"], label_a, label_b),
        "structure": winner_label(profile_a["structure_score"], profile_b["structure_score"], label_a, label_b),
    }

    comparison_insights = []
    if only_a:
        comparison_insights.append(f"{label_a} stands out with: {', '.join(only_a[:5])}.")
    if only_b:
        comparison_insights.append(f"{label_b} stands out with: {', '.join(only_b[:5])}.")
    if shared_skills:
        comparison_insights.append(f"Both resumes overlap on: {', '.join(shared_skills[:6])}.")
    if category_winners["overall"] != "Tie":
        comparison_insights.append(f"{category_winners['overall']} currently has the stronger overall resume profile.")

    head_to_head_tips = {
        label_a: [],
        label_b: [],
    }
    if only_b:
        head_to_head_tips[label_a].append(f"To catch up, consider adding evidence around: {', '.join(only_b[:4])}.")
    if profile_a["impact_score"] < profile_b["impact_score"]:
        head_to_head_tips[label_a].append("Use more measurable outcomes and stronger action verbs in project bullets.")
    if profile_a["structure_score"] < profile_b["structure_score"]:
        head_to_head_tips[label_a].append("Make section headings and resume organization clearer for faster scanning.")

    if only_a:
        head_to_head_tips[label_b].append(f"To catch up, consider adding evidence around: {', '.join(only_a[:4])}.")
    if profile_b["impact_score"] < profile_a["impact_score"]:
        head_to_head_tips[label_b].append("Add stronger quantified outcomes and ownership language to key projects.")
    if profile_b["structure_score"] < profile_a["structure_score"]:
        head_to_head_tips[label_b].append("Tighten section structure so recruiters can scan technical depth faster.")

    return {
        "resume_a": profile_a,
        "resume_b": profile_b,
        "shared_skills": shared_skills,
        "unique_a": only_a,
        "unique_b": only_b,
        "category_winners": category_winners,
        "comparison_insights": comparison_insights or ["Both resumes are closely matched and need more differentiating evidence."],
        "head_to_head_tips": head_to_head_tips,
    }


def build_report(result):
    lines = [
        "RESUME MATCH REPORT",
        "===================",
        f"Overall Match: {result['overall_score']}%",
        f"Readiness Band: {result['readiness_band']}",
        f"Weighted Skill Match: {result['skill_score']}%",
        f"Semantic Alignment: {result['semantic_score']}%",
        f"Resume Evidence Score: {result['impact_score']}%",
        "",
        f"Suggested Role: {result['role_summary']['role_name'].title()}",
        "",
        "Matched Skills:",
        ", ".join(result["matched_skills"]) if result["matched_skills"] else "None",
        "",
        "Missing Skills:",
        ", ".join(result["missing_skills"]) if result["missing_skills"] else "None",
        "",
        "Strengths:",
    ]
    lines.extend(f"- {item}" for item in result["strengths"] or ["No strong strengths detected yet."])
    lines.append("")
    lines.append("Suggested Improvements:")
    lines.extend(f"- {item}" for item in result["improvements"] or ["No immediate improvements suggested."])
    lines.append("")
    lines.append("ATS Tips:")
    lines.extend(f"- {item}" for item in result["ats_tips"])
    lines.append("")
    lines.append("Role-Specific Recommendations:")
    lines.extend(f"- {item}" for item in result["role_summary"]["recommendations"])
    if result["role_summary"]["priority_gaps"]:
        lines.append("")
        lines.append("Priority Role Gaps:")
        lines.extend(f"- {item}" for item in result["role_summary"]["priority_gaps"])
    lines.append("")
    return "\n".join(lines)


def analyze_match(resume_text, job_description):
    resume_skills = extract_resume_skills(resume_text)
    job_skill_weights = extract_job_skill_weights(job_description)
    job_skills = sorted(job_skill_weights, key=lambda skill: (-job_skill_weights[skill], skill))

    matched_skills = match_skills(resume_skills, job_skills)
    matched_set = set(matched_skills)
    missing_skills = [skill for skill in job_skills if skill not in matched_set]
    extra_skills = sorted(set(resume_skills) - set(job_skills))

    skill_score = weighted_skill_score(matched_skills, job_skill_weights)
    semantic_score = tfidf_similarity(resume_text, job_description) * 100
    impact_score = evidence_score(resume_text)
    overall_score = round(skill_score * 0.58 + semantic_score * 0.22 + impact_score * 0.20)

    strengths = []
    if matched_skills:
        strengths.append(f"Matched {len(matched_skills)} job-relevant skill keywords.")
    if any(skill in matched_set for skill in IMPORTANT_SKILLS):
        strengths.append("Resume covers important technical signals employers usually screen for.")
    if impact_score >= 50:
        strengths.append("Resume includes action-oriented language and measurable evidence.")
    if semantic_score >= 55:
        strengths.append("Resume wording is aligned well with the target role.")

    improvements = []
    if missing_skills:
        improvements.append(f"Add or strengthen these missing skills: {', '.join(missing_skills[:6])}.")
    if impact_score < 40:
        improvements.append("Add stronger achievement bullets with metrics, outcomes, or project impact.")
    if semantic_score < 45:
        improvements.append("Mirror important job-description wording in your project and experience bullets.")
    if len(job_skills) >= 5 and len(matched_skills) <= 2:
        improvements.append("Move your most relevant projects and tools higher in the resume.")

    ats_tips = [
        "Use the exact tool names from the job description where they are genuinely true.",
        "Lead project bullets with action verbs and measurable outcomes.",
        "Group technical skills into a clean Skills section for easier screening.",
    ]
    if missing_skills:
        ats_tips.insert(0, f"Tailor the resume for this role by addressing: {', '.join(missing_skills[:4])}.")

    role_name = infer_role(job_skill_weights, job_description)
    role_summary = role_recommendations(role_name, resume_skills, job_skills)

    result = {
        "overall_score": overall_score,
        "readiness_band": readiness_band(overall_score),
        "skill_score": round(skill_score, 1),
        "semantic_score": round(semantic_score, 1),
        "impact_score": round(impact_score, 1),
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_skills": extra_skills,
        "strengths": strengths,
        "improvements": improvements,
        "ats_tips": ats_tips,
        "role_summary": role_summary,
    }
    result["report_text"] = build_report(result)
    return result
