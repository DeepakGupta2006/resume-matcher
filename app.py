import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

st.set_page_config(page_title="Resume ↔ Job Match Score", page_icon="📄", layout="centered")



STOPWORDS = set("""
a an the and or but if while is are was were be been being to of in on for with
as by at from this that these those it its it's your you i we our us they their
he she his her not no nor so than then too very can will just don should now
""".split())


def extract_text_from_pdf(file) -> str:
    text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def top_keywords(text: str, n: int = 25):
    words = [w for w in clean_text(text).split() if w not in STOPWORDS and len(w) > 2]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:n]


def compute_match(resume_text: str, jd_text: str):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([clean_text(resume_text), clean_text(jd_text)])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    resume_kw = set(top_keywords(resume_text, 40))
    jd_kw = top_keywords(jd_text, 40)
    missing = [w for w in jd_kw if w not in resume_kw]

    return round(score * 100, 1), missing[:15]



st.title("📄 Resume ↔ Job Description Matcher")
st.write(
    "Upload your resume and paste a job description to get a match score "
    "and see which important keywords from the job posting are missing "
    "from your resume."
)

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    resume_text_input = st.text_area("...or paste resume text", height=200)

with col2:
    jd_text = st.text_area("Paste Job Description", height=280)

if st.button("Check Match", type="primary"):
    resume_text = ""
    if resume_file is not None:
        resume_text = extract_text_from_pdf(resume_file)
    elif resume_text_input.strip():
        resume_text = resume_text_input

    if not resume_text.strip() or not jd_text.strip():
        st.error("Please provide both a resume (PDF or text) and a job description.")
    else:
        score, missing_keywords = compute_match(resume_text, jd_text)

        st.subheader(f"Match Score: {score}%")
        st.progress(min(int(score), 100))

        if score >= 75:
            st.success("Strong match! Your resume aligns well with this job.")
        elif score >= 50:
            st.warning("Moderate match. Consider adding some missing keywords below.")
        else:
            st.error("Low match. Your resume may need significant tailoring.")

        if missing_keywords:
            st.subheader("Keywords in the job description you may be missing")
            st.write(", ".join(missing_keywords))
        else:
            st.info("No major keyword gaps found.")

st.caption("Tip: this score is based on keyword/text similarity (TF-IDF), similar to how many ATS systems screen resumes.")