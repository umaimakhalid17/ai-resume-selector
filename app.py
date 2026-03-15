import streamlit as st
import pandas as pd
import re
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "resume123"

st.set_page_config(page_title="AI Resume Selector", page_icon="🤖", layout="wide")

# ─────────────────────────────────────────
# LOAD BERT MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join(para.text for para in doc.paragraphs)

def extract_text(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    return ""

def rank_resumes(jd, resumes):
    cleaned_jd = clean_text(jd)
    cleaned_resumes = [clean_text(r) for r in resumes]
    jd_embedding = model.encode([cleaned_jd])
    resume_embeddings = model.encode(cleaned_resumes)
    scores = cosine_similarity(jd_embedding, resume_embeddings).flatten()
    return scores

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ─────────────────────────────────────────
# ADMIN LOGIN PAGE
# ─────────────────────────────────────────
def login_page():
    st.title("🔐 Admin Login")
    st.markdown("Only the hiring manager can access results.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials!")

# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
def main_app():
    st.title("🤖 AI Resume Selector")
    st.markdown("Upload CVs and rank candidates automatically using BERT.")

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.divider()

    # ── Job Description ──
    st.subheader("📝 Step 1 — Enter Job Description")
    job_description = st.text_area(
        "Paste the job description here:",
        height=150,
        placeholder="We are looking for a Data Scientist with Python, ML, SQL..."
    )

    # ── Upload CVs ──
    st.subheader("📂 Step 2 — Upload Candidate CVs")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    # ── Run Ranking ──
    if st.button("🚀 Rank Candidates", use_container_width=True):
        if not job_description:
            st.warning("⚠️ Please enter a job description.")
        elif not uploaded_files:
            st.warning("⚠️ Please upload at least one CV.")
        else:
            with st.spinner("⏳ Analyzing resumes with BERT..."):
                names, texts = [], []
                for file in uploaded_files:
                    text = extract_text(file)
                    names.append(file.name)
                    texts.append(text)

                scores = rank_resumes(job_description, texts)

                results_df = pd.DataFrame({
                    "Rank": range(1, len(names) + 1),
                    "Candidate": names,
                    "Match Score (%)": (scores * 100).round(2),
                }).sort_values("Match Score (%)", ascending=False).reset_index(drop=True)

                results_df["Rank"] = range(1, len(results_df) + 1)

            # ── Results ──
            st.divider()
            st.subheader("🏆 Ranked Candidates")

            st.success(f"🥇 Best Match: **{results_df.iloc[0]['Candidate']}** — {results_df.iloc[0]['Match Score (%)']}%")

            st.dataframe(
                results_df.style.background_gradient(
                    subset=["Match Score (%)"], cmap="Greens"
                ),
                use_container_width=True
            )

            # ── Bar Chart ──
            st.subheader("📊 Score Comparison")
            st.bar_chart(results_df.set_index("Candidate")["Match Score (%)"])

            # ── Download ──
            st.subheader("💾 Download Results")
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Shortlisted Candidates CSV",
                data=csv,
                file_name="shortlisted_candidates.csv",
                mime="text/csv",
                use_container_width=True
            )

# ─────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────
if st.session_state.logged_in:
    main_app()
else:
    login_page()