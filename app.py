import streamlit as st
import pandas as pd
import re
import pdfplumber
import docx
import gspread
from google.oauth2.service_account import Credentials
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "resume123"

st.set_page_config(page_title="AI Resume Selector", page_icon="🤖", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

@st.cache_resource
def get_client():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    return gspread.authorize(creds)

def get_candidates_sheet():
    return get_client().open("ai-resume-selector").worksheet("Candidates")

def get_jd_sheet():
    return get_client().open("ai-resume-selector").worksheet("JobDescription")

def save_job_description(jd):
    try:
        sheet = get_jd_sheet()
        sheet.clear()
        sheet.append_row(["Job Description"])
        sheet.append_row([jd])
    except Exception as e:
        st.error(f"Error saving job description: {e}")

def load_job_description():
    try:
        sheet = get_jd_sheet()
        data = sheet.get_all_values()
        if len(data) > 1:
            return data[1][0]
        return ""
    except:
        return ""

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

def get_match_score(resume_text, job_description):
    cleaned_jd = clean_text(job_description)
    cleaned_resume = clean_text(resume_text)
    jd_embedding = model.encode([cleaned_jd])
    resume_embedding = model.encode([cleaned_resume])
    score = cosine_similarity(jd_embedding, resume_embedding).flatten()[0]
    return round(float(score) * 100, 2)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def candidate_page():
    st.title("📄 Submit Your CV")
    st.markdown("### Welcome! Please fill in your details and upload your resume.")

    jd = load_job_description()
    if jd:
        st.info(f"📋 **Current Job Opening:**\n\n{jd}")
    else:
        st.warning("⚠️ No job description posted yet. Please check back later.")
        return

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("👤 Full Name")
    with col2:
        email = st.text_input("📧 Email Address")

    uploaded_file = st.file_uploader(
        "📂 Upload Your CV (PDF or DOCX)",
        type=["pdf", "docx"]
    )

    if st.button("📤 Submit CV", use_container_width=True):
        if not name:
            st.warning("⚠️ Please enter your full name.")
        elif not email:
            st.warning("⚠️ Please enter your email address.")
        elif not uploaded_file:
            st.warning("⚠️ Please upload your CV.")
        else:
            with st.spinner("⏳ Submitting your CV..."):
                try:
                    text = extract_text(uploaded_file)
                    score = get_match_score(text, jd)
                    sheet = get_candidates_sheet()
                    sheet.append_row([
                        name,
                        email,
                        uploaded_file.name,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        score,
                        text[:300]
                    ])
                    st.success("✅ CV submitted successfully! We will contact you soon.")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error submitting CV: {e}")

def login_page():
    st.title("🔐 Admin Login")
    st.markdown("Only the hiring manager can access this panel.")

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

def admin_page():
    st.title("🤖 AI Resume Selector - Admin Panel")

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.divider()

    st.subheader("📝 Set Job Description")
    current_jd = load_job_description()

    new_jd = st.text_area(
        "Enter Job Description:",
        value=current_jd,
        height=200,
        placeholder="We are looking for a Data Scientist with Python, ML, SQL..."
    )

    if st.button("💾 Save Job Description", use_container_width=True):
        if not new_jd:
            st.warning("⚠️ Job description cannot be empty.")
        else:
            save_job_description(new_jd)
            st.success("✅ Job description saved! Candidates will see this immediately.")

    st.divider()

    st.subheader("🗄️ All Submitted Candidates")
    if st.button("📋 Load Candidates", use_container_width=True):
        with st.spinner("Loading..."):
            try:
                sheet = get_candidates_sheet()
                data = sheet.get_all_records()
                if data:
                    db_df = pd.DataFrame(data)
                    db_df.columns = ["Name", "Email", "CV File", "Upload Time", "Match Score (%)", "Text Preview"]
                    db_df = db_df.sort_values("Match Score (%)", ascending=False).reset_index(drop=True)
                    db_df.index += 1

                    st.success(f"Best Match: {db_df.iloc[0]['Name']} - {db_df.iloc[0]['Match Score (%)']}%")

                    st.dataframe(
                        db_df[["Name", "Email", "CV File", "Upload Time", "Match Score (%)"]],
                        use_container_width=True
                    )

                    st.subheader("📊 Candidate Score Comparison")
                    st.bar_chart(db_df.set_index("Name")["Match Score (%)"])

                    csv = db_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download All Candidates CSV",
                        data=csv,
                        file_name="all_candidates.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("📭 No candidates have submitted their CVs yet.")
            except Exception as e:
                st.error(f"❌ Error loading candidates: {e}")

page = st.sidebar.selectbox(
    "Navigation",
    ["📄 Submit CV", "🔐 Admin Panel"]
)

if page == "🔐 Admin Panel":
    if st.session_state.logged_in:
        admin_page()
    else:
        login_page()
else:
    candidate_page()
```

---

### ⚠️ Make sure your Google Sheet has 2 tabs:

**Tab 1 — Candidates** with headers in Row 1:
```
Name | Email | CV File | Upload Time | Match Score | Text Preview
```

**Tab 2 — JobDescription** with headers in Row 1:
```
Job Description
