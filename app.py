from flask import Flask, render_template, request, send_file
from PyPDF2 import PdfReader
import os
import time
import io
import logging
from dotenv import load_dotenv

# ================== LOAD ENV ==================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ================== IMPORT LLM ==================

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ================== PDF REPORT ==================

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ================== INIT ==================

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
latest_rankings = []

# ================== LLM SETUP ==================

llm = None
if GROQ_API_KEY:
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=500
    )

json_parser = JsonOutputParser()

# ================== SAFE LLM ==================

def safe_llm_invoke(chain, inputs, fallback=None):
    if not chain:
        return fallback
    try:
        return chain.invoke(inputs)
    except Exception as e:
        logging.error(f"LLM ERROR: {e}")
        return fallback

# ================== PDF TEXT ==================

def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        logging.error(f"PDF ERROR: {e}")
        return ""

# ================== NAME EXTRACTION PROMPT ==================

name_prompt = PromptTemplate(
    template="""
You are an expert resume parser.

Extract ONLY the candidate's full name from the resume.

Rules:
- Return only the full name
- No explanation
- No extra text
- If unclear return "Unknown"

Resume:
{resume_text}
""",
    input_variables=["resume_text"]
)

name_chain = name_prompt | llm if llm else None

def clean_name(raw_name):
    if not raw_name:
        return "Unknown"

    name = str(raw_name).strip()
    name = name.replace("\n", " ")
    name = name.replace("Name:", "")
    name = name.replace("Full Name:", "")
    name = name.strip()

    if len(name.split()) > 5:
        return "Unknown"

    return name if name else "Unknown"

# ================== EVALUATION PROMPT ==================

evaluation_prompt = PromptTemplate(
    template="""
You are an AI recruitment evaluator.

Job Role:
{job_role}

Resume:
{resume_text}

Return ONLY valid JSON:
{{
  "score": number (0-100),
  "decision": "HIRE" or "CONSIDER" or "REJECT"
}}
""",
    input_variables=["resume_text", "job_role"]
)

evaluation_chain = (
    evaluation_prompt | llm | json_parser
    if llm else None
)

# ================== MAIN ROUTE ==================

@app.route("/", methods=["GET", "POST"])
def index():
    global latest_rankings
    result = {}

    if request.method == "POST":

        start_time = time.time()
        job_role = request.form.get("job_role")
        files = request.files.getlist("resumes")

        if not job_role:
            return render_template("index.html", result={"error": "Enter job role."})

        if not files:
            return render_template("index.html", result={"error": "Upload resumes."})

        evaluated_candidates = []

        for file in files[:10]:  # Limit for memory safety

            text = extract_text_from_pdf(file)

            if not text:
                continue

            # Limit resume size (prevents memory crash)
            text = text[:4000]

            # ---------- NAME EXTRACTION ----------
            name_response = safe_llm_invoke(
                name_chain,
                {"resume_text": text},
                fallback="Unknown"
            )

            raw_name = name_response.content if hasattr(name_response, "content") else name_response
            name = clean_name(raw_name)

            if name == "Unknown":
                name = file.filename.replace(".pdf", "")

            # ---------- EVALUATION ----------
            evaluation = safe_llm_invoke(
                evaluation_chain,
                {
                    "resume_text": text,
                    "job_role": job_role
                },
                fallback={"score": 0, "decision": "ERROR"}
            )

            if isinstance(evaluation, dict):
                score = evaluation.get("score", 0)
                decision = evaluation.get("decision", "UNKNOWN")
            else:
                score = 0
                decision = "UNKNOWN"

            evaluated_candidates.append({
                "name": name,
                "final_score": score,
                "decision": decision
            })

        evaluated_candidates.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        latest_rankings = evaluated_candidates

        execution_time = round(time.time() - start_time, 2)

        result = {
            "candidates": evaluated_candidates,
            "execution_time": execution_time
        }

    return render_template("index.html", result=result)

# ================== DOWNLOAD REPORT ==================

@app.route("/download_report")
def download_report():
    if not latest_rankings:
        return "No rankings available yet."

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("HireGenius AI Ranked Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    for idx, candidate in enumerate(latest_rankings, start=1):
        text = f"{idx}. {candidate['name']} - Score: {candidate['final_score']} - Decision: {candidate['decision']}"
        elements.append(Paragraph(text, styles["Normal"]))
        elements.append(Spacer(1, 8))

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="HireGenius_Ranked_Report.pdf",
        mimetype="application/pdf"
    )

# ================== RUN ==================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
