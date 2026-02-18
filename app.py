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

# ================== LLM ==================

llm = None
if GROQ_API_KEY:
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=600
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
    except:
        return ""

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
  "score": 0,
  "decision": ""
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

        for file in files[:20]:

            text = extract_text_from_pdf(file)
            if not text:
                continue

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
                "name": file.filename,
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

# ================== DOWNLOAD ==================

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
    app.run()
