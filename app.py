from flask import Flask, render_template, request, send_file
from PyPDF2 import PdfReader
import os
import time
import io
import logging
import re
from dotenv import load_dotenv

# ================== LOAD ENV ==================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ================== INIT ==================

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
latest_rankings = []

# ================== IMPORT LLM ==================

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ================== PDF REPORT ==================

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ================== LLM SETUP ==================

llm = None
evaluation_chain = None
question_chain = None

if GROQ_API_KEY:
    try:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=400
        )

        # ---------- Evaluation Prompt ----------
        evaluation_prompt = PromptTemplate(
            template="""
You are an AI recruitment evaluator.

Evaluate the resume strictly for the given role.

Job Role:
{job_role}

Resume:
{resume_text}

Return ONLY valid JSON in this exact format:
{{
  "score": number between 0-100,
  "decision": "HIRE" or "CONSIDER" or "REJECT"
}}
""",
            input_variables=["resume_text", "job_role"]
        )

        json_parser = JsonOutputParser()
        evaluation_chain = evaluation_prompt | llm | json_parser

        # ---------- Interview Question Prompt ----------
        question_prompt = PromptTemplate(
            template="""
You are a senior technical interviewer.

Generate interview questions for the role: {job_role}

Format STRICTLY as:

Technical Questions:
1. ...
2. ...
3. ...
4. ...
5. ...

Behavioral Questions:
1. ...
2. ...
3. ...

Do not add extra commentary.
""",
            input_variables=["job_role"]
        )

        question_chain = question_prompt | llm

    except Exception as e:
        logging.error(f"LLM INIT ERROR: {e}")
        llm = None

# ================== SAFE LLM CALL ==================

def safe_llm_invoke(chain, inputs, expect_json=False):
    if not chain:
        return {"score": 0, "decision": "ERROR"} if expect_json else "AI unavailable"

    try:
        response = chain.invoke(inputs)

        if expect_json:
            if isinstance(response, dict):
                return response
            return {"score": 0, "decision": "ERROR"}

        if hasattr(response, "content"):
            return response.content
        return str(response)

    except Exception as e:
        logging.error(f"LLM ERROR: {e}")
        return {"score": 0, "decision": "ERROR"} if expect_json else "AI generation failed."

# ================== IMPROVED SEMANTIC MATCH ==================

def calculate_semantic_match(job_role, resume_text):
    if not job_role or not resume_text:
        return 0

    job_role = job_role.lower()
    resume_text = resume_text.lower()

    # Remove special characters
    job_role = re.sub(r'[^a-z0-9 ]', ' ', job_role)
    resume_text = re.sub(r'[^a-z0-9 ]', ' ', resume_text)

    job_words = set(job_role.split())
    resume_words = set(resume_text.split())

    if len(job_words) == 0:
        return 0

    # Count direct matches
    direct_matches = job_words.intersection(resume_words)

    # Count partial matches
    partial_matches = set()
    for word in job_words:
        for resume_word in resume_words:
            if word in resume_word or resume_word in word:
                partial_matches.add(word)

    total_matches = direct_matches.union(partial_matches)

    score = (len(total_matches) / len(job_words)) * 100

    return round(score, 2)

# ================== PDF TEXT EXTRACTION ==================

def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text.strip()
    except Exception as e:
        logging.error(f"PDF ERROR: {e}")
        return ""

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

        for file in files[:5]:

            resume_text = extract_text_from_pdf(file)

            if not resume_text:
                continue

            resume_text = resume_text[:3000]
            candidate_name = file.filename.replace(".pdf", "")

            # ---- LLM SCORE ----
            evaluation = safe_llm_invoke(
                evaluation_chain,
                {
                    "resume_text": resume_text,
                    "job_role": job_role
                },
                expect_json=True
            )

            llm_score = evaluation.get("score", 0)
            decision = evaluation.get("decision", "REJECT")

            if not isinstance(llm_score, (int, float)):
                llm_score = 0

            # ---- SEMANTIC SCORE ----
            semantic_score = calculate_semantic_match(job_role, resume_text)

            # ---- FINAL SCORE ----
            final_score = int((llm_score * 0.7) + (semantic_score * 0.3))

            evaluated_candidates.append({
                "name": candidate_name,
                "llm_score": int(llm_score),
                "embedding_similarity": semantic_score,
                "final_score": final_score,
                "decision": decision
            })

            time.sleep(1)

        evaluated_candidates.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        latest_rankings = evaluated_candidates

        # ---- Generate Interview Questions ----
        questions_text = safe_llm_invoke(
            question_chain,
            {"job_role": job_role},
            expect_json=False
        )

        execution_time = round(time.time() - start_time, 2)

        result = {
            "candidates": evaluated_candidates,
            "questions": questions_text,
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
        line = f"{idx}. {candidate['name']} - Final Score: {candidate['final_score']} - Decision: {candidate['decision']}"
        elements.append(Paragraph(line, styles["Normal"]))
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