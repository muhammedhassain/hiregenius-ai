from flask import Flask, render_template, request, send_file
from PyPDF2 import PdfReader
import os
import time
import io
import logging
from dotenv import load_dotenv

# ================== LOAD ENV VARIABLES ==================

# Loads .env locally (ignored safely in Render)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found. Running without LLM.")

# ================== IMPORT LLM & VECTOR DB ==================

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ================== PDF REPORT ==================

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ================== INITIAL SETUP ==================

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
latest_rankings = []

# ================== LLM SETUP (SAFE) ==================

llm = None

if GROQ_API_KEY:
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=600
    )

json_parser = JsonOutputParser()

# ================== SAFE LLM CALL ==================

def safe_llm_invoke(chain, inputs, fallback=None):
    if not chain:
        return fallback
    try:
        return chain.invoke(inputs)
    except Exception as e:
        logging.error(f"LLM ERROR: {e}")
        return fallback

# ================== EMBEDDINGS ==================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ================== PDF EXTRACTION ==================

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

# ================== NAME EXTRACTION ==================

name_prompt = PromptTemplate(
    template="""
Extract only the candidate full name from this resume.
Return only the name. If unclear return "Unknown".

Resume:
{resume_text}
""",
    input_variables=["resume_text"]
)

name_chain = name_prompt | llm if llm else None

def clean_name(raw_name):
    if not raw_name:
        return "Unknown"
    cleaned = str(raw_name).replace("\n", "").strip()
    return cleaned if cleaned else "Unknown"

# ================== EVALUATION ==================

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

# ================== INTERVIEW ==================

interview_prompt = PromptTemplate(
    template="""
Generate:
- 5 technical interview questions
- 3 behavioral interview questions

For role: {job_role}
""",
    input_variables=["job_role"]
)

interview_chain = interview_prompt | llm if llm else None

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

        resume_data = []

        for file in files[:50]:
            text = extract_text_from_pdf(file)
            if not text:
                continue

            name_response = safe_llm_invoke(
                name_chain,
                {"resume_text": text},
                fallback="Unknown"
            )

            raw_name = name_response.content if hasattr(name_response, "content") else name_response
            name = clean_name(raw_name)

            resume_data.append({
                "name": name,
                "text": text
            })

        if not resume_data:
            return render_template("index.html", result={"error": "No readable resume content found."})

        texts = [r["text"] for r in resume_data]
        metadatas = [{"name": r["name"]} for r in resume_data]

        try:
            vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=embedding_model,
                metadatas=metadatas
            )

            results = vectorstore.similarity_search_with_score(
                job_role,
                k=len(resume_data)
            )

        except Exception as e:
            logging.error(f"Vector DB ERROR: {e}")
            return render_template("index.html", result={"error": "Embedding system failed."})

        evaluated_candidates = []

        for doc, score in results:
            resume_text = doc.page_content
            candidate_name = doc.metadata["name"]
            similarity = max(0, round((1 - score) * 100, 2))

            evaluation = safe_llm_invoke(
                evaluation_chain,
                {
                    "resume_text": resume_text,
                    "job_role": job_role
                },
                fallback={"score": 0, "decision": "ERROR"}
            )

            llm_score = evaluation.get("score", 0) if isinstance(evaluation, dict) else 0
            final_score = round(llm_score * 0.6 + similarity * 0.4, 2)

            evaluated_candidates.append({
                "name": candidate_name,
                "llm_score": llm_score,
                "embedding_similarity": similarity,
                "final_score": final_score,
                "decision": evaluation.get("decision", "UNKNOWN") if isinstance(evaluation, dict) else "UNKNOWN"
            })

        evaluated_candidates.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        latest_rankings = evaluated_candidates

        questions = safe_llm_invoke(
            interview_chain,
            {"job_role": job_role},
            fallback="Interview generation unavailable."
        )

        questions_text = questions.content if hasattr(questions, "content") else str(questions)

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

# ================== RUN (RENDER SAFE) ==================

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     app.run(host="0.0.0.0", port=port)
