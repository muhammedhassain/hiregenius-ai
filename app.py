from flask import Flask, render_template, request, send_file
from PyPDF2 import PdfReader
import os
import time
import io
import logging
from dotenv import load_dotenv

# ================== LOAD ENV VARIABLES ==================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ================== BASIC SETUP ==================
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
latest_rankings = []

# ================== LAZY LOAD VARIABLES ==================
llm = None
embedding_model = None

# ================== SAFE LLM LOADER ==================
def get_llm():
    global llm
    if llm is None and GROQ_API_KEY:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=600
        )
    return llm

# ================== SAFE EMBEDDING LOADER ==================
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return embedding_model

# ================== SAFE LLM INVOKE ==================
def safe_llm_invoke(chain, inputs, fallback=None):
    try:
        if not chain:
            return fallback
        return chain.invoke(inputs)
    except Exception as e:
        logging.error(f"LLM ERROR: {e}")
        return fallback

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

            resume_data.append({
                "name": "Candidate",
                "text": text
            })

        if not resume_data:
            return render_template("index.html", result={"error": "No readable resume content found."})

        # ================== VECTOR SEARCH ==================
        try:
            from langchain_community.vectorstores import Chroma

            texts = [r["text"] for r in resume_data]
            metadatas = [{"name": r["name"]} for r in resume_data]

            vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=get_embedding_model(),
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
            candidate_name = doc.metadata["name"]
            similarity = max(0, round((1 - score) * 100, 2))

            evaluated_candidates.append({
                "name": candidate_name,
                "final_score": similarity,
                "decision": "Evaluated"
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

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import letter

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

# ================== RENDER SAFE RUN ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
