from flask import Flask, render_template, request, jsonify, send_file
import re
import io
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Helper: Summarize text ---
def summarize_text(text, max_sentences=3):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return " ".join(sentences[:min(max_sentences, len(sentences))]) if sentences else "No text provided."

# --- Helper: Extract Clauses ---
def extract_clauses(text):
    clauses = {}
    text_lower = text.lower()
    if re.search(r'\btermination\b', text_lower):
        clauses["Termination"] = "Contains termination-related clause."
    if re.search(r'\b(penalty|fine)\b', text_lower):
        clauses["Penalty"] = "Mentions penalties or fines."
    if re.search(r'\b(obligation|shall|must)\b', text_lower):
        clauses["Obligations"] = "Specifies obligations."
    if re.search(r'\bconfidential(ity)?\b', text_lower):
        clauses["Confidentiality"] = "Mentions confidentiality obligations."
    if re.search(r'\b(indemnify|indemnification)\b', text_lower):
        clauses["Indemnity"] = "Mentions indemnity clause."
    if not clauses:
        clauses["Info"] = "No key clauses detected."
    return clauses

# --- Helper: Risk Analysis ---
def risk_analysis(text):
    risk_keywords = {
        "penalty": 4, "fine": 4, "breach": 4, "default": 4,
        "termination": 3, "liability": 3, "indemnify": 3, "indemnification": 3,
        "obligation": 2, "shall": 2, "must": 2, "dispute": 2, "damages": 2,
        "confidential": 2, "non-disclosure": 2, "agreement": 1, "contract": 1
    }
    text_lower = text.lower()
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    
    total_score = 0
    sentence_risks = []
    for sentence in sentences:
        sentence_score = 0
        words = sentence.lower().split()
        for i, word in enumerate(words):
            if word in risk_keywords:
                weight = risk_keywords[word]
                negation = i > 0 and words[i-1] == "no"
                base_score = weight if not negation else -weight * 0.5
                for j in range(max(0, i-2), min(len(words), i+3)):
                    if j != i and words[j] in risk_keywords:
                        base_score += weight * 0.3
                sentence_score += base_score
                if sum(1 for w in words if w in risk_keywords) > 1:
                    sentence_score += 3
        sentence_risks.append({"sentence": sentence, "score": sentence_score})
        total_score += sentence_score
        logging.info(f"Sentence: {sentence[:50]}... Score: {sentence_score}")
    
    max_possible_score = sum(len(re.findall(rf'\b{word}\b', text_lower)) * weight * 1.5 for word, weight in risk_keywords.items()) + 3 * len(sentences)
    normalized_score = min((total_score / max(max_possible_score, 1)) * 100, 100) if max_possible_score > 0 else 0
    risk_level = "Low" if normalized_score <= 30 else "Medium" if normalized_score <= 70 else "High"
    
    return {
        "risk_score": round(normalized_score, 2),
        "risk_level": risk_level,
        "sentence_risks": sentence_risks
    }

# --- Helper: Risk Heatmap ---
def risk_heatmap(text, sentence_risks, overall_risk_level):
    highlighted = text
    highlighted_positions = []
    default_thresholds = {"sentence_low": 2, "sentence_medium": 5}
    for risk in sentence_risks:
        sentence = risk["sentence"]
        score = risk["score"]
        start_pos = highlighted.find(sentence)
        if start_pos == -1:
            continue
        end_pos = start_pos + len(sentence)
        if any(start_pos < pos_end and end_pos > pos_start for pos_start, pos_end in highlighted_positions):
            continue
        if overall_risk_level == "High" and score > 0:
            highlight_class = "highlight-high" if score > default_thresholds["sentence_medium"] else "highlight-medium"
        else:
            highlight_class = "highlight-low" if score <= default_thresholds["sentence_low"] else "highlight-medium" if score <= default_thresholds["sentence_medium"] else "highlight-high"
        highlighted = highlighted[:start_pos] + f'<span class="{highlight_class}">{sentence}</span>' + highlighted[end_pos:]
        highlighted_positions.append((start_pos, end_pos + len(f'<span class="{highlight_class}">') + len('</span>')))
    return highlighted

# --- Gemini AI Simplifier ---
def simplify_text(text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Simplify this legal text into plain English in 2-3 sentences:\n\n{text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Simplification error: {str(e)}")
        return f"⚠️ AI Simplification Error: {str(e)}"

# --- Gemini AI Q&A ---
def legal_qa(text, question, context=None):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        context_prompt = f"Previous context: {context}\n" if context else ""
        prompt = f"You are a legal assistant. Based on this contract:\n\n{text}\n\n{context_prompt}Answer the question: {question}\nIf it's a follow-up or term explanation, use the context."
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Q&A error: {str(e)}")
        return f"⚠️ AI Q&A Error: {str(e)}"

# --- Analyze Text ---
def analyze_text(text):
    if not text.strip():
        return {"result": "⚠️ No text provided."}
    summary = summarize_text(text)
    clauses = extract_clauses(text)
    risk = risk_analysis(text)
    simplified = simplify_text(text)
    highlighted = risk_heatmap(text, risk["sentence_risks"], risk["risk_level"])
    
    return {
        "summary": summary,
        "clauses": clauses,
        "risk": {"score": risk["risk_score"], "level": risk["risk_level"]},
        "simplified": simplified,
        "highlighted": highlighted
    }

# --- Compare Documents ---
def compare_documents(texts):
    results = [analyze_text(text) for text in texts]
    comparison = {
        "summary": [r["summary"] for r in results],
        "clauses": [r["clauses"] for r in results],
        "risk": [{"score": r["risk"]["score"], "level": r["risk"]["level"]} for r in results],
        "simplified": [r["simplified"] for r in results],
        "highlighted": [r["highlighted"] for r in results]
    }
    return comparison

# --- Export to PDF ---
def export_to_pdf(content):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    for doc_title, doc_content in content.items():
        story.append(Paragraph(f"<b>{doc_title}</b>", styles['Heading1']))
        for section, text in doc_content.items():
            clean_text = re.sub(r'<[^>]+>', '', text)
            story.append(Paragraph(f"<b>{section}</b>", styles['Heading2']))
            story.append(Paragraph(clean_text, styles['Normal']))
            story.append(Spacer(1, 12))
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/app')
def app_page():
    return render_template('app.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get("text", "")
    result = analyze_text(text)
    return jsonify(result)

@app.route('/voice', methods=['POST'])
def voice():
    return jsonify({"result": "Voice input handled client-side."})

@app.route('/compare', methods=['POST'])
def compare():
    if 'files' not in request.files:
        return jsonify({"result": "⚠️ No files uploaded."}), 400
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"result": "⚠️ No files selected."}), 400
    texts = []
    for file in files:
        if not file.filename.endswith('.pdf'):
            continue
        try:
            with pdfplumber.open(file) as pdf:
                text = "".join(page.extract_text() or "" for page in pdf.pages)
                if text.strip():
                    texts.append(text)
        except Exception as e:
            logging.error(f"Error processing PDF: {str(e)}")
            continue
    if not texts:
        return jsonify({"result": "⚠️ Unable to extract text from PDFs."}), 400
    comparison = compare_documents(texts)
    return jsonify(comparison)

@app.route('/qa', methods=['POST'])
def qa():
    data = request.json
    text = data.get("text", "")
    question = data.get("question", "")
    context = data.get("context", "")
    if not text or not question:
        return jsonify({"result": "⚠️ Both text and question are required."}), 400
    answer = legal_qa(text, question, context)
    return jsonify({"answer": answer, "context": answer})

@app.route('/download', methods=['POST'])
def download():
    data = request.json
    logging.info(f"Download data received: {data}")
    content = {}
    try:
        if "clauses" in data and isinstance(data["clauses"], dict):
            content["Document Analysis"] = {
                "Summary": data.get("summary", ""),
                "Key Clauses": "\n".join(f"{k}: {v}" for k, v in data.get("clauses", {}).items()),
                "Risk Level": f"{data.get('risk', {}).get('level', 'N/A')} (Score: {data.get('risk', {}).get('score', 'N/A')})",
                "Simplified Version": data.get("simplified", ""),
                "Risk Heatmap": data.get("highlighted", "").replace('<span class="highlight-', '<span style="background-color:').replace('">', '"; color: white;">').replace('</span>', '</span>')
            }
        else:
            for i, (summary, clauses, risk, simplified, highlighted) in enumerate(zip(
                data.get("summary", []),
                data.get("clauses", []),
                data.get("risk", []),
                data.get("simplified", []),
                data.get("highlighted", [])
            )):
                content[f"Document {i+1}"] = {
                    "Summary": summary,
                    "Key Clauses": "\n".join(f"{k}: {v}" for k, v in clauses.items()),
                    "Risk Level": f"{risk.get('level', 'N/A')} (Score: {risk.get('score', 'N/A')})",
                    "Simplified Version": simplified,
                    "Risk Heatmap": highlighted.replace('<span class="highlight-', '<span style="background-color:').replace('">', '"; color: white;">').replace('</span>', '</span>')
                }
        buffer = export_to_pdf(content)
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='legal_analysis.pdf'
        )
    except Exception as e:
        logging.error(f"PDF generation error: {str(e)}")
        return jsonify({"result": f"⚠️ Error generating PDF: {str(e)}"}), 500

@app.route('/voice_output', methods=['POST'])
def voice_output_route():
    return jsonify({"status": "Voice output handled client-side."})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
