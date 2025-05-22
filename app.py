from flask import Flask, render_template, request, send_file, after_this_request
import os
import fitz  # PyMuPDF
import re
import nltk
import joblib
import pandas as pd
from fpdf import FPDF
import os
from werkzeug.utils import secure_filename
import uuid

nltk.download('punkt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your ML model and vectorizer here
model = joblib.load("quiz_classifier_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def sanitize_text(text):
    replacements = {
        '\u2013': '-',   
        '\u2014': '--',  
        '\u2018': "'",   
        '\u2019': "'",   
        '\u201c': '"',   
        '\u201d': '"',   
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.encode('latin-1', errors='ignore').decode('latin-1')
    return text

def clean_and_split_text(raw_text):
    text = raw_text.replace('▪', '. ').replace('▫', '. ')
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.([A-Za-z])', r'. \1', text)
    sentences = nltk.tokenize.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def filter_header_sentences(sentences):
    filtered = []
    for s in sentences:
        if s.endswith(':'):
            continue
        if len(s.split()) < 6 and sum(1 for w in s.split() if w.isupper()) > 1:
            continue
        filtered.append(s)
    return filtered

def classify_sentences(sentences):
    processed = [s.lower() for s in sentences]
    X = vectorizer.transform(processed)
    preds = model.predict(X)
    return pd.DataFrame({"sentence": sentences, "quiz_worthy": preds})

def export_to_pdf(sentences, output_path):
    pdf = FPDF()
    pdf.add_page()

    margin = 10
    pdf.rect(margin, margin, pdf.w - 2*margin, pdf.h - 2*margin)

    logo_path = 'static/logo.png'
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=20, y=8, w=30)

    try:
        pdf.add_font('CatPaw', '', 'fonts/CatPaw-zr0OG.ttf', uni=True)
        pdf.set_font('CatPaw', '', 36)  
    except RuntimeError:
        pdf.set_font('Arial', 'B', 16)


    title_x = 20 + 30 + 20
    title_y = 15
    pdf.set_xy(title_x, title_y)
    pdf.cell(0, 10, "Reviewer Generator")

    pdf.ln(30)

    try:
        pdf.add_font('DejaVu', '', 'fonts/DejaVuSans.ttf', uni=True)
        pdf.set_font('DejaVu', '', 12)
    except RuntimeError:
        pdf.set_font('Arial', '', 12)

    pdf.set_auto_page_break(auto=True, margin=15)

    for sentence in sentences:
        safe_sentence = sanitize_text(sentence)

        pdf.multi_cell(0, 10, f"• {safe_sentence}", border=1)
        pdf.ln(2)

    pdf.output(output_path)


from flask import after_this_request

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('pdf_file')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            raw_text = extract_text_from_pdf(filepath)
            sentences = clean_and_split_text(raw_text)
            sentences = filter_header_sentences(sentences)
            df_results = classify_sentences(sentences)
            quiz_sentences = df_results[df_results['quiz_worthy'] == 1]['sentence'].tolist()

            unique_id = str(uuid.uuid4())
            output_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f'generated_reviewer+catify_{unique_id}.pdf')
            export_to_pdf(quiz_sentences, output_pdf_path)

            @after_this_request
            def cleanup(response):
                try:
                    os.remove(filepath)
                    os.remove(output_pdf_path)
                except Exception as e:
                    app.logger.error(f"Error deleting temp files: {e}")
                return response

            return send_file(output_pdf_path, as_attachment=True, download_name="Reviewer.pdf", mimetype='application/pdf')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
