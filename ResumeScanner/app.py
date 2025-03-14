from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import pdfplumber

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model for text similarity
model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Function to extract text from a PDF resume
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to analyze resume and match it to a job description
def analyze_resume(resume_text, job_description):
    result = model(resume_text, candidate_labels=[job_description])
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if 'resume' not in request.files or 'job_desc' not in request.form:
        return jsonify({"error": "Missing resume or job description!"})

    resume_file = request.files['resume']
    job_description = request.form['job_desc']

    if resume_file.filename.endswith('.pdf'):
        resume_text = extract_text_from_pdf(resume_file)
    else:
        return jsonify({"error": "Unsupported resume format!"})

    analysis_result = analyze_resume(resume_text, job_description)
    return jsonify(analysis_result)

if __name__ == '__main__':
    app.run(debug=True)