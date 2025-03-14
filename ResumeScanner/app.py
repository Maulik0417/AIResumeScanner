from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer, util
import re
import string
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load models
zero_shot_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device="cpu")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Load stopwords
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text.strip()

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    return " ".join(words)

# Extract n-grams
def get_ngrams(text, n=2):
    words = text.split()
    return set([" ".join(gram) for gram in ngrams(words, n)])

# Analyze resume similarity
def analyze_resume(resume_text, job_description):
    resume_text = preprocess_text(resume_text)
    job_description = preprocess_text(job_description)
    
    # Cosine similarity
    resume_embedding = embedding_model.encode(resume_text, convert_to_tensor=True)
    job_embedding = embedding_model.encode(job_description, convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
    
    # Keyword match
    resume_words = set(resume_text.split())
    job_words = set(job_description.split())
    matched_keywords = list(resume_words & job_words)
    keyword_match_score = len(matched_keywords) / max(len(job_words), 1)
    
    # N-gram match
    resume_ngrams = get_ngrams(resume_text, n=2) | get_ngrams(resume_text, n=3)
    job_ngrams = get_ngrams(job_description, n=2) | get_ngrams(job_description, n=3)
    matched_ngrams = list(resume_ngrams & job_ngrams)
    ngram_match_score = len(matched_ngrams) / max(len(job_ngrams), 1)
    
    # Final Score Calculation
    final_score = (0.75 * cosine_similarity) + (0.15 * keyword_match_score) + (0.10 * ngram_match_score)
    
    return {
        "cosine_similarity": round(cosine_similarity, 3),
        "keyword_match_score": round(keyword_match_score, 3),
        "ngram_match_score": round(ngram_match_score, 3),
        "final_score": round(final_score, 3),
        "matched_keywords": matched_keywords,
        "matched_ngrams": matched_ngrams
    }

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
