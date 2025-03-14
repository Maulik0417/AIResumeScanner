from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import pdfplumber
import torch
from sentence_transformers import SentenceTransformer, util
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')

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



# Load a more accurate NLP model for similarity scoring


# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")  # Force CPU execution

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

def extract_keywords(text, top_n=10):
    """Extracts meaningful keywords by removing stopwords and counting frequency."""
    words = re.findall(r'\b\w+\b', text.lower())  # Tokenize words
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]  # Remove stopwords and very short words
    word_freq = {word: words.count(word) for word in set(words)}
    sorted_keywords = sorted(word_freq, key=word_freq.get, reverse=True)[:top_n]
    return set(sorted_keywords)

# def analyze_resume(resume_text, job_description):
#     # Ensure tensors are processed on CPU
#     resume_embedding = model.encode(resume_text, convert_to_tensor=True).to("cpu")
#     job_embedding = model.encode(job_description, convert_to_tensor=True).to("cpu")

#     # Calculate cosine similarity (ranges from 0 to 1)
#     similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
    
#     return {"score": similarity_score}


def analyze_resume(resume_text, job_description):
    """Analyzes resume using cosine similarity and keyword matching."""
    # Compute cosine similarity
    resume_embedding = model.encode(resume_text, convert_to_tensor=True).to("cpu")
    job_embedding = model.encode(job_description, convert_to_tensor=True).to("cpu")
    cosine_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()

    # Extract keywords from job description
    job_keywords = extract_keywords(job_description, top_n=10)
    resume_keywords = extract_keywords(resume_text, top_n=10)

    # Calculate keyword match percentage
    matched_keywords = job_keywords.intersection(resume_keywords)
    keyword_match_score = len(matched_keywords) / len(job_keywords) if job_keywords else 0  # Avoid divide by zero

    # Combine scores (weight cosine similarity more)
    final_score = (0.7 * cosine_score) + (0.3 * keyword_match_score)

    return {
        "cosine_similarity": round(cosine_score, 3),
        "keyword_match_score": round(keyword_match_score, 3),
        "final_score": round(final_score, 3),
        "matched_keywords": list(matched_keywords)  # Show which keywords matched
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