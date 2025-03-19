from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import pdfplumber
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain
import string
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import wordnet

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model for text similarity
model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",device="cpu")

STOP_WORDS = set(stopwords.words('english'))

# Function to extract text from a PDF resume
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text.strip()

# Function to get synonyms
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return synonyms

# Function to extract keywords using TF-IDF
def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    return feature_array[tfidf_sorting][:top_n]

# Function to match resume against job description
def analyze_resume(resume_text, job_description):
    resume_tokens = word_tokenize(resume_text.lower())
    job_tokens = word_tokenize(job_description.lower())
    
    resume_tokens = [word for word in resume_tokens if word not in STOP_WORDS and word not in string.punctuation]
    job_tokens = [word for word in job_tokens if word not in STOP_WORDS and word not in string.punctuation]
    
    matched_keywords = list(set(resume_tokens) & set(job_tokens))
    
    # Synonym Matching
    expanded_job_keywords = set(chain(*[get_synonyms(word) for word in job_tokens]))
    matched_keywords += [word for word in resume_tokens if word in expanded_job_keywords]
    
    # N-gram Matching
    resume_bigrams = list(ngrams(resume_tokens, 2))
    job_bigrams = list(ngrams(job_tokens, 2))
    matched_ngrams = [" ".join(pair) for pair in set(resume_bigrams) & set(job_bigrams)]
    
    # TF-IDF Similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    
    # Scoring
    keyword_match_score = len(matched_keywords) / len(set(job_tokens)) if job_tokens else 0
    ngram_match_score = len(matched_ngrams) / len(set(job_bigrams)) if job_bigrams else 0
    final_score = (cosine_sim + keyword_match_score + ngram_match_score) / 3
    
    return {
        "cosine_similarity": cosine_sim,
        "keyword_match_score": keyword_match_score,
        "ngram_match_score": ngram_match_score,
        "final_score": final_score,
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
