from flask import Flask, request, render_template, jsonify
import pdfplumber
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from key import keywords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from sentence_transformers import SentenceTransformer, util

# Place this early in the script (once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize Flask app
app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

stop_words = set(stopwords.words('english'))

# Function to extract text from PDF

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text.strip()

# Clean text for matching
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words and word not in string.punctuation]
    return " ".join(tokens)

# Match unigrams and n-grams in text
def match_keywords(text, keyword_list):
    cleaned_text = clean_text(text)
    vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english')
    vectorizer.fit(keyword_list)
    text_vector = vectorizer.transform([cleaned_text])
    found_keywords = set()
    for keyword in keyword_list:
        if keyword.lower() in vectorizer.get_feature_names_out():
            idx = list(vectorizer.get_feature_names_out()).index(keyword.lower())
            if text_vector[0, idx] > 0:
                found_keywords.add(keyword)
    return found_keywords

def semantic_match(keywords_resume, keywords_job, threshold=0.50):
    # Generate embeddings for each keyword
    embeddings_resume = model.encode(keywords_resume, convert_to_tensor=True)
    embeddings_job = model.encode(keywords_job, convert_to_tensor=True)

    # Compute cosine similarity between all keyword pairs
    similarity_matrix = util.cos_sim(embeddings_resume, embeddings_job)

    matched_keywords = set()
    for i, keyword_r in enumerate(keywords_resume):
        for j, keyword_j in enumerate(keywords_job):
            if similarity_matrix[i][j] >= threshold:  # Only match if similarity is above the threshold
                matched_keywords.add((keyword_r, keyword_j))
    return matched_keywords

# Calculate score based on keyword match
def calculate_score(matching_keywords, job_matches):
    if not job_matches:
        return 0
    return round(len(matching_keywords) / len(job_matches) * 100, 2)

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
        resume_path = os.path.join('uploads', resume_file.filename)
        resume_file.save(resume_path)

        resume_text = extract_text_from_pdf(resume_path)
        job_text = job_description

        # Extract keywords using ngrams and clean up
        resume_matches = match_keywords(resume_text, keywords)
        job_matches = match_keywords(job_text, keywords)

        # Perform semantic match
        semantic_matches = semantic_match(list(resume_matches), list(job_matches))

        # Combine exact matches and semantic matches
        matching_keywords = list(set(resume_matches & job_matches) | {match[1] for match in semantic_matches})

        score = calculate_score(matching_keywords, job_matches)

        return jsonify({
            "matching_keywords": matching_keywords,
            "resume_keywords_found": list(resume_matches),
            "job_keywords_found": list(job_matches),
            "semantic_matches": list(semantic_matches),
            "score": score
        })
    else:
        return jsonify({"error": "Unsupported file format. Please upload a PDF."})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)