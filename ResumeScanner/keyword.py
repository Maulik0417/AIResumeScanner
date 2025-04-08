from flask import Flask, request, render_template, jsonify
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer
import nltk
import os
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keybert import KeyBERT
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag, word_tokenize, ne_chunk
from nltk.tree import Tree
import yake
from summa import keywords
import numpy as np

# Initialize Flask app
app = Flask(__name__)
# nltk.download('words')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')

# Initialize KeyBERT model
keybert_model = KeyBERT(model="all-MiniLM-L6-v2")  # Using the same model as SBERT


# Function to extract text from a PDF resume
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text.strip()

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    words = [word for word in words if word.isalpha() and word not in stop_words]  # Remove punctuation & stopwords
    return " ".join(words)

# Function to extract keywords from job description using KeyBERT
def extract_keywords_with_keybert(text, num, range):
    keywords = keybert_model.extract_keywords(text, keyphrase_ngram_range=(range, range), stop_words='english', top_n=num)
    return [kw[0] for kw in keywords]  # Return just the keywords, not the scores

def extract_keywords_with_keybert2(keywords,num):
    # Check if keywords is a list of tuples and extract the first element (the keyword) from each
    if isinstance(keywords, list) and isinstance(keywords[0], tuple):
        keywords = [kw[0] for kw in keywords]  # Extract only the keyword (first element of the tuple)
    
    # Now keywords is a list of strings, and we can safely join them
    keyword_scores = keybert_model.extract_keywords(' '.join(keywords), keyphrase_ngram_range=(1, 1), stop_words='english', top_n=num)
    
    # Return just the keywords (ignore the scores)
    return [kw[0] for kw in keyword_scores]

def extract_keywords_with_tfidf(text):
    # Use TF-IDF to extract important keywords
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    return tfidf_vectorizer.get_feature_names_out()


def extract_keywords_with_tfidf2(keywords, num):
    # Ensure 'keywords' is a list of strings, as needed by the TF-IDF vectorizer
    if isinstance(keywords, list) and isinstance(keywords[0], tuple):
        keywords = [kw[0] for kw in keywords]  # Extract only the keyword (first element of the tuple)
    
    # Join keywords into a single string to be processed by the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
    tfidf_matrix = tfidf_vectorizer.fit_transform(keywords)  # Fit the vectorizer on the keywords
    
    # Get feature names (words)
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    
    # Calculate the sum of TF-IDF scores for each keyword (sum over all documents)
    tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
    
    # Get the indices of the top 20 keywords based on the highest TF-IDF scores
    top_indices = tfidf_scores.argsort()[-num:][::-1]
    
    # Get the top 20 keywords
    top_keywords = feature_names[top_indices]
    
    return top_keywords.tolist()


def extract_keywords_yake(text, num, range):
    keyword_extractor = yake.KeywordExtractor(n=range, lan="en", dedupLim=0.9, top=num)
    keywords = keyword_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

def extract_keywords_yake2(keywords,num):
    # Ensure 'keywords' is a list of strings, as needed by YAKE
    if isinstance(keywords, list) and isinstance(keywords[0], tuple):
        keywords = [kw[0] for kw in keywords]  # Extract only the keyword (first element of the tuple)

    # Combine keywords into a single string (since YAKE expects a large block of text)
    text = ' '.join(keywords)
    
    # Initialize the YAKE extractor
    keyword_extractor = yake.KeywordExtractor(n=1, lan="en", dedupLim=0.9, top=num)
    
    # Extract keywords from the combined text
    extracted_keywords = keyword_extractor.extract_keywords(text)
    
    # Extract just the keyword (not the score)
    top_keywords = [kw[0] for kw in extracted_keywords]
    
    return top_keywords


def extract_keywords_combined(text):

    keybert_keywords = extract_keywords_with_keybert(text,100,1)

    tfidf_keywords = extract_keywords_with_tfidf(text)

    yake_keywords= extract_keywords_yake(text, 100, 1)

    combined_keywords = list(set(keybert_keywords).union(set(tfidf_keywords)).union(set(yake_keywords)))
    combined_keywords= extract_keywords_with_keybert2(combined_keywords,50)
    return combined_keywords

def extract_ngrams_combined(text):

    keybert_ngrams = extract_keywords_with_keybert(text,1000,2)

    # tfidf_keywords = extract_keywords_with_tfidf(text)

    #yake_ngrams= extract_keywords_yake(text, 1000, 2)
    #!!!!!!!!!!YAKE NGRAMS DONT WORK

    # combined_ngrams = list(set(keybert_ngrams).union(set(yake_ngrams)))
    # combined_keywords= extract_keywords_with_keybert2(combined_keywords,50)
    # return combined_keywords
    return keybert_ngrams

def get_matching_keywords(resume_text, job_description):
    # Extract keywords from both the resume and job description
    resume_keywords = extract_keywords_combined(resume_text)
    job_keywords = extract_keywords_combined(job_description)
    
    
    # Find common keywords
    matching_keywords = set(resume_keywords).intersection(set(job_keywords))
    
    return matching_keywords

def get_matching_ngrams(resume_text, job_description):
    # Extract keywords from both the resume and job description
    resume_ngrams = extract_ngrams_combined(resume_text)
    job_ngrams = extract_ngrams_combined(job_description)
    
    
    # Find common keywords
    matching_ngrams = set(resume_ngrams).intersection(set(job_ngrams))
    
    return matching_ngrams

# Route to display the form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    # Check if resume file is provided
    if 'resume' not in request.files or 'job_desc' not in request.form:
        return jsonify({"error": "Missing resume or job description!"})
    
    resume_file = request.files['resume']
    job_description = request.form['job_desc']

    #################
    job_description = preprocess_text(job_description)
    
    if resume_file.filename.endswith('.pdf'):
        # Save the uploaded resume temporarily
        resume_path = os.path.join('uploads', resume_file.filename)
        resume_file.save(resume_path)
        
        # Extract text from the resume PDF
        resume_text = preprocess_text(extract_text_from_pdf(resume_path))

        matching_keywords = get_matching_keywords(resume_text, job_description)
        matching_ngrams = get_matching_ngrams(resume_text, job_description)
        
        # Calculate semantic similarity between resume and job description

        # Return the similarity score as a response
        return jsonify({
            "matching_keywords": list(matching_keywords),
            "matching_ngrams": list(matching_ngrams)
        })
    else:
        return jsonify({"error": "Unsupported file format. Please upload a PDF."})

if __name__ == '__main__':
    # Make sure the 'uploads' folder exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)