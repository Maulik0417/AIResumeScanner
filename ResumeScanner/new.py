from flask import Flask, request, render_template, jsonify
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import nltk
from nltk.corpus import wordnet
# Initialize Flask app
app = Flask(__name__)
nltk.download('wordnet')
nltk.download('stopwords')
# Load a pretrained model for semantic similarity (MiniLM)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2',device="cpu")
model2 = SentenceTransformer('all-MiniLM-L6-v2',device="cpu")
# Function to extract text from a PDF resume
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text.strip()

# Function to calculate semantic similarity using Sentence-Transformers
def calculate_semantic_similarity(resume_text, job_description):
    # Encode both the resume and the job description using the Sentence-Transformer model
    embeddings = model.encode([resume_text, job_description], convert_to_tensor=True)
    
    # Calculate cosine similarity between the resume and job description embeddings
    cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    
    return cosine_sim.item()

# def get_synonyms(word):
#     synonyms = set()
#     for syn in wordnet.synsets(word):
#         for lemma in syn.lemmas():
#             synonyms.add(lemma.name().replace("_", " "))
#     return synonyms

def get_similar_words(word, all_words, threshold=0.7):
    word_embedding = model2.encode([word])[0]
    all_embeddings = model2.encode(list(all_words))
    
    similarities = cosine_similarity([word_embedding], all_embeddings)[0]
    similar_words = {word for idx, score in enumerate(similarities) if score >= threshold and list(all_words)[idx] != word}
    
    return similar_words

def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    return set(feature_array[tfidf_sorting][:top_n])

def calculate_keyword_match(resume_text, job_description):
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description)

    # Expand job description keywords using embeddings
    expanded_job_keywords = set(job_keywords)
    for word in job_keywords:
        expanded_job_keywords.update(get_similar_words(word, job_keywords))

    # Calculate the intersection of expanded keywords
    matched_keywords = resume_keywords & expanded_job_keywords
    keyword_match_score = len(matched_keywords) / len(expanded_job_keywords) if expanded_job_keywords else 0
    
    return matched_keywords, keyword_match_score

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
    
    if resume_file.filename.endswith('.pdf'):
        # Save the uploaded resume temporarily
        resume_path = os.path.join('uploads', resume_file.filename)
        resume_file.save(resume_path)
        
        # Extract text from the resume PDF
        resume_text = extract_text_from_pdf(resume_path)
        
        # Calculate semantic similarity between resume and job description
        similarity_score = calculate_semantic_similarity(resume_text, job_description)

        matched_keywords, keyword_match_score = calculate_keyword_match(resume_text, job_description)
        
        # Return the similarity score as a response
        return jsonify({
            "cosine_similarity": similarity_score,
            "message": f"Semantic Similarity: {similarity_score:.4f}",
            "keyword_match_score": keyword_match_score,
            "matched_keywords": list(matched_keywords)
    
        })
    else:
        return jsonify({"error": "Unsupported file format. Please upload a PDF."})

if __name__ == '__main__':
    # Make sure the 'uploads' folder exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)