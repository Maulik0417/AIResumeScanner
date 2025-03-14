# from flask import Flask, request, jsonify, render_template
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer, util
# import pdfplumber
# import nltk
# from nltk.corpus import stopwords, wordnet
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.preprocessing import MinMaxScaler
# import string

# # Initialize Flask app
# app = Flask(__name__)

# # Load sentence transformer model for better similarity matching
# embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")

# # Function to extract text from a PDF resume
# def extract_text_from_pdf(pdf_path):
#     with pdfplumber.open(pdf_path) as pdf:
#         text = ''
#         for page in pdf.pages:
#             text += page.extract_text() or ''
#     return text

# # Function to clean text
# def clean_text(text):
#     text = text.lower().translate(str.maketrans('', '', string.punctuation))
#     words = text.split()
#     words = [word for word in words if word not in stopwords.words('english')]
#     return ' '.join(words)

# # Function to extract keywords using TF-IDF
# def extract_keywords(text, top_n=10):
#     vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = vectorizer.fit_transform([text])
#     feature_names = vectorizer.get_feature_names_out()
#     scores = tfidf_matrix.toarray()[0]
#     keyword_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
#     return [word for word, _ in keyword_scores[:top_n]]

# # Function to get synonyms for a word
# def get_synonyms(word):
#     synonyms = set()
#     for syn in wordnet.synsets(word):
#         for lemma in syn.lemmas():
#             synonyms.add(lemma.name())
#     return synonyms

# # Function to calculate similarity score
# def analyze_resume(resume_text, job_description):
#     resume_clean = clean_text(resume_text)
#     job_clean = clean_text(job_description)
    
#     # Compute sentence embeddings
#     resume_embedding = embedding_model.encode(resume_clean, convert_to_tensor=True)
#     job_embedding = embedding_model.encode(job_clean, convert_to_tensor=True)
#     cosine_similarity = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
    
#     # Extract keywords
#     resume_keywords = set(extract_keywords(resume_clean))
#     job_keywords = set(extract_keywords(job_clean))
    
#     # Expand with synonyms
#     expanded_job_keywords = job_keywords.copy()
#     for word in job_keywords:
#         expanded_job_keywords.update(get_synonyms(word))
    
#     # Calculate keyword match score
#     matched_keywords = resume_keywords.intersection(expanded_job_keywords)
#     keyword_match_score = len(matched_keywords) / max(len(job_keywords), 1)
    
#     # Normalize scores
#     scaler = MinMaxScaler()
#     scaled_scores = scaler.fit_transform([[cosine_similarity], [keyword_match_score]])
#     final_score = (scaled_scores[0][0] * 0.85) + (scaled_scores[1][0] * 0.15)
    
#     return {
#         "cosine_similarity": round(cosine_similarity, 3),
#         "keyword_match_score": round(keyword_match_score, 3),
#         "final_score": round(final_score, 3),
#         "matched_keywords": list(matched_keywords)
#     }

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/submit', methods=['POST'])
# def submit():
#     if 'resume' not in request.files or 'job_desc' not in request.form:
#         return jsonify({"error": "Missing resume or job description!"})

#     resume_file = request.files['resume']
#     job_description = request.form['job_desc']

#     if resume_file.filename.endswith('.pdf'):
#         resume_text = extract_text_from_pdf(resume_file)
#     else:
#         return jsonify({"error": "Unsupported resume format!"})

#     analysis_result = analyze_resume(resume_text, job_description)
#     return jsonify(analysis_result)

# if __name__ == '__main__':
#     nltk.download('stopwords')
#     nltk.download('wordnet')
#     app.run(debug=True)
