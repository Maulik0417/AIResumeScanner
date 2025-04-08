from flask import Flask, request, render_template, jsonify
import pdfplumber
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Initialize Flask app
app = Flask(__name__)

# Load Hugging Face NER model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Function to extract text from a PDF resume
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text.strip()

# Function to preprocess text by removing stopwords and punctuation
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    words = [word for word in words if word.isalpha() and word not in stop_words]  # Remove punctuation & stopwords
    return " ".join(words)

# Function to extract entities using Hugging Face NER
def extract_entities(text):
    entities = {
        "PERSON": [],
        "ORG": [],
        "SKILL": [],
        "JOB_TITLE": []
    }
    
    # Define a list of common skills (expand as needed)
    skill_keywords = ["Python", "Java", "JavaScript", "SQL", "C++", "React", "Node.js", "Ruby", "HTML", "CSS", "Docker"]
    
    results = ner_pipeline(text)
    
    for entity in results:
        word = entity['word']
        label = entity['entity'].split('-')[-1]  # Extract the entity type
        
        if label == "PER":
            entities["PERSON"].append(word)
        elif label == "ORG":
            entities["ORG"].append(word)
        elif word in skill_keywords:
            entities["SKILL"].append(word)
        elif label == "MISC":  # General category, sometimes includes job titles
            entities["JOB_TITLE"].append(word)
    
    return entities

# Function to calculate the match score between job description and resume using NER
def calculate_ner_score(job_desc, resume):
    job_entities = extract_entities(job_desc)
    resume_entities = extract_entities(resume)
    
    total_matches = 0
    total_entities = 0
    
    for category in job_entities:
        total_entities += len(job_entities[category])
        matched_entities = set(job_entities[category]) & set(resume_entities[category])
        total_matches += len(matched_entities)
    
    if total_entities == 0:
        return 0
    
    score = total_matches / total_entities
    return score * 100

# Route to display the form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission
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
        
        ner_score = calculate_ner_score(job_description, resume_text)
        
        return jsonify({
            "ner_score": ner_score,
        })
    else:
        return jsonify({"error": "Unsupported file format. Please upload a PDF."})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)