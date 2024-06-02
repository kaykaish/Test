from flask import Flask, request, jsonify, render_template_string
from transformers import pipeline
import pdfplumber
from PIL import Image
import pytesseract
import io

app = Flask(__name__)

# Initialize the Hugging Face pipeline for question-answering
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

@app.route('/')
def index():
    return render_template_string(open('index.html').read())

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        content = ""
        if file.filename.lower().endswith('.pdf'):
            content = extract_text_from_pdf(file)
        elif any(file.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            content = extract_text_from_image(file)
        else:
            content = file.read().decode('utf-8')
        
        return jsonify({"content": content}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get("question")
    content = data.get("content")
    
    if not question or not content:
        return jsonify({"error": "Question or content missing"}), 400

    # Use the Hugging Face pipeline to answer the question
    response = qa_pipeline(question=question, context=content)
    
    return jsonify({"answer": response['answer']}), 200

if __name__ == '__main__':
    app.run(debug=True)
