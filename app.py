# app.py - minimal working version

from flask import Flask, request, jsonify
from pytesseract import pytesseract
from transformers import pipeline
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load NLP models (you have this knowledge)
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", revision="4c53496")  # Named entity recognition

@app.route('/extract', methods=['POST'])
def extract():
    """
    Accept image/PDF.
    Run OCR.
    Extract entities.
    Return JSON.
    """
    try:
        # Get image from request
        data = request.json
        image_b64 = data.get('image')
        
        # Decode
        img = Image.open(BytesIO(base64.b64decode(image_b64)))
        
        # OCR
        text = pytesseract.image_to_string(img)
        
        # NLP extraction
        entities = ner(text)
        
        return jsonify({
            'text': text,
            'entities': entities,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)
