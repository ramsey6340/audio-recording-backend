# app.py
from flask import Flask, request, jsonify, send_file
import requests
import json
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
import torch
import os
from io import BytesIO

app = Flask(__name__)

# Définissez votre clé API Hugging Face
os.environ["HUGGINGFACE_TOKEN"] = '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'

# Fonction pour transcrire l'audio en texte Bamanankan
def transcribe_audio(file_path):
    url = "https://bamanankanapi.kabakoo.africa/hackathon/transcribe_to_bam"
    token = "6f13469b-7bf9-4442-ae55-58689acb2c6d"

    with open(file_path, 'rb') as audio_file:
        files = {'file': audio_file}
        data = {'token': token}
        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            response_data = json.loads(response.text)
            message = response_data.get('data', {}).get('message', '')
            return message
        else:
            raise Exception(f"Error in transcription: {response.status_code}, {response.text}")

# Charger le modèle et le tokenizer pour la traduction
translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# Fonction pour traduire le texte Bamanankan en anglais
def translate_text_bamanankan_to_english(text):
    inputs = translator_tokenizer(text, return_tensors="pt")
    translated_tokens = translator_model.generate(**inputs, forced_bos_token_id=translator_tokenizer.lang_code_to_id["eng_Latn"])
    translated_text = translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translated_text[0]

# Fonction pour générer des images en utilisant Stable Diffusion
def generate_image(description):
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    pipe = pipe.to("cuda")  # Utiliser GPU si disponible

    # Générer l'image
    image = pipe(description, num_inference_steps=50, guidance_scale=7.5).images[0]
    return image

@app.route('/generate-image', methods=['POST'])
def generate_image_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Transcription de l'audio
        file_path = "/tmp/audio_file.mpeg"
        file.save(file_path)
        transcription = transcribe_audio(file_path)
        if transcription:
            # Traduction en anglais
            translated_text = translate_text_bamanankan_to_english(transcription)

            # Génération de l'image
            image = generate_image(translated_text)

            # Sauvegarder l'image dans un buffer
            img_io = BytesIO()
            image.save(img_io, 'PNG')
            img_io.seek(0)

            return send_file(img_io, mimetype='image/png')
        else:
            return jsonify({"error": "Transcription failed."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
