import os
import pickle
import torch
import torch.nn.functional as F
from flask import Flask, request, render_template, url_for
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer
import pandas as pd

app = Flask(__name__)

# path to folder containing images
IMAGE_FOLDER = 'coco_images_resized'

# load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B-32"
pretrained = "openai"

# load model, preprocess, tokenizer
model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = get_tokenizer(model_name)
model = model.to(device).eval()

# load image embeddings
with open('image_embeddings.pickle', 'rb') as f:
    embeddings_df = pd.read_pickle(f)

# Utility function: cosine similarity
def cosine_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2.unsqueeze(0)).item()

# Search function
def search(query_embedding, k=5):
    scores = []
    for _, row in embeddings_df.iterrows():
        similarity = cosine_similarity(query_embedding, torch.tensor(row['embedding']).to(device))
        scores.append((row['file_name'], similarity))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores[:k]

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        query_type = request.form.get('query_type')  # Get selected query type
        text_query = request.form.get('text_query', '').strip()
        image_file = request.files.get('image_query')
        hybrid_weight = float(request.form.get('hybrid_weight', 0.5))

        query_embedding = None

        # query types
        if query_type == 'text' and text_query: # Text Query: Get text embedding
            text_tokens = tokenizer([text_query])
            query_embedding = F.normalize(model.encode_text(text_tokens.to(device)))
        elif query_type == 'image' and image_file: # Image Query: Get image embedding
            image = Image.open(image_file).convert('RGB')
            image_tensor = preprocess_val(image).unsqueeze(0).to(device)
            query_embedding = F.normalize(model.encode_image(image_tensor))
        elif query_type == 'hybrid' and text_query and image_file: # Hybrid Query: Combine text and image embeddings
            text_tokens = tokenizer([text_query])
            text_embedding = F.normalize(model.encode_text(text_tokens.to(device)))
            image = Image.open(image_file).convert('RGB')
            image_tensor = preprocess_val(image).unsqueeze(0).to(device)
            image_embedding = F.normalize(model.encode_image(image_tensor))
            query_embedding = F.normalize(hybrid_weight * text_embedding + (1 - hybrid_weight) * image_embedding)

        if query_embedding is not None: # Perform search if a query_embedding was successfully created
            results = search(query_embedding)

    return render_template('index.html', results=results, image_folder=IMAGE_FOLDER)

if __name__ == '__main__':
    app.run(debug=True)