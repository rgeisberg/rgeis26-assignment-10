from flask import Flask, request, render_template, send_from_directory
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import os

# Flask App Setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and precomputed embeddings
model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
tokenizer = get_tokenizer('ViT-B-32')
model.eval()
df = pd.read_pickle('image_embeddings.pickle')  # Precomputed embeddings
dataset_embeddings = torch.tensor([row for row in df['embedding']])
dataset_image_paths = df['file_name']

def reduce_with_pca(embeddings, n_components):
    """
    Reduces embeddings using PCA to the first `n_components` principal components.

    Args:
        embeddings (torch.Tensor): Original embeddings (2D or 1D).
        n_components (int): Number of principal components to retain.

    Returns:
        torch.Tensor: PCA-reduced embeddings (2D or 1D depending on input).
    """
    pca = PCA(n_components=min(n_components, embeddings.shape[-1]))

    # Ensure embeddings are 2D for PCA
    embeddings_np = embeddings.detach().cpu().numpy()
    if embeddings_np.ndim == 3:  # Unexpected extra dimension
        embeddings_np = embeddings_np.squeeze(0)  # Remove extra dimension if present
    if embeddings_np.ndim == 1:  # Single query embedding
        embeddings_np = embeddings_np.reshape(1, -1)  # Reshape to (1, n_features)
    
    reduced_embeddings = pca.fit_transform(embeddings_np)

    # Convert back to torch tensor
    return torch.tensor(reduced_embeddings)



# Serve dataset images
@app.route('/coco_images_resized/<filename>')
def dataset_images(filename):
    dataset_folder = '/Users/rebeccageisberg/Desktop/cs 506/assignments/assignment-10/rgeis26-assignment-10/coco_images_resized/'  # Update with your dataset folder
    return send_from_directory(dataset_folder, filename)

# Home page and search functionality
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get query type, weight (lambda), and text query
        query_type = request.form.get("query_type")
        text_query = request.form.get("text_query")
        lam = float(request.form.get("lam", 0.8))
        use_pca = request.form.get("use_pca") == "true"
        pca_components = int(request.form.get("pca_components", 50)) # use k 
        pca_components = min(pca_components, dataset_embeddings.shape[0], dataset_embeddings.shape[1])

        # Handle image upload
        image_path = None
        if "image_query" in request.files:
            uploaded_image = request.files["image_query"]
            if uploaded_image.filename != "":
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image.filename)
                uploaded_image.save(image_path)

        # Calculate query embedding
        if query_type == "Hybrid query" and image_path and text_query:
            # Process both image and text
            image = preprocess(Image.open(image_path)).unsqueeze(0)
            image_query = F.normalize(model.encode_image(image))
            text_tokens = tokenizer([text_query])
            text_query_embedding = F.normalize(model.encode_text(text_tokens))
            query_embedding = F.normalize(lam * text_query_embedding + (1 - lam) * image_query)
        elif query_type == "Text query" and text_query:
            # Process only text
            text_tokens = tokenizer([text_query])
            query_embedding = F.normalize(model.encode_text(text_tokens))
        elif query_type == "Image query" and image_path:
            # Process only image
            image = preprocess(Image.open(image_path)).unsqueeze(0)
            query_embedding = F.normalize(model.encode_image(image))
        else:
            return render_template("index.html", error="Invalid query type or missing inputs.")
        
        
        print("Dataset embeddings shape before PCA:", dataset_embeddings.shape)
        print("Query embedding shape before PCA:", query_embedding.shape)
        
        if use_pca:
            pca = PCA(n_components=pca_components)
            reduced_dataset_np = pca.fit_transform(dataset_embeddings.detach().cpu().numpy())
            reduced_dataset = torch.tensor(reduced_dataset_np)

            query_embedding = query_embedding.squeeze()  # Now likely [512]
            query_embedding = query_embedding.unsqueeze(0)
            query_reduced_np = pca.transform(query_embedding.detach().cpu().numpy())
            query_reduced = torch.tensor(query_reduced_np)
            query_reduced = query_reduced.squeeze(0)  # Now should be [1, 512]
            similarities = torch.matmul(reduced_dataset, query_reduced.T)

        else:
            reduced_dataset = dataset_embeddings
            query_reduced = query_embedding
            similarities = torch.matmul(reduced_dataset, query_reduced.T).squeeze(1)


        print("Reduced dataset shape:", reduced_dataset.shape)
        print("Query reduced shape:", query_reduced.shape)
        print("Query reduced.T shape:", query_reduced.T.shape)
        
        # Find the top 5 most similar images
        top_k = 5
        top_indices = torch.topk(similarities, top_k).indices.tolist() 
        dataset_image_paths = df['file_name'].reset_index(drop=True)  # Reset to ensure indices match embeddings

        print("Dataset paths length:", len(dataset_image_paths))
        print("Dataset embeddings length:", len(dataset_embeddings))
        top_similar_images = [
            (f"/coco_images_resized/{os.path.basename(dataset_image_paths.iloc[idx])}", similarities[idx].item())
            for idx in top_indices
        ]

        print("Top indices:", top_indices)
        


        # Pass results to the template
        return render_template(
            "index.html",
            query_type=query_type,
            top_similar_images=top_similar_images
        )

    return render_template("index.html")
