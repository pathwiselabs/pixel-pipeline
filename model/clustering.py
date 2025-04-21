#model/clustering.py
# A model to implement emedding and clustering for facial images
#The workflow is now:
#
#Detect and align faces using MTCNN
#Generate 512-dimensional face embeddings using FaceNet
#Cluster these embeddings using k-means
#Select representative images from each cluster

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import os
from sklearn.cluster import KMeans
import numpy as np
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import umap
print(umap.__file__)
print(hasattr(umap, "UMAP"))


from matplotlib.gridspec import GridSpec
import json

class FaceImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        # Initialize face detection
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=0,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        # Get face tensor
        face_tensor = self.mtcnn(image)
        return face_tensor, image_path

def extract_face_embeddings(model, dataloader, device):
    embeddings = []
    valid_paths = []
    model.eval()
    
    with torch.no_grad():
        for batch, image_paths in tqdm(dataloader, desc="Extracting face embeddings"):
            # Filter out None values (where faces weren't detected)
            valid_batch = [tensor for tensor in batch if tensor is not None]
            valid_batch_paths = [path for tensor, path in zip(batch, image_paths) if tensor is not None]
            
            if not valid_batch:
                continue
                
            # Stack valid tensors
            batch_tensor = torch.stack(valid_batch).to(device)
            output = model(batch_tensor)
            embeddings.append(output.cpu().numpy())
            valid_paths.extend(valid_batch_paths)
            
    return np.vstack(embeddings) if embeddings else np.array([]), valid_paths

def generate_interactive_visualization(embeddings, cluster_labels, selected_indices, image_paths, output_dir):
    """
    Generate an interactive HTML visualization of the clustering results.
    """
    # Create UMAP embedding
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    # Read the HTML template
    template_path = 'cluster_visualization_template.html'
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"HTML template not found at {template_path}")

    with open(template_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Ensure selected_indices is a list of Python ints
    selected_indices = selected_indices.tolist() if isinstance(selected_indices, np.ndarray) else selected_indices
    selected_indices = [int(idx) for idx in selected_indices]

    # Prepare the data for the plot with explicit type conversion
    plot_data = {
        'x': embedding_2d[:, 0].tolist(),
        'y': embedding_2d[:, 1].tolist(),
        'clusters': [int(label) for label in cluster_labels],  # Convert to native Python int
        'selectedIndices': [int(idx) for idx in selected_indices]  # Ensure indices are Python ints
    }

    image_data = {
        'paths': [os.path.join('images', os.path.basename(image_paths[i])).replace('\\', '/') for i in selected_indices],
        'clusters': [int(cluster_labels[i]) for i in selected_indices]  # Convert to native Python int
    }

    # Serialize JSON safely
    print("Plot Data Sample:", plot_data['x'][:5], plot_data['y'][:5])
    print("Image Data Sample:", image_data['paths'][:5], image_data['clusters'][:5])
    plot_data_json = json.dumps(plot_data)
    image_data_json = json.dumps(image_data)

    # Replace unique placeholders in the HTML template
    html_content = html_content.replace(
        '__PLOT_DATA__',
        plot_data_json
    ).replace(
        '__IMAGE_DATA__',
        image_data_json
    )

    # Save the final HTML file
    output_path = os.path.join(output_dir, 'cluster_visualization.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Visualization saved to {output_path}")

def cluster_and_select_images(input_dir, output_dir, n_clusters=50, batch_size=32):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained FaceNet model
    model = InceptionResnetV1(pretrained='vggface2').to(device)
    model.eval()

    # Create dataset and dataloader
    dataset = FaceImageDataset(input_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: list(zip(*x))  # Custom collate to handle None values
    )

    # Extract embeddings
    print("Extracting face embeddings from images...")
    embeddings, valid_image_paths = extract_face_embeddings(model, dataloader, device)
    
    if len(embeddings) == 0:
        raise ValueError("No valid face embeddings were extracted from the images")
    
    # Adjust n_clusters if we have fewer images than requested clusters
    n_clusters = min(n_clusters, len(embeddings))

    # Perform k-means clustering
    print(f"Performing k-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Select representative images from each cluster
    selected_images = []
    selected_indices = []
    for cluster_idx in range(n_clusters):
        # Get indices of images in this cluster
        cluster_indices = np.where(cluster_labels == cluster_idx)[0]
        
        if len(cluster_indices) > 0:
            # Find the image closest to cluster center
            cluster_embeddings = embeddings[cluster_indices]
            center_distances = np.linalg.norm(
                cluster_embeddings - kmeans.cluster_centers_[cluster_idx], axis=1
            )
            representative_idx = cluster_indices[np.argmin(center_distances)]
            selected_images.append(valid_image_paths[representative_idx])
            selected_indices.append(representative_idx)

    # Generate interactive visualization
    print("Generating interactive visualization...")
    generate_interactive_visualization(embeddings, cluster_labels, selected_indices, valid_image_paths, output_dir)

    # Copy selected images to output directory
    print(f"Copying {len(selected_images)} selected images to output directory...")
    for img_path in selected_images:
        shutil.copy2(img_path, output_dir)

    return selected_images, cluster_labels, embeddings