# ui\clustering_tab.py

import gradio as gr
import os
import shutil
import torch
import numpy as np
from model.clustering import FaceImageDataset, extract_face_embeddings
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from PIL import Image
import tkinter as tk
from tkinter import filedialog

def create_clustering_tab(tabs):
    """Create the face clustering tab UI and attach it to the tabs container"""
    
    def select_directory(button_name):
        """Open a directory selection dialog and return the selected path"""
        # Create a hidden Tkinter root window to avoid it showing in taskbar
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)  # Ensure dialog appears on top
        
        # Choose appropriate title based on which button was clicked
        title = "Select Input Directory" if button_name == "input" else "Select Output Directory"
        folder_path = filedialog.askdirectory(title=title)
        root.destroy()
        
        # Return the selected path or None if canceled
        if folder_path:
            return folder_path
        return None
    
    def cluster_images(input_dir, output_dir, n_clusters, batch_size, clustering_method, progress=gr.Progress()):
        """Process images in a directory to cluster faces and select representative images"""
        if not input_dir or not output_dir:
            return [], "Please select both input and output directories."
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize results and stats
        gallery_results = []
        
        progress(0, desc="Starting face clustering...")
        
        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        progress(0.05, desc=f"Using device: {device}")
        
        # Create visualization directory for HTML visualization
        viz_dir = os.path.join(output_dir, "visualization")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create a specific cluster output directory based on the chosen method
        cluster_output_dir = os.path.join(output_dir, f"{clustering_method}_cluster")
        os.makedirs(cluster_output_dir, exist_ok=True)
        
        try:
            # Create dataset and dataloader
            progress(0.1, desc="Loading face detection model...")
            dataset = FaceImageDataset(input_dir)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda x: list(zip(*x))  # Custom collate to handle None values
            )
            
            # Load pre-trained FaceNet model
            progress(0.15, desc="Loading face embedding model...")
            model = InceptionResnetV1(pretrained='vggface2').to(device)
            model.eval()
            
            # Extract embeddings
            progress(0.2, desc="Extracting face embeddings from images...")
            embeddings, valid_image_paths = extract_face_embeddings(
                model, dataloader, device
            )
            
            progress(0.5, desc=f"Performing k-means clustering with {n_clusters} clusters...")
            
            # Adjust n_clusters if we have fewer images than requested clusters
            n_clusters = min(n_clusters, len(embeddings))
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Select representative images based on the clustering method
            selected_images = []
            selected_indices = []
            
            if clustering_method == "diversity":
                # Maximize diversity - select images furthest from other cluster centers
                progress(0.7, desc="Selecting diverse representative images...")
                
                for cluster_idx in range(n_clusters):
                    # Get indices of images in this cluster
                    cluster_indices = np.where(cluster_labels == cluster_idx)[0]
                    
                    if len(cluster_indices) > 0:
                        # Find the image furthest from OTHER cluster centers (to maximize diversity)
                        max_min_distance = -1
                        most_diverse_idx = -1
                        
                        for idx in cluster_indices:
                            # Calculate distances to all OTHER cluster centers
                            distances = []
                            for c_idx in range(n_clusters):
                                if c_idx != cluster_idx:
                                    dist = np.linalg.norm(
                                        embeddings[idx] - kmeans.cluster_centers_[c_idx]
                                    )
                                    distances.append(dist)
                            
                            # We want the image with the maximum minimum distance to other clusters
                            min_dist = min(distances) if distances else 0
                            if min_dist > max_min_distance:
                                max_min_distance = min_dist
                                most_diverse_idx = idx
                        
                        if most_diverse_idx != -1:
                            selected_images.append(valid_image_paths[most_diverse_idx])
                            selected_indices.append(most_diverse_idx)
            
            else:  # "consistency"
                # Maximize consistency - select images closest to their own cluster center
                progress(0.7, desc="Selecting consistent representative images...")
                
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
            
            # Create images directory inside visualization for the HTML to reference
            viz_images_dir = os.path.join(viz_dir, "images")
            os.makedirs(viz_images_dir, exist_ok=True)
            
            # Copy images for visualization only (not as the main output)
            progress(0.8, desc="Creating visualization data...")
            for img_path in selected_images:
                # Copy to visualization directory
                viz_img_path = os.path.join(viz_images_dir, os.path.basename(img_path))
                if os.path.normpath(img_path) != os.path.normpath(viz_img_path):
                    shutil.copy2(img_path, viz_img_path)
            
            # Generate the HTML visualization
            from model.clustering import generate_interactive_visualization
            generate_interactive_visualization(
                embeddings, cluster_labels, np.array(selected_indices), 
                valid_image_paths, viz_dir
            )
            
            # Copy selected images to the specific cluster output directory
            progress(0.9, desc=f"Copying {len(selected_images)} selected images to {clustering_method}_cluster directory...")
            for i, img_path in enumerate(selected_images):
                # Add cluster number to filename for better organization
                cluster_num = cluster_labels[selected_indices[i]]
                filename = os.path.basename(img_path)
                base, ext = os.path.splitext(filename)
                cluster_filename = f"{base}_cluster{cluster_num}{ext}"
                
                dest_path = os.path.join(cluster_output_dir, cluster_filename)
                # Fix for SameFileError: check if source and destination are different
                if os.path.normpath(img_path) != os.path.normpath(dest_path):
                    shutil.copy2(img_path, dest_path)
            
            # Collect results for gallery display (using the clustered images)
            for i, img_path in enumerate(selected_images):
                try:
                    thumb = Image.open(img_path)
                    thumb = thumb.copy()  # Create a copy to avoid resource issues
                    cluster_num = cluster_labels[selected_indices[i]]
                    gallery_results.append([thumb, f"Cluster {cluster_num}: {os.path.basename(img_path)}"])
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            
            # Final success message
            html_path = os.path.join(viz_dir, "cluster_visualization.html")
            status_message = (
                f"Successfully clustered images into {n_clusters} groups.\n"
                f"Selected {len(selected_images)} representative images using '{clustering_method}' method.\n"
                f"Images saved to: {cluster_output_dir}\n"
                f"Files are named with their cluster number for reference.\n"
                f"Interactive visualization available at: {html_path}"
            )
            
            progress(1.0, desc="Clustering complete!")
            
        except Exception as e:
            status_message = f"Error during clustering: {str(e)}"
            import traceback
            print(traceback.format_exc())
            return [], status_message
        
        return gallery_results, status_message
    
    with gr.TabItem("3. Image Set Refinement"):
        # First create the header with expandable info section
        gr.Markdown("""
       ## 🎯 Image Set Refinement

        **Purpose:** Automatically rezise an image dataset to a fixed amount of images while maintaining consitency or diversity.

        **When to use:** Use this step when you need to reduce dataset size.
        """)
        
        # Create an accordion for the "How it Works" section 
        with gr.Accordion("ℹ️ How Image Set Refinement Works", open=False):
            gr.Markdown("""
            This tool:
            1. Detects faces in images
            2. Generates 512-dimensional embeddings for each face
            3. Clusters similar faces together
            4. Selects representative images based on your preferred method
            
            **Clustering Methods:**
            - **Diversity**: Selects images that are most different from each other
            - **Consistency**: Selects images that are most representative of each cluster
            
            **Note**: This process requires significant computational resources. For best results:
            - Run on a dataset that's already been filtered for duplicates and face quality
            - Ensure all images contain the target character or style
            """)
        
        # Main content area
        with gr.Row():
            # Input controls column
            with gr.Column(scale=1):
                # Separate textbox and button into different components
                input_dir = gr.Textbox(
                    label="Input Directory Path",
                    placeholder="Select input directory using button below",
                    info="Directory containing processed face images to cluster"
                )
                # Button now appears below the textbox
                input_btn = gr.Button("📂 Browse Input Directory", size="sm")
                
                # Same pattern for output directory
                output_dir = gr.Textbox(
                    label="Output Directory Path",
                    placeholder="Select output directory using button below",
                    info="Directory where selected representative images will be saved"
                )
                # Button appears below the textbox
                output_btn = gr.Button("📂 Browse Output Directory", size="sm")
                
                n_clusters = gr.Slider(
                    minimum=5,
                    maximum=100,
                    value=30,
                    step=5,
                    label="Number of Images to Select",
                    info="Target number of representative images to keep"
                )
                
                clustering_method = gr.Radio(
                    choices=["diversity", "consistency"],
                    value="diversity",
                    label="Clustering Method",
                    info="Diversity: maximize differences between selected images. Consistency: select most typical images from each cluster."
                )
                
                batch_size = gr.Slider(
                    minimum=4,
                    maximum=64,
                    value=32,
                    step=4,
                    label="Batch Size",
                    info="Larger batch sizes are faster but require more VRAM"
                )
                
                btn_process = gr.Button("Cluster Images", variant="primary")
            
            # Results gallery column and status (moved below)
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="🖼️ Selected Representative Images", 
                    columns=3,
                    height=550,  # Slightly reduced height to make room for status below
                    preview=True,
                    elem_id="clustering_gallery",
                    allow_preview=True,
                    show_download_button=True,
                    object_fit="contain"
                )
                
                # Status box moved below the gallery
                status = gr.Textbox(
                    label="Status", 
                    value="Ready to cluster images...", 
                    interactive=False,
                    lines=6
                )
                
                # Add custom CSS for gallery scrolling
                custom_css = """
                #clustering_gallery {
                    overflow-y: auto !important;
                    max-height: 550px;
                }
                .gradio-gallery .thumbnail-item {
                    cursor: pointer;
                }
                .gradio-gallery .preview-image {
                    max-height: 80vh;
                    max-width: 100%;
                    object-fit: contain;
                }
                .gradio-gallery .preview-container {
                    overflow: hidden;
                    display: flex;
                    justify-content: center;
                }
                .gradio-gallery .navigation-buttons {
                    display: flex;
                    justify-content: space-between;
                    width: 100%;
                    position: absolute;
                    bottom: 10px;
                }
                """
                gr.HTML(f"<style>{custom_css}</style>")
        
        # Connect directory selection buttons
        def handle_select_input():
            path = select_directory("input")
            return path if path else gr.update()
            
        def handle_select_output():
            path = select_directory("output")
            return path if path else gr.update()
            
        input_btn.click(
            fn=handle_select_input,
            inputs=None,
            outputs=input_dir,
            api_name=False
        )
        
        output_btn.click(
            fn=handle_select_output,
            inputs=None,
            outputs=output_dir,
            api_name=False
        )
        
        # Process button
        btn_process.click(
            fn=cluster_images,
            inputs=[input_dir, output_dir, n_clusters, batch_size, clustering_method],
            outputs=[gallery, status],
            show_progress="full"
        )