# ui\similarity.py

import gradio as gr
import os
import tempfile
import shutil
from PIL import Image
import numpy as np
from model.image_hash_model import ImageHashModel
from model.vgg16_similarity_model import VGG16SimilarityModel

def create_similarity_tab(tabs):
    """Create the image similarity detection tab UI and attach it to the tabs container"""
    
    hash_model = ImageHashModel()
    vgg_model = None  # Initialize later on demand to save memory
    
    def init_vgg_model():
        """Initialize the VGG16 model on demand"""
        nonlocal vgg_model
        if vgg_model is None:
            vgg_model = VGG16SimilarityModel()
        return "VGG16 model loaded successfully!"
    
    def process_directory(input_dir, method, similarity_threshold=0.85, progress=gr.Progress()):
        """Process images in a directory to find similar/duplicate images"""
        # Create temporary output directory
        temp_output_dir = tempfile.mkdtemp()
        
        progress(0, desc="Starting image similarity analysis...")
        
        def progress_callback(current, total):
            progress(current / total, desc=f"Processing image {current}/{total}")
        
        results = []
        stats = {}
        
        if method == "perceptual_hash":
            # Process using image hash
            duplicate_count = hash_model.process_images(
                input_dir, temp_output_dir, progress_callback
            )
            
            # Collect results
            if duplicate_count > 0:
                results = [[Image.open(os.path.join(temp_output_dir, f)), f] 
                          for f in os.listdir(temp_output_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
            stats = {
                "duplicates_found": duplicate_count,
                "total_images": len([f for f in os.listdir(input_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            }
            
            if duplicate_count == 0:
                return [], f"No duplicate images found in {stats['total_images']} images."
            else:
                return (results, 
                       f"Found {duplicate_count} duplicate images out of {stats['total_images']} total images.")
            
        elif method == "vgg16_similarity":
            # Make sure VGG16 model is loaded
            if vgg_model is None:
                progress(0.1, desc="Loading VGG16 model...")
                init_vgg_model()
                
            # Process using VGG16
            similar_count = vgg_model.process_images(
                input_dir, temp_output_dir, similarity_threshold, progress_callback
            )
            
            # Collect similar pairs
            pairs_dir = os.path.join(temp_output_dir, "similar_pairs")
            if os.path.exists(pairs_dir) and similar_count > 0:
                results = [[Image.open(os.path.join(pairs_dir, f)), f] 
                          for f in os.listdir(pairs_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
            stats = {
                "similar_pairs": similar_count,
                "total_images": len([f for f in os.listdir(input_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            }
            
            if similar_count == 0:
                return [], f"No similar images found in {stats['total_images']} images (threshold: {similarity_threshold})."
            else:
                return (results, 
                       f"Found {similar_count} similar image pairs out of {stats['total_images']} total images (threshold: {similarity_threshold}).")
    
    def create_zip_file(input_dir, method, similarity_threshold=0.85, progress=gr.Progress()):
        """Create a ZIP file of similar/duplicate images"""
        progress(0, desc="Starting image similarity analysis...")
        
        # Create temporary output directory
        temp_output_dir = tempfile.mkdtemp()
        
        def progress_callback(current, total):
            progress(current / total, desc=f"Processing image {current}/{total}")
        
        # Process the images
        if method == "perceptual_hash":
            hash_model.process_images(input_dir, temp_output_dir, progress_callback)
        elif method == "vgg16_similarity":
            if vgg_model is None:
                progress(0.1, desc="Loading VGG16 model...")
                init_vgg_model()
            vgg_model.process_images(input_dir, temp_output_dir, similarity_threshold, progress_callback)
        
        # Create ZIP file
        progress(0.9, desc="Creating ZIP file...")
        zip_path = os.path.join(tempfile.gettempdir(), f"similarity_results_{method}.zip")
        
        # Add similar_pairs directory to ZIP if it exists
        if method == "vgg16_similarity" and os.path.exists(os.path.join(temp_output_dir, "similar_pairs")):
            shutil.make_archive(zip_path[:-4], 'zip', os.path.join(temp_output_dir, "similar_pairs"))
        else:
            shutil.make_archive(zip_path[:-4], 'zip', temp_output_dir)
        
        progress(1.0, desc="ZIP file created!")
        
        return zip_path, "Similarity analysis results ZIP file created and ready for download!"
            
    with gr.TabItem("Image Similarity"):
        gr.Markdown("## 🔍 Image Similarity & Duplicate Detection")
        
        with gr.Row():
            # Input controls column
            with gr.Column(scale=1):
                input_dir = gr.Textbox(
                    label="Input Directory Path",
                    placeholder="Enter the full path to your image directory",
                    info="Directory containing images to analyze"
                )
                
                method = gr.Radio(
                    choices=["perceptual_hash", "vgg16_similarity"],
                    value="perceptual_hash",
                    label="Detection Method",
                    info="Perceptual hash is faster but only finds near-duplicates. VGG16 can find visually similar images."
                )
                
                with gr.Row(visible=lambda: method == "vgg16_similarity"):
                    similarity_threshold = gr.Slider(
                        minimum=0.5,
                        maximum=0.99,
                        value=0.85,
                        step=0.01,
                        label="Similarity Threshold",
                        info="Higher values require images to be more similar (0.5-0.99)"
                    )
                
                with gr.Row():
                    btn_analyze = gr.Button("Analyze Images", variant="primary")
                    btn_download = gr.Button("Download Results as ZIP")
                
                with gr.Row():
                    load_vgg_btn = gr.Button("Load VGG16 Model", visible=lambda: method == "vgg16_similarity")
                
                result_file = gr.File(label="Results ZIP File")
                status = gr.Textbox(label="Status", value="Ready to analyze images...", interactive=False)
                
                vgg_status = gr.Textbox(
                    label="VGG16 Model Status",
                    value="Not loaded (will load automatically when needed)",
                    visible=lambda: method == "vgg16_similarity",
                    interactive=False
                )
                
            # Results gallery column
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="🖼️ Similar/Duplicate Images", 
                    columns=2,
                    height=600,
                    preview=True,
                    elem_id="similarity_gallery"
                )
        
        # Set up event handlers
        load_vgg_btn.click(
            fn=init_vgg_model,
            inputs=None,
            outputs=vgg_status
        )
        
        # Show/hide threshold slider based on method selection
        method.change(
            fn=lambda x: gr.update(visible=(x == "vgg16_similarity")),
            inputs=method,
            outputs=similarity_threshold
        )
        
        method.change(
            fn=lambda x: gr.update(visible=(x == "vgg16_similarity")),
            inputs=method,
            outputs=load_vgg_btn
        )
        
        method.change(
            fn=lambda x: gr.update(visible=(x == "vgg16_similarity")),
            inputs=method,
            outputs=vgg_status
        )
        
        # Analyze button
        btn_analyze.click(
            fn=process_directory,
            inputs=[input_dir, method, similarity_threshold],
            outputs=[gallery, status],
            show_progress="full"
        )
        
        # Download button
        btn_download.click(
            fn=create_zip_file,
            inputs=[input_dir, method, similarity_threshold],
            outputs=[result_file, status],
            show_progress="full"
        )