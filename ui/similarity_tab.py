# ui\similarity_tab.py

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
    vgg_model = VGG16SimilarityModel()  # Load VGG model directly
    
    def process_directory(input_dir, output_dir, methods, similarity_threshold=0.85, progress=gr.Progress()):
        """Process images in a directory to find similar/duplicate images and move them to separate folders"""
        # Check if output directory exists, create if not
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize results and stats
        gallery_results = []
        stats = {}
        moved_images = []
        
        progress(0, desc="Starting image similarity analysis...")
        
        # Get total image count
        total_images = len([f for f in os.listdir(input_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Create progress callback
        def progress_callback(current, total, method_name=""):
            progress(current / total, desc=f"Processing image {current}/{total} with {method_name}")
        
        # Process with perceptual hash if selected
        if "perceptual_hash" in methods:
            hash_output_dir = os.path.join(output_dir, "duplicate_images")
            
            progress(0.1, desc="Running perceptual hash analysis...")
            
            # Process images with hash model - it will move duplicates to output dir
            duplicate_count = hash_model.process_images(
                input_dir, 
                hash_output_dir, 
                lambda current, total: progress_callback(current, total, "perceptual hash")
            )
            
            # Collect results for gallery if duplicates were found
            if duplicate_count > 0 and os.path.exists(hash_output_dir):
                for filename in os.listdir(hash_output_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        filepath = os.path.join(hash_output_dir, filename)
                        thumb = Image.open(filepath)
                        thumb = thumb.copy()  # Create a copy to avoid resource issues
                        gallery_results.append([thumb, f"Duplicate: {filename}"])
                        moved_images.append(filename)
            
            stats["perceptual_hash"] = {
                "duplicates_found": duplicate_count,
                "total_images": total_images
            }
            
        # Process with VGG16 if selected
        if "vgg16_similarity" in methods:
            vgg_output_dir = os.path.join(output_dir, "similar_images")
            
            progress(0.5, desc="Running VGG16 similarity analysis...")
            
            # Process images with VGG16 model - it will move only one image from each similar pair to output dir
            similar_count = vgg_model.process_images(
                input_dir, 
                vgg_output_dir, 
                similarity_threshold, 
                lambda current, total: progress_callback(current, total, "VGG16 similarity")
            )
            
            # Collect similar pairs for gallery
            pairs_dir = os.path.join(vgg_output_dir, "similar_pairs")
            if similar_count > 0 and os.path.exists(pairs_dir):
                for filename in os.listdir(pairs_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        filepath = os.path.join(pairs_dir, filename)
                        thumb = Image.open(filepath)
                        thumb = thumb.copy()  # Create a copy to avoid resource issues
                        gallery_results.append([thumb, f"Similar pair: {filename}"])
            
            # Add moved similar images to the list
            if os.path.exists(vgg_output_dir):
                for filename in os.listdir(vgg_output_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and filename not in moved_images:
                        moved_images.append(filename)
            
            stats["vgg16_similarity"] = {
                "similar_pairs": similar_count,
                "total_images": total_images
            }
        
        # Generate status message based on results
        status_msg = []
        
        if "perceptual_hash" in methods:
            hash_stats = stats.get("perceptual_hash", {})
            dup_count = hash_stats.get("duplicates_found", 0)
            if dup_count > 0:
                status_msg.append(f"Found and moved {dup_count} duplicate images to {os.path.join(output_dir, 'duplicate_images')}")
            else:
                status_msg.append("No duplicate images found with perceptual hash")
                
        if "vgg16_similarity" in methods:
            vgg_stats = stats.get("vgg16_similarity", {})
            sim_count = vgg_stats.get("similar_pairs", 0)
            if sim_count > 0:
                status_msg.append(f"Found {sim_count} similar image pairs with VGG16 (threshold: {similarity_threshold})")
                status_msg.append(f"One image from each pair moved to {os.path.join(output_dir, 'similar_images')}")
                status_msg.append(f"Comparison images created in {os.path.join(output_dir, 'similar_images', 'similar_pairs')}")
            else:
                status_msg.append(f"No similar images found with VGG16 (threshold: {similarity_threshold})")
        
        status_message = "\n".join(status_msg)
        
        # Final cleanup
        progress(1.0, desc="Completed image processing")
        
        # If we have no results to display in gallery
        if not gallery_results:
            return [], status_message
            
        return gallery_results, status_message
    
    with gr.TabItem("1. Image Similarity"):
        gr.Markdown("""
                    ## 🔍 Image Similarity & Duplicate Detection
                    **Purpose:** Remove duplicates and/or visually similar images to improve dataset quality.

                    **When to use:** As a first step in the dataset refinement process.
                    """)
        
        with gr.Accordion("ℹ️ How Similarity Detection Works", open=False):
            gr.Markdown("""
            ### How Similarity Detection Works
            
            This tool offers two complementary methods for finding and removing redundant images:
            
            **Perceptual Hash:**
            - Generates a unique "fingerprint" for each image based on its visual content
            - Fast and efficient for identifying exact or near-duplicate images
            - Great for finding resized, compressed, or slightly modified versions of the same image
            
            **VGG16 Similarity:**
            - Uses deep learning to identify images that are visually similar but not identical
            - Can find different photos of the same scene, subject, or composition
            - More resource-intensive but provides more sophisticated analysis
            - The similarity threshold controls how closely matched images need to be (higher = more similar)
            
            **Output Organization:**
            - Duplicate images are moved to a "duplicate_images" folder
            - One image from each similar pair is moved to a "similar_images" folder
            - Comparison images showing similar pairs are created for your reference
            
            **Recommended Workflow:**
            1. First run with perceptual hash to remove exact duplicates
            2. Then run with VGG16 similarity to identify and remove visually similar images
            3. Start with a higher threshold (0.85-0.90) and adjust as needed
            """)

        
        with gr.Row():
            # Input controls column
            with gr.Column(scale=1):
                input_dir = gr.Textbox(
                    label="Input Directory Path",
                    placeholder="Enter the full path to your image directory",
                    info="Directory containing images to analyze"
                )
                
                output_dir = gr.Textbox(
                    label="Output Directory Path",
                    placeholder="Enter the full path to your output directory",
                    info="Directory where duplicate/similar images will be moved to"
                )
                
                methods = gr.CheckboxGroup(
                    choices=["perceptual_hash", "vgg16_similarity"],
                    value=["perceptual_hash"],
                    label="Detection Methods",
                    info="Select one or both methods. Perceptual hash is faster but only finds near-duplicates. VGG16 can find visually similar images."
                )
                
                with gr.Row(visible=lambda methods: "vgg16_similarity" in methods):
                    similarity_threshold = gr.Slider(
                        minimum=0.5,
                        maximum=0.99,
                        value=0.85,
                        step=0.01,
                        label="Similarity Threshold",
                        info="Higher values require images to be more similar (0.5-0.99)"
                    )
                
                with gr.Row():
                    btn_process = gr.Button("Process Directory", variant="primary")
                
                status = gr.Textbox(
                    label="Status", 
                    value="Ready to analyze images...", 
                    interactive=False,
                    lines=5
                )
                
            # Results gallery column
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="🖼️ Detected Duplicate/Similar Images", 
                    columns=2,
                    height=600,
                    preview=True,
                    elem_id="similarity_gallery",
                    allow_preview=True,
                    show_download_button=True,
                    object_fit="contain"  # Ensures images fit properly
                )

            # Add custom CSS to ensure scrolling works
            custom_css = """
            #similarity_gallery {
                overflow-y: auto !important;
                max-height: 600px;
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

            # Add this after creating the gallery component:
            gr.HTML(f"<style>{custom_css}</style>")
    
        # Show/hide threshold slider based on method selection
        methods.change(
            fn=lambda m: gr.update(visible=("vgg16_similarity" in m)),
            inputs=methods,
            outputs=similarity_threshold
        )
        
        # Process button
        btn_process.click(
            fn=process_directory,
            inputs=[input_dir, output_dir, methods, similarity_threshold],
            outputs=[gallery, status],
            show_progress="full"
        )