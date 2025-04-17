# ui\face_detection_tab.py

import gradio as gr
import os
from model.face_detection_model import FaceDetectionModel
from PIL import Image

def create_face_detection_tab(tabs):
    """Create the face detection tab UI and attach it to the tabs container"""
    
    face_model = FaceDetectionModel()
    
    def process_directory(input_dir, output_dir, detector_backend, align, progress=gr.Progress()):
        """Process images in a directory to detect and categorize faces"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create subdirectories
        no_faces_dir = os.path.join(output_dir, "no_faces")
        multiple_faces_dir = os.path.join(output_dir, "multiple_faces")
        
        if not os.path.exists(no_faces_dir):
            os.makedirs(no_faces_dir)
        if not os.path.exists(multiple_faces_dir):
            os.makedirs(multiple_faces_dir)
            
        # Initialize gallery results
        gallery_results = []
        
        progress(0, desc="Starting face detection...")
        
        def progress_callback(current, total):
            progress(current / total, desc=f"Processing image {current}/{total}")
        
        # Process images with face detection model
        no_face_count, multiple_faces_count, total_images = face_model.process_images(
            input_dir,
            output_dir,
            detector_backend,
            align,
            progress_callback
        )
        
        # Collect results for gallery display
        if no_face_count > 0 and os.path.exists(no_faces_dir):
            for filename in os.listdir(no_faces_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepath = os.path.join(no_faces_dir, filename)
                    try:
                        thumb = Image.open(filepath)
                        thumb = thumb.copy()  # Create a copy to avoid resource issues
                        gallery_results.append([thumb, f"No face: {filename}"])
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
                        
        if multiple_faces_count > 0 and os.path.exists(multiple_faces_dir):
            for filename in os.listdir(multiple_faces_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepath = os.path.join(multiple_faces_dir, filename)
                    try:
                        thumb = Image.open(filepath)
                        thumb = thumb.copy()  # Create a copy to avoid resource issues
                        gallery_results.append([thumb, f"Multiple faces: {filename}"])
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
        
        # Generate status message
        status_message = (
            f"Processed {total_images} images:\n"
            f"• {no_face_count} images with no faces moved to {no_faces_dir}\n"
            f"• {multiple_faces_count} images with multiple faces moved to {multiple_faces_dir}\n"
            f"• {total_images - no_face_count - multiple_faces_count} images with single faces remained in the input directory"
        )
        
        # If we have no results to display in gallery
        if not gallery_results:
            return [], status_message
            
        return gallery_results, status_message
    
    with gr.TabItem("Face Detection"):
        gr.Markdown("## 👤 Face Detection & Sorting")
        
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
                    info="Directory where categorized images will be moved to"
                )
                
                detector_backend = gr.Dropdown(
                    choices=face_model.detector_backends,
                    value="retinaface",
                    label="Face Detector Backend",
                    info="Select which face detection algorithm to use. RetinaFace offers good accuracy-performance balance."
                )
                
                align = gr.Checkbox(
                    value=True,
                    label="Align Faces",
                    info="Enable face alignment during detection (recommended)"
                )
                
                with gr.Row():
                    btn_process = gr.Button("Process Directory", variant="primary")
                
                status = gr.Textbox(
                    label="Status", 
                    value="Ready to detect faces...", 
                    interactive=False,
                    lines=5
                )
                
                gr.Markdown("""
                ### Important Note
                
                Face detection is resource-intensive. For best results:
                
                1. Use after duplicate/similarity removal
                2. Choose RetinaFace for accuracy, MTCNN for speed
                3. Processing large datasets may take time
                
                Images will be organized into:
                - No faces - Images with no detected faces
                - Multiple faces - Images with more than one face
                - Input directory - Only single-face images remain
                """)
                
            # Results gallery column
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="🖼️ Categorized Face Images", 
                    columns=2,
                    height=600,
                    preview=True,
                    elem_id="face_gallery",
                    allow_preview=True,
                    show_download_button=True,
                    object_fit="contain"
                )
                
                # Add custom CSS for gallery scrolling
                custom_css = """
                #face_gallery {
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
                gr.HTML(f"<style>{custom_css}</style>")
        
        # Process button
        btn_process.click(
            fn=process_directory,
            inputs=[input_dir, output_dir, detector_backend, align],
            outputs=[gallery, status],
            show_progress="full"
        )