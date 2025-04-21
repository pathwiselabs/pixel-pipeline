# app.py

# Supress deprecation noise
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      # Silence TensorFlow INFO/WARN logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'     # Kill the oneDNN custom-ops message

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Suppress any stdout from UMAP's __init__.py
import contextlib, sys
with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
    import umap  # Supress boolean umap message

import gradio as gr
from ui.caption_tab import create_caption_tab
from ui.help_tab import create_help_tab
from ui.similarity_tab import create_similarity_tab
from ui.face_detection_tab import create_face_detection_tab
from ui.clustering_tab import create_clustering_tab
import base64


# Global CSS to fix gallery issues and improve header layout
gallery_css = """
.gradio-gallery {
    overflow-y: auto !important;
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
    z-index: 100;
}
.gradio-gallery .navigation-button {
    background-color: rgba(0,0,0,0.5);
    color: white;
    border: none;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    font-size: 24px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 10px;
}
/* Custom header styling */
#app-header {
    display: flex;
    align-items: center;
    padding: 20px;
    border-bottom: 1px solid #eaeaea;
    margin-bottom: 20px;
}
#logo-container {
    width: 120px;
    height: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
}
#logo-container img {
    max-width: 100%;
    max-height: 100%;
}
#header-text {
    margin-left: 20px;
}
#header-title {
    font-size: 32px;
    font-weight: bold;
    margin: 0;
    color: #333;
}
#header-subtitle {
    font-size: 18px;
    color: #666;
    margin: 5px 0 0 0;
}
"""

# Create resources
def get_base64_logo():
    try:
        with open("resources/logo.png", "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print("Logo file not found at resources/logo.png")
        return ""

# Create the main application
def create_app():
    with gr.Blocks(theme=gr.themes.Soft(), css=gallery_css) as demo:
        # Get the logo before creating the HTML
        logo_base64 = get_base64_logo()

        # Custom header with logo on left and title/subtitle to right
        gr.HTML(f"""
            <div id="app-header">
                <div id="logo-container">
                    <img src="data:image/png;base64,{logo_base64}" alt="Pixel Pipeline Logo" />
                </div>
                <div id="header-text">
                    <h1 id="header-title">Pixel Pipeline</h1>
                    <p id="header-subtitle">Automated Image Set Cleaning</p>
                </div>
            </div>
        """)
        
        with gr.Tabs() as tabs:
            # Complete workflow in suggested order
            similarity_tab = create_similarity_tab(tabs)        # Step 1: Remove duplicates & similar images
            face_detection_tab = create_face_detection_tab(tabs)  # Step 2: Filter by face count
            clustering_tab = create_clustering_tab(tabs)        # Step 3: Cluster and reduce dataset size
            caption_tab_info = create_caption_tab(tabs)        # Step 4: Caption the refined dataset
            create_help_tab(tabs)                              # Help & documentation
            
        # Initialize system info on load if needed
        if caption_tab_info:
            update_fn, outputs = caption_tab_info
            demo.load(fn=update_fn, inputs=None, outputs=outputs)
            
    return demo

if __name__ == "__main__":
    # Add dependency check
    missing_deps = []
    try:
        import psutil
    except ImportError:
        missing_deps.append("psutil")
        
    try:
        import imagehash
    except ImportError:
        missing_deps.append("imagehash")
        
    try:
        import scipy
    except ImportError:
        missing_deps.append("scipy")
        
    try:
        import deepface
    except ImportError:
        missing_deps.append("deepface")
        
    try:
        import facenet_pytorch
    except ImportError:
        missing_deps.append("facenet-pytorch")
        
    try:
        import umap
    except ImportError:
        missing_deps.append("umap-learn")
    
    if missing_deps:
        print("Missing dependencies. Please install:")
        print(f"pip install {' '.join(missing_deps)}")
        exit(1)
    
    # Launch the app with custom port to avoid conflicts with other Gradio apps
    app = create_app()
    app.launch(
        favicon_path=r"resources\favicon-32x32.png", 
        inbrowser=True,
        server_port=7865  # Use specific port to avoid conflicts with other Gradio apps
    )