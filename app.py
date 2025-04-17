# app.py

import gradio as gr
from ui.caption_tab import create_caption_tab
from ui.help_tab import create_help_tab
from ui.similarity_tab import create_similarity_tab
from ui.face_detection_tab import create_face_detection_tab
from ui.clustering_tab import create_clustering_tab

# Global CSS to fix gallery issues
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
"""

# Create the main application
def create_app():
    with gr.Blocks(theme=gr.themes.Soft(), css=gallery_css) as demo:
        gr.Markdown("## 🖼️ Pixel Pipeline - An Image Dataset Refinement Pipeline")
        
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
    
    # Launch the app
    app = create_app()
    app.launch()