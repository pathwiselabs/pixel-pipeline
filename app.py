import gradio as gr
from ui.caption_tab import create_caption_tab
from ui.help_tab import create_help_tab
from ui.similarity_tab import create_similarity_tab  # Import the new tab

# Create the main application
def create_app():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 🖼️ Qwen-VL Batch Image Captioner with Flux-compatible IDs")
        
        with gr.Tabs() as tabs:
            caption_tab_info = create_caption_tab(tabs)
            create_similarity_tab(tabs)  # Add the new tab
            create_help_tab(tabs)
            
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
    
    if missing_deps:
        print("Missing dependencies. Please install:")
        print(f"pip install {' '.join(missing_deps)}")
        exit(1)
    
    # Launch the app
    app = create_app()
    app.launch()