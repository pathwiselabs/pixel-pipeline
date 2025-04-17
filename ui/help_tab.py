# ui\help_tab.py

import gradio as gr

def create_help_tab(tabs):
    """Create the help tab UI and attach it to the tabs container"""
    with gr.TabItem("Help & Documentation"):
        gr.Markdown("""
        # How to Use the Image Captioner
        
        This tool helps you create captions for multiple images using AI vision models.
        
        ## Quick Start Guide
        
        1. **Upload Images**: Click 'Upload Images' and select multiple image files
        2. **Set Options**:
           - Choose a model (3B is faster, 7B is more detailed)
           - Enter a prompt to guide the captioning
           - Enter an identifier/name for the person/subject
           - Check 'Use Hash ID' to create Flux-compatible IDs
        3. **Generate Captions**: Click the button to process your images
        4. **Review Results**: Click on images in the gallery to see their captions
        5. **Download**: Click 'Download Captions ZIP' to get text files for each image
        
        ## Tips
        
        - For better people descriptions, use prompts that specify the details you want
        - Clear the cache if you're seeing unexpected results
        - Check system info to monitor resource usage
        """)