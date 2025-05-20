# ui\help_tab.py

import gradio as gr

def create_help_tab(tabs):
    """Create the help tab UI and attach it to the tabs container"""
    with gr.TabItem("Help & Documentation"):
        gr.Markdown("""
        # Pixel Pipeline - Image Dataset Refinement Pipeline
        
        This tool helps you refine image datasets for AI training by removing duplicates, identifying similar images,
        filtering by face count, and adding high-quality captions.
        
        ## Recommended Workflow
        
        For best results, follow this sequence:
        
        1. **Image Similarity Tab**: First, remove duplicates and overly similar images
        2. **Face Detection Tab**: Next, filter images based on face count (for character LoRA training)
        3. **Image Captioning Tab**: Finally, add AI-generated captions to your refined dataset
        
        ## Image Similarity Features
        
        - **Perceptual Hash**: Quickly identifies exact or near-duplicate images
        - **VGG16 Similarity**: Finds images that are visually similar but not exact duplicates
        - Images are automatically moved to categorized folders
        
        ## Face Detection Features
        
        - **Multiple Detection Models**: Choose from various face detection algorithms
        - **Automatic Sorting**: Separates images with no faces, single faces, and multiple faces
        - **Resource Intensive**: Run this after duplicate removal for best performance
        
        ## Captioning Features
        
        - **Qwen-VL Models**: Generate high-quality AI captions with vision-language models
        - **Custom Prompts**: Tailor caption generation to your specific needs
        - **Flux-Compatible**: Creates captions in a format ready for AI training
        
        ## Tips for Efficient Processing
        
        - Process smaller batches if you're low on VRAM/RAM
        - For face detection, RetinaFace offers good accuracy-performance balance
        - For captioning, the 3B model is faster while the 7B model provides more detailed captions
        - Use perceptual hash first, then VGG16 for similarity detection
        
        ## System Requirements
        
        - CUDA-compatible GPU recommended for all features
        - At least 6GB VRAM for optimal performance
        - 16GB+ system RAM recommended
        - Python 3.11+ and required dependencies
        """)