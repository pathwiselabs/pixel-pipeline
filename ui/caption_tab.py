# ui\caption_tab.py

import gradio as gr
from image_processor import ImageProcessor, MODEL_OPTIONS

# Create a singleton processor instance
processor = ImageProcessor()

def update_system_info():
    """Update system info for display"""
    system_info = processor.check_cuda_availability()
    return (
        system_info["status"],
        system_info["total_vram"],
        system_info["used_vram"],
        system_info["free_vram"]
    )

def caption_images(files, prompt, identifier, use_hash, model_key, progress=gr.Progress()):
    """Function for the Generate Captions button"""
    processed_data = processor.process_images(
        files, prompt, identifier, use_hash, model_key, progress, use_cache=False
    )
    
    return (
        processed_data['gallery_data'], 
        processed_data['captions_list'],
        "Captions generated successfully! Click an image to view its detailed caption.",
        ""  # Reset the selected caption
    )

def captions_zip(files, prompt, identifier, use_hash, model_key, progress=gr.Progress()):
    """Function for the Download Captions ZIP button"""
    processed_data = processor.process_images(
        files, prompt, identifier, use_hash, model_key, progress, use_cache=True
    )
    
    progress(0.9, desc="Creating ZIP file...")
    zip_path = processor.create_captions_zip(processed_data)
    progress(1.0, desc="ZIP file created!")
    
    return str(zip_path), "Captions ZIP file created and ready for download!"

def show_selected_caption(evt: gr.SelectData, caption_list):
    """Display the caption for the selected image"""
    if caption_list and 0 <= evt.index < len(caption_list):
        return caption_list[evt.index]
    return ""

def clear_cache():
    """Clear the caption cache"""
    return processor.clear_cache()

def create_caption_tab(tabs):
    """Create the caption tab UI and attach it to the tabs container"""
    with gr.TabItem("4. Image Captioning"):

        gr.Markdown("""
                    ## 🏷️ Image Captioning

                    **Purpose:** Generate high-quality AI descriptions for your refined images.

                    **When to use:** This final step prepares your cleaned dataset for training with detailed natural language captions.
                    """)
        
        with gr.Accordion("ℹ️ How Image Captioning Works", open=False):
            gr.Markdown("""
            ### How Image Captioning Works
            
            This tool uses Qwen2.5-VL vision-language models to analyze your images and generate descriptive captions:
            
            1. The model processes each image and interprets its visual content
            2. Based on your custom prompt, it generates natural language descriptions
            3. Captions are prefixed with your identifier (full name or hashed format)
            4. Results can be downloaded as a ZIP file containing individual text files for each image
            
            **Model Options:**
            - **Qwen2.5-VL-3B-Instruct**: Faster processing, smaller memory footprint (~8GB)
            - **Qwen2.5-VL-7B-Instruct**: More detailed and accurate captions, requires more VRAM (16GB+)
            
            **Prompt Tips:**
            - Be specific about what aspects you want described (e.g., "describe clothing, expression, pose")
            - Shorter prompts often yield more concise and focused descriptions
            - The default prompt is optimized for character descriptions but can be customized
            
            **Hash ID Option:**
            - When enabled, converts "John Smith" to something like "JS_ab123"
            - Recommended for better compatibility with training formats
            """)

        captions_store = gr.State([])  # Store for captions to be used for selection
        
        with gr.Row():
            # System info and model selection column
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_OPTIONS.keys()),
                    value=list(MODEL_OPTIONS.keys())[0],
                    label="Select Vision-Language Model",
                    info="Choose which model to use for image captioning"
                )
                
                with gr.Row():
                    cuda_status = gr.Textbox(label="CUDA Status", value="Checking...", interactive=False)
                
                with gr.Row():
                    total_vram = gr.Textbox(label="Total VRAM/RAM", value="Checking...", interactive=False)
                    used_vram = gr.Textbox(label="Used VRAM/RAM", value="Checking...", interactive=False)
                
                with gr.Row():
                    free_vram = gr.Textbox(label="Free VRAM/RAM", value="Checking...", interactive=False)
                
                with gr.Row():
                    refresh_btn = gr.Button("Refresh System Info", size="sm")
                    clear_cache_btn = gr.Button("Clear Results Cache", size="sm")
                
                gr.Markdown("---")
                
                # Input controls
                files = gr.File(label="Upload Images", file_types=["image"], file_count="multiple")
                
                with gr.Row():
                    prompt = gr.Textbox(
                        label="LLM Caption Prompt", 
                        placeholder="Briefly and definitively describe the person in the photograph",
                        value="Briefly and definitively describe the person in the photograph, using the format: gender, emotion, clothing, scenery, lighting",
                        lines=3
                    )
                
                with gr.Row():
                    identifier = gr.Textbox(
                        label="Identifier or Name", 
                        placeholder="E.g: Laura Rose or Jonathan Doe",
                        info="Used as prefix for all captions"
                    )
                    use_hash = gr.Checkbox(
                        label="Use Hash ID", 
                        value=True,
                        info="Convert name to initials + hash (e.g. JD_1a2b3)"
                    )
                
                with gr.Row():
                    btn_generate = gr.Button("Generate Captions", variant="primary")
                    btn_zip = gr.Button("Download Captions ZIP")
                
                file_zip = gr.File(label="Generated Captions ZIP")
                status = gr.Textbox(label="Status", value="Ready to process images...", interactive=False)
        
            # Gallery display column
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="🖼️ Image Previews", 
                    columns=3, 
                    height=600,
                    preview=True,
                    elem_id="caption_gallery",
                    allow_preview=True,
                    show_download_button=True,
                    object_fit="contain"  # Ensures images fit properly
                )

                # Add custom CSS to ensure scrolling works
                custom_css = """
                #caption_gallery {
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

                # Caption section under the gallery
                selected_caption = gr.Textbox(
                    label="Selected Image Caption", 
                    placeholder="Click on an image above to view its caption",
                    lines=4,
                    interactive=False
                )
        
        # Set up event handlers
        refresh_btn.click(
            fn=update_system_info,
            inputs=None,
            outputs=[cuda_status, total_vram, used_vram, free_vram]
        )
        
        clear_cache_btn.click(
            fn=clear_cache,
            inputs=None,
            outputs=status
        )

        btn_generate.click(
            fn=caption_images, 
            inputs=[files, prompt, identifier, use_hash, model_dropdown], 
            outputs=[gallery, captions_store, status, selected_caption],
            show_progress="full"
        ).then(
            fn=update_system_info,
            inputs=None,
            outputs=[cuda_status, total_vram, used_vram, free_vram]
        )
        
        btn_zip.click(
            fn=captions_zip, 
            inputs=[files, prompt, identifier, use_hash, model_dropdown], 
            outputs=[file_zip, status],
            show_progress="full"
        ).then(
            fn=update_system_info,
            inputs=None,
            outputs=[cuda_status, total_vram, used_vram, free_vram]
        )
        
        gallery.select(
            fn=show_selected_caption,
            inputs=[captions_store],
            outputs=selected_caption
        )
        
        # Initialize system info on load
        # We'll need to access the parent demo object for this in the main app
        return [update_system_info, [cuda_status, total_vram, used_vram, free_vram]]