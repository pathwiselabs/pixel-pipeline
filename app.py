import gradio as gr
from image_processor import ImageProcessor, MODEL_OPTIONS

# Create the image processor instance
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
    # Use the core processing function with status callback
    processed_data = processor.process_images(
        files, prompt, identifier, use_hash, model_key, progress
    )
    
    # Return the required outputs for the gallery view
    # Also return empty string for selected_caption to reset it
    return (
        processed_data['gallery_data'], 
        processed_data['captions_list'],
        "Captions generated successfully! Click an image to view its detailed caption.",
        ""  # Reset the selected caption
    )

def captions_zip(files, prompt, identifier, use_hash, model_key, progress=gr.Progress()):
    """Function for the Download Captions ZIP button"""
    # Use the core processing function (will use cache if available)
    processed_data = processor.process_images(
        files, prompt, identifier, use_hash, model_key, progress
    )
    
    # Create ZIP file from the cached captions
    progress(0.9, desc="Creating ZIP file...")
    zip_path = processor.create_captions_zip(processed_data)
    progress(1.0, desc="ZIP file created!")
    
    # Return the zip file path and status message
    return str(zip_path), "Captions ZIP file created and ready for download!"

def show_selected_caption(evt: gr.SelectData, caption_list):
    """Display the caption for the selected image"""
    if caption_list and 0 <= evt.index < len(caption_list):
        return caption_list[evt.index]
    return ""

def clear_cache():
    """Clear the caption cache"""
    return processor.clear_cache()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🖼️ Qwen-VL Batch Image Captioner with Flux-compatible IDs")
    
    captions_store = gr.State([])  # Store for captions to be used for selection
    
    with gr.Row():
        # System info and model selection
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
                elem_id="gallery"
            )
            
            # Caption section under the gallery
            selected_caption = gr.Textbox(
                label="Selected Image Caption", 
                placeholder="Click on an image above to view its caption",
                lines=4,
                interactive=False
            )

    # Initialize system info on load
    demo.load(
        fn=update_system_info,
        inputs=None,
        outputs=[cuda_status, total_vram, used_vram, free_vram]
    )
    
    # Refresh system info button
    refresh_btn.click(
        fn=update_system_info,
        inputs=None,
        outputs=[cuda_status, total_vram, used_vram, free_vram]
    )
    
    # Clear cache button
    clear_cache_btn.click(
        fn=clear_cache,
        inputs=None,
        outputs=status
    )

    # Generate captions button
    btn_generate.click(
        fn=caption_images, 
        inputs=[files, prompt, identifier, use_hash, model_dropdown], 
        outputs=[gallery, captions_store, status, selected_caption],  # Add selected_caption to outputs
        show_progress="full"
    ).then(
        fn=update_system_info,
        inputs=None,
        outputs=[cuda_status, total_vram, used_vram, free_vram]
    )
    
    # Download ZIP button
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
    
    # Add click handler to show caption when image is clicked
    gallery.select(
        fn=show_selected_caption,
        inputs=[captions_store],
        outputs=selected_caption
    )

if __name__ == "__main__":
    # Add dependency check
    missing_deps = []
    try:
        import psutil
    except ImportError:
        missing_deps.append("psutil")
    
    if missing_deps:
        print("Missing dependencies. Please install:")
        print(f"pip install {' '.join(missing_deps)}")
        exit(1)
        
    demo.launch()