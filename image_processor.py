# image_processor.py
# 
# A module for processing images using vision-language models.
# Supports Qwen2.5-VL models for image captioning and analysis.

import hashlib
import gc
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import file_utils

# Available models configuration
MODEL_OPTIONS = {
    "Qwen2.5-VL-3B-Instruct": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "description": "Smaller, faster model (3B parameters)"
    },
    "Qwen2.5-VL-7B-Instruct": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "description": "Larger, more accurate model (7B parameters)"
    }
}

class ImageProcessor:
    """
    A class for processing images with vision-language models.
    
    Handles model loading, image processing, and caption generation
    while maintaining a cache to avoid redundant processing.
    """
    
    def __init__(self):
        """Initialize the ImageProcessor with empty model and processor."""
        self.model = None
        self.processor = None
        self.current_model_key = None
        # Cache for processed captions
        self.caption_cache = {}
    
    def check_cuda_availability(self):
        """
        Check if CUDA is available and return system information.
        
        Returns:
            dict: System information including GPU/CPU status and memory details.
        """
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            used_memory = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            # Calculate free memory
            free_memory = total_memory - used_memory
            return {
                "status": f"✅ CUDA available: {device_name}",
                "total_vram": f"{total_memory:.2f} GB",
                "used_vram": f"{used_memory:.2f} GB",
                "free_vram": f"{free_memory:.2f} GB"
            }
        else:
            try:
                import psutil
                system_ram = psutil.virtual_memory().total / (1024**3)  # GB
                available_ram = psutil.virtual_memory().available / (1024**3)  # GB
                return {
                    "status": "❌ CUDA not available. Using CPU mode.",
                    "total_vram": f"System RAM: {system_ram:.2f} GB",
                    "used_vram": f"Used RAM: {psutil.virtual_memory().used / (1024**3):.2f} GB",
                    "free_vram": f"Available RAM: {available_ram:.2f} GB"
                }
            except ImportError:
                return {
                    "status": "❌ CUDA not available. Using CPU mode.",
                    "total_vram": "System RAM: N/A (psutil not installed)",
                    "used_vram": "Used RAM: N/A",
                    "free_vram": "Available RAM: N/A"
                }
    
    def initialize_model_and_processor(self, model_key):
        """
        Initialize model and processor based on selected model key.
        
        Args:
            model_key (str): Key identifying which model to load from MODEL_OPTIONS.
            
        Returns:
            tuple: The initialized model and processor objects.
        """
        # Only reload if model has changed
        if self.model is not None and model_key == self.current_model_key:
            return self.model, self.processor
        
        # Free up memory before loading a new model
        if self.model is not None:
            del self.model
            del self.processor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        # Get model configuration
        model_config = MODEL_OPTIONS[model_key]
        model_id = model_config["model_id"]
        
        # Set model token limits for images
        min_pixels = 256*28*28  # Min image token size
        max_pixels = 1280*28*28  # Max image token size
        
        # Load the new model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels
        )
        
        # Update the current model key
        self.current_model_key = model_key
        
        return self.model, self.processor
    
    def process_image_with_prompts(self, image_file, prompts):
        """
        Process a single image with given prompts.
        
        Args:
            image_file: File object of the image to process.
            prompts (list): List of text prompts to apply to the image.
            
        Returns:
            list: Generated captions for each prompt.
        """
        # Resize image if needed
        processed_image_path = file_utils.resize_image_if_needed(image_file.name)
        
        base_message = {
            "role": "user",
            "content": [{"type": "image", "image": processed_image_path}]
        }
        results = []
        for prompt in prompts:
            current_message = {
                "role": base_message["role"],
                "content": base_message["content"] + [{"type": "text", "text": prompt}]
            }

            messages = [current_message]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(self.model.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            trimmed_tokens = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            caption = self.processor.batch_decode(trimmed_tokens, skip_special_tokens=True)[0]
            results.append(caption)
        
        # Clean up temporary file if it was created
        file_utils.clean_temp_file(processed_image_path, image_file.name)
            
        return results
    
    def fluxkey(self, name, use_hash=True):
        """
        Generate a Flux-compatible unique identifier.
        
        Args:
            name (str): Base name to generate identifier from.
            use_hash (bool): Whether to include a hash in the identifier.
            
        Returns:
            str: Generated identifier.
        """
        if not use_hash:
            return name.strip()
            
        initials = ''.join([word[0].upper() for word in name.strip().split() if word.isalpha()])
        hashcode = hashlib.sha1(name.encode('utf-8')).hexdigest()[:5]
        identifier = f"{initials}_{hashcode}"
        return identifier
    
    def generate_cache_key(self, files, prompt, identifier, use_hash, model_key):
        """
        Generate a unique key for the caption cache.
        
        Args:
            files (list): List of file objects to process.
            prompt (str): The prompt text used for processing.
            identifier (str): Base identifier for the captions.
            use_hash (bool): Whether to include a hash in the identifier.
            model_key (str): The model key used for processing.
            
        Returns:
            str: Unique MD5 hash to identify this processing request.
        """
        # Create a string of all filenames and last modified times
        files_info = "|".join([
            f"{file.name}:{file_utils.get_file_modification_time(file.name)}" 
            for file in files
        ])
        # Combine with all parameters to create unique key
        return hashlib.md5(f"{files_info}|{prompt}|{identifier}|{use_hash}|{model_key}".encode()).hexdigest()
    
    def process_images(self, files, prompt, identifier, use_hash, model_key, progress=None, status_callback=None, force_refresh=False, use_cache=True):
        """
        Process multiple images with a given prompt.
        
        Args:
            files (list): List of file objects to process.
            prompt (str): The prompt text used for processing.
            identifier (str): Base identifier for the captions.
            use_hash (bool): Whether to include a hash in the identifier.
            model_key (str): The model key to use for processing.
            progress (function, optional): Callback for progress updates.
            status_callback (function, optional): Callback for status messages.
            force_refresh (bool): Whether to bypass cache and reprocess images.
            use_cache (bool): Whether to use the cache for this operation.
            
        Returns:
            dict: Processed data including gallery, captions, and identifiers.
        """
        # Generate a cache key
        cache_key = self.generate_cache_key(files, prompt, identifier, use_hash, model_key)

        # Check if we have cached results (only if not forcing refresh and use_cache is True)
        if use_cache and not force_refresh and cache_key in self.caption_cache:
            if progress is not None:
                progress(1.0, desc="Using cached results!")
            if status_callback:
                status_callback("Using cached results from previous processing")
            return self.caption_cache[cache_key]
        
        self.initialize_model_and_processor(model_key)
        id_text = self.fluxkey(identifier, use_hash)
        
        results = []
        captions_list = []
        file_captions = {}
        
        # Enable progress tracking
        if progress is not None:
            progress(0, desc="Starting captioning...")
        if status_callback:
            status_callback("Starting image processing...")
            
        total_files = len(files)
        
        for i, file in enumerate(files):
            progress_message = f"Processing image {i+1}/{total_files}: {file_utils.get_filename(file.name)}"
            if progress is not None:
                progress(i/total_files, desc=progress_message)
            if status_callback:
                status_callback(progress_message)
            
            # Process the image
            caption = self.process_image_with_prompts(file, [prompt])[0]
            full_caption = f"{id_text}, {caption}"
            
            # Store results
            with Image.open(file.name) as img:
                thumb = file_utils.create_thumbnail(file.name)
                results.append([thumb, file_utils.get_filename(file.name)])
            
            captions_list.append(full_caption)
            file_captions[file_utils.get_filename(file.name)] = full_caption
                
        if progress is not None:
            progress(1.0, desc="Completed!")
        if status_callback:
            status_callback("Image processing completed")
        
        # Cache the results
        processed_data = {
            'gallery_data': results,
            'captions_list': captions_list,
            'file_captions': file_captions,
            'id_text': id_text
        }
        self.caption_cache[cache_key] = processed_data
        
        return processed_data
    
    def create_captions_zip(self, processed_data):
        """
        Create a ZIP file from processed captions.
        
        Args:
            processed_data (dict): The processed data containing captions.
            
        Returns:
            str: Path to the created ZIP file.
        """
        file_captions = processed_data['file_captions']
        id_text = processed_data['id_text']
        use_hash = '_' in id_text  # Simple check if hash was used
        
        return file_utils.create_captions_zip(file_captions, id_text, use_hash)
    
    def clear_cache(self):
        """
        Clear the caption cache.
        
        Returns:
            str: Confirmation message.
        """
        self.caption_cache = {}
        return "Cache cleared. Images will be processed again on next operation."