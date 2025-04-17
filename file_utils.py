# file_utils.py

import os
import tempfile
import zipfile
from pathlib import Path
import math
from PIL import Image, ImageOps

def create_thumbnail(image_path, max_size=(256, 256)):
    """
    Create a thumbnail of an image
    
    Args:
        image_path (str): Path to the image file
        max_size (tuple): Maximum width and height of the thumbnail
        
    Returns:
        PIL.Image: Thumbnail image
    """
    with Image.open(image_path) as img:
        thumb = ImageOps.contain(img, max_size)
        return thumb

def resize_image_if_needed(image_path, max_dimension=2048):
    """
    Resize image if it's too large for efficient processing
    
    Args:
        image_path (str): Path to the image file
        max_dimension (int): Maximum dimension (width or height)
        
    Returns:
        str: Path to the resized image (or original if no resize needed)
    """
    with Image.open(image_path) as img:
        width, height = img.size
        
        if width > max_dimension or height > max_dimension:
            # Calculate new dimensions while preserving aspect ratio
            ratio = min(max_dimension / width, max_dimension / height)
            new_width = math.floor(width * ratio)
            new_height = math.floor(height * ratio)
            
            # Resize and save to temporary file
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            resized_img.save(temp_file.name, format='JPEG', quality=95)
            return temp_file.name
            
    # If no resizing needed, return original path
    return image_path

def create_text_file(directory, filename, content):
    """
    Create a text file with specified content
    
    Args:
        directory (str): Directory to create the file in
        filename (str): Name of the file (with extension)
        content (str): Content to write to the file
        
    Returns:
        Path: Path to the created file
    """
    file_path = Path(directory) / filename
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path

def create_captions_zip(file_captions, id_text, use_hash=True):
    """
    Create a ZIP file containing caption text files
    
    Args:
        file_captions (dict): Dictionary mapping filenames to captions
        id_text (str): Identifier text for the ZIP file name
        use_hash (bool): Whether to use a hash in the filename
        
    Returns:
        str: Path to the created ZIP file
    """
    # Create a temporary directory for the text files
    tmp_dir = tempfile.mkdtemp()
    
    # Create text files from the captions
    txt_paths = []
    for filename, caption in file_captions.items():
        txt_name = Path(filename).with_suffix('.txt')
        txt_path = create_text_file(tmp_dir, txt_name, caption)
        txt_paths.append(txt_path)
    
    # Create ZIP file
    zip_filename = f"{id_text}_captions.zip" if use_hash else "captions.zip"
    zip_path = Path(tmp_dir) / zip_filename
    
    with zipfile.ZipFile(zip_path, "w") as zf:
        for txt in txt_paths:
            zf.write(txt, txt.name)
    
    return str(zip_path)

def clean_temp_file(file_path, original_path):
    """
    Clean up a temporary file if it's different from the original
    
    Args:
        file_path (str): Path to possibly temporary file
        original_path (str): Path to original file
    """
    if file_path != original_path and os.path.exists(file_path):
        os.unlink(file_path)

def get_filename(file_path):
    """
    Get the filename from a file path
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Filename without directory
    """
    return os.path.basename(file_path)

def get_file_modification_time(file_path):
    """
    Get the last modification time of a file
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        float: Last modification time as a timestamp
    """
    return os.path.getmtime(file_path)

def change_extension(file_path, new_extension):
    """
    Change the extension of a file path
    
    Args:
        file_path (str): Original file path
        new_extension (str): New extension (with or without dot)
        
    Returns:
        str: File path with new extension
    """
    if not new_extension.startswith('.'):
        new_extension = '.' + new_extension
    
    return str(Path(file_path).with_suffix(new_extension))