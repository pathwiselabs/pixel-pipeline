# model/image_hash_model.py

import os
import imagehash
from PIL import Image

class ImageHashModel:
    def process_images(self, input_dir, output_dir, progress_callback=None):
        """
        Process images to find duplicates using perceptual hashing.
        Moves duplicate images to the output directory.
        
        Args:
            input_dir (str): Directory containing original images
            output_dir (str): Directory to move duplicate images to
            progress_callback (callable): Function to report progress
            
        Returns:
            int: Number of duplicate images found and moved
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_hashes = {}
        duplicate_count = 0
        total_images = len([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        for i, filename in enumerate(os.listdir(input_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(input_dir, filename)
                
                try:
                    with Image.open(filepath) as img:
                        hash = str(imagehash.average_hash(img))

                    is_duplicate = False
                    for existing_hash, existing_filename in image_hashes.items():
                        if hash == existing_hash:
                            # This is a duplicate - move it to output dir
                            output_path = os.path.join(output_dir, filename)
                            os.rename(filepath, output_path)
                            duplicate_count += 1
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        image_hashes[hash] = filename
                        
                except Exception as e:
                    print(f"Error processing file {filename}: {str(e)}")

                if progress_callback:
                    progress_callback(i + 1, total_images)

        return duplicate_count