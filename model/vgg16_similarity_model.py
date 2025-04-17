# model/vgg16_similarity_model.py

import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cosine
from PIL import Image

class VGG16SimilarityModel:
    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=False, pooling='max')

    def process_images(self, input_dir, output_dir, threshold, progress_callback=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        similar_pairs_dir = os.path.join(output_dir, "similar_pairs")
        if not os.path.exists(similar_pairs_dir):
            os.makedirs(similar_pairs_dir)

        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(image_files)
        image_features = {}
        similar_count = 0

        for i, filename in enumerate(image_files):
            filepath = os.path.join(input_dir, filename)
            features = self.extract_features(filepath)
            image_features[filename] = features

            for other_filename, other_features in list(image_features.items())[:-1]:  # Exclude the current image
                if other_filename != filename:
                    similarity = 1 - cosine(features, other_features)
                    if similarity > threshold:
                        similar_count += 1
                        self.create_similar_pair_image(
                            os.path.join(input_dir, filename),
                            os.path.join(input_dir, other_filename),
                            os.path.join(similar_pairs_dir, f"similar_{similar_count}.jpg"),
                            similarity
                        )
                        os.rename(filepath, os.path.join(output_dir, filename))
                        del image_features[filename]  # Remove the moved image from future comparisons
                        break

            if progress_callback:
                progress_callback(i + 1, total_images)

        return similar_count

    def extract_features(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.model.predict(x)
        return features.flatten()

    def create_similar_pair_image(self, img1_path, img2_path, output_path, similarity):
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # Resize images to be the same height
        height = 300
        img1 = img1.resize((int(img1.width * height / img1.height), height))
        img2 = img2.resize((int(img2.width * height / img2.height), height))

        # Create a new image with the two images side by side
        total_width = img1.width + img2.width
        comparison = Image.new('RGB', (total_width, height))
        comparison.paste(img1, (0, 0))
        comparison.paste(img2, (img1.width, 0))

        # Add similarity score text
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        font = ImageFont.load_default()
        text = f"Similarity: {similarity:.2f}"
        draw.text((10, 10), text, (255, 255, 255), font=font)

        comparison.save(output_path)