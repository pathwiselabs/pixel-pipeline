# model/face_detection_model.py

import os
import shutil
from deepface import DeepFace

class FaceDetectionModel:
    def __init__(self):
        self.detector_backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'centerface']

    def detect_faces(self, image_path, detector_backend='retinaface', align=True):
        if detector_backend not in self.detector_backends:
            raise ValueError(f"Invalid detector backend. Choose from: {', '.join(self.detector_backends)}")
        
        try:
            face_objs = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=detector_backend,
                align=align
            )
            print(f"Detection result for {image_path}: {len(face_objs)} face(s) detected")
            return len(face_objs)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return 0  # Return 0 instead of -1 to indicate no faces detected due to error

    def move_image(self, image_path, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        shutil.move(image_path, os.path.join(output_dir, os.path.basename(image_path)))

    def process_images(self, input_dir, output_dir, detector_backend, align, progress_callback=None):
        images = [img for img in os.listdir(input_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        no_face_count, multiple_faces_count = 0, 0

        for i, image_name in enumerate(images, 1):
            image_path = os.path.join(input_dir, image_name)
            num_faces = self.detect_faces(image_path, detector_backend, align)

            if num_faces == 0:
                no_face_count += 1
                self.move_image(image_path, os.path.join(output_dir, "no_faces"))
            elif num_faces > 1:
                multiple_faces_count += 1
                self.move_image(image_path, os.path.join(output_dir, "multiple_faces"))

            if progress_callback:
                progress_callback(i, len(images))

        return no_face_count, multiple_faces_count, len(images)