import cv2
import dlib
import numpy as np
from PIL import Image
import logging
from typing import List, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceEmbedder:
    def __init__(self, predictor_path: str, recognition_model_path: str):
        """Initialize the face embedding generator."""
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(predictor_path)
            self.face_rec_model = dlib.face_recognition_model_v1(recognition_model_path)
            logger.info("Successfully initialized FaceEmbedder")
        except Exception as e:
            logger.error(f"Failed to initialize FaceEmbedder: {str(e)}")
            raise

    def get_face_embeddings(self, image) -> List[np.ndarray]:
        """Generate face embeddings from an image."""
        try:
            # Handle different image input types
            if isinstance(image, str):
                # Load image from file path
                img = cv2.imread(image)
                if img is None:
                    raise ValueError(f"Failed to load image from path: {image}")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif isinstance(image, Image.Image):
                # Convert PIL Image to numpy array
                img_rgb = np.array(image)
            elif isinstance(image, np.ndarray):
                # Handle numpy array input
                if len(image.shape) == 3 and image.shape[2] == 3:
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
                else:
                    raise ValueError("Invalid image array shape")
            else:
                raise TypeError("Unsupported image type")

            # Detect faces
            faces = self.detector(img_rgb, 1)
            if not faces:
                logger.warning("No faces detected in the image")
                return []

            # Generate embeddings
            face_embeddings = []
            for face in faces:
                shape = self.predictor(img_rgb, face)
                face_embedding = np.array(
                    self.face_rec_model.compute_face_descriptor(img_rgb, shape, 1)
                )
                face_embeddings.append(face_embedding)

            return face_embeddings

        except Exception as e:
            logger.error(f"Error generating face embeddings: {str(e)}")
            raise

    def process_image_folder(self, folder_path: str) -> dict:
        """Process all images in a folder and return their embeddings."""
        results = {
            'embeddings': [],
            'filenames': [],
            'errors': [],
            'stats': {'processed': 0, 'failed': 0}
        }

        try:
            # Check if folder exists
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Folder not found: {folder_path}")

            # Process each image in the folder
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        image_path = os.path.join(folder_path, filename)
                        embeddings = self.get_face_embeddings(image_path)
                        
                        if embeddings:
                            results['embeddings'].extend(embeddings)
                            results['filenames'].extend([filename] * len(embeddings))
                            results['stats']['processed'] += 1
                        else:
                            results['errors'].append({
                                'file': filename,
                                'error': 'No faces detected'
                            })
                            results['stats']['failed'] += 1
                            
                    except Exception as e:
                        results['errors'].append({
                            'file': filename,
                            'error': str(e)
                        })
                        results['stats']['failed'] += 1
                        logger.error(f"Error processing {filename}: {str(e)}")

            return results

        except Exception as e:
            logger.error(f"Error processing folder {folder_path}: {str(e)}")
            raise