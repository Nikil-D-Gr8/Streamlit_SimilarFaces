from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
import json
import logging
from typing import List, Dict, Any, Optional
import uuid
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, url: str, api_key: str, config_path: str = 'config.json'):
        """Initialize the database manager."""
        try:
            self.client = QdrantClient(url=url, api_key=api_key)
            self.config_path = config_path
            logger.info("Successfully initialized DatabaseManager")
        except Exception as e:
            logger.error(f"Failed to initialize DatabaseManager: {str(e)}")
            raise

    def load_config(self) -> Dict:
        """Load configuration from JSON file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    return json.load(file)
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}

    def save_config(self, config: Dict) -> None:
        """Save configuration to JSON file."""
        try:
            with open(self.config_path, 'w') as file:
                json.dump(config, file, indent=4)
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            raise

    def create_collection(self, collection_name: str) -> str:
        """Create a new collection in the database."""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=128, distance=Distance.DOT),
            )
            return "Success"
        except UnexpectedResponse as e:
            if "already exists" in str(e):
                return "Collection already exists"
            raise
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise

    def store_embeddings(self, collection_name: str, embeddings: List[Any], 
                        filenames: List[str]) -> Dict[str, Any]:
        """Store embeddings in the database."""
        results = {
            'success_count': 0,
            'error_count': 0,
            'errors': []
        }

        try:
            for embedding, filename in zip(embeddings, filenames):
                try:
                    point_id = str(uuid.uuid4())
                    self.client.upsert(
                        collection_name=collection_name,
                        wait=True,
                        points=[PointStruct(
                            id=point_id,
                            vector=embedding.tolist(),
                            payload={"image": filename}
                        )]
                    )
                    results['success_count'] += 1
                except Exception as e:
                    results['error_count'] += 1
                    results['errors'].append({
                        'file': filename,
                        'error': str(e)
                    })
                    logger.error(f"Error storing embedding for {filename}: {str(e)}")

            return results

        except Exception as e:
            logger.error(f"Error in store_embeddings: {str(e)}")
            raise

    def search_similar_faces(self, collection_name: str, query_vector: List[float], 
                            limit: int = 5) -> List[Dict]:
        """Search for similar faces in the database."""
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return [result.payload for result in results]
        except Exception as e:
            logger.error(f"Error searching similar faces: {str(e)}")
            raise

    def get_collection_info(self, collection_name: str) -> Dict:
        """Get information about a collection."""
        try:
            return self.client.get_collection(collection_name=collection_name)
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise
