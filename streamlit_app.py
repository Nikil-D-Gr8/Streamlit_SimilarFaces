import streamlit as st
from face_embeddings import FaceEmbedder
from database_manager import DatabaseManager
from datetime import datetime
from PIL import Image
import os
from typing import Dict, Any
from password import URL, APIKEY

# Constants
PREDICTOR_PATH = "DAT/shape_predictor_68_face_landmarks.dat"
RECOGNITION_MODEL_PATH = "DAT/dlib_face_recognition_resnet_model_v1.dat"
CONFIG_FILE = 'config.json'

def initialize_services():
    """Initialize the required services."""
    try:
        embedder = FaceEmbedder(PREDICTOR_PATH, RECOGNITION_MODEL_PATH)
        db_manager = DatabaseManager(URL, APIKEY, CONFIG_FILE)
        return embedder, db_manager
    except Exception as e:
        st.error(f"Failed to initialize services: {str(e)}")
        st.stop()

def process_folder(folder_path: str, embedder: FaceEmbedder, 
                  db_manager: DatabaseManager) -> Dict[str, Any]:
    """Process a folder of images and store in database."""
    try:
        # Generate embeddings
        embedding_results = embedder.process_image_folder(folder_path)
        
        # Create new collection
        collection_name = datetime.now().strftime("%Y%m%d%H%M%S")
        creation_status = db_manager.create_collection(collection_name)
        
        if creation_status != "Success":
            return {"status": "error", "message": creation_status}
        
        # Store embeddings
        if embedding_results['embeddings']:
            storage_results = db_manager.store_embeddings(
                collection_name,
                embedding_results['embeddings'],
                embedding_results['filenames']
            )
            
            # Update config
            config = db_manager.load_config()
            config[folder_path] = collection_name
            db_manager.save_config(config)
            
            return {
                "status": "success",
                "collection_name": collection_name,
                "processed": embedding_results['stats']['processed'],
                "failed": embedding_results['stats']['failed'],
                "stored": storage_results['success_count'],
                "store_errors": storage_results['error_count']
            }
        
        return {"status": "error", "message": "No embeddings generated"}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def main():
    st.set_page_config(page_title="Face Similarity Search", layout="wide")
    st.title("Face Similarity Search")

    # Initialize services
    embedder, db_manager = initialize_services()

    # Create tabs
    tab1, tab2 = st.tabs(["Upload Database", "Search Faces"])

    with tab1:
        st.header("Upload Face Database")
        folder_path = st.text_input("Enter folder path containing images:")
        
        if st.button("Process Folder"):
            if not folder_path:
                st.error("Please enter a folder path")
            elif not os.path.exists(folder_path):
                st.error("Folder path does not exist")
            else:
                with st.spinner("Processing images..."):
                    results = process_folder(folder_path, embedder, db_manager)
                    
                    if results["status"] == "success":
                        st.success(f"""
                            Processing complete:
                            - Images processed: {results['processed']}
                            - Processing errors: {results['failed']}
                            - Embeddings stored: {results['stored']}
                            - Storage errors: {results['store_errors']}
                            - Collection name: {results['collection_name']}
                        """)
                    else:
                        st.error(f"Error: {results['message']}")

    with tab2:
        st.header("Search Similar Faces")
        uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
        search_folder = st.text_input("Enter the folder path to search in:")
        
        if uploaded_file and search_folder:
            try:
                config = db_manager.load_config()
                collection_name = config.get(search_folder)
                
                if not collection_name:
                    st.error("No database found for this folder. Please process it first.")
                else:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", width=300)
                    
                    with st.spinner("Searching for similar faces..."):
                        embeddings = embedder.get_face_embeddings(image)
                        
                        if not embeddings:
                            st.warning("No faces detected in the uploaded image")
                        else:
                            results = db_manager.search_similar_faces(
                                collection_name, 
                                embeddings[0].tolist()
                            )
                            
                            if results:
                                st.subheader("Similar Faces Found:")
                                cols = st.columns(len(results))
                                for idx, (result, col) in enumerate(zip(results, cols)):
                                    try:
                                        img_path = os.path.join(search_folder, result["image"])
                                        match_img = Image.open(img_path)
                                        col.image(match_img, caption=f"Match {idx + 1}")
                                    except Exception as e:
                                        col.error(f"Error loading image: {result['image']}")
                            else:
                                st.info("No similar faces found")
                                
            except Exception as e:
                st.error(f"Error during search: {str(e)}")

if __name__ == "__main__":
    main()