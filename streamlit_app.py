import streamlit as st
from face_embeddings import FaceEmbedder
from database_manager import DatabaseManager
from config_manager import ConfigManager
from datetime import datetime
from PIL import Image
import os
from typing import Dict, Any
import logging

# Constants
PREDICTOR_PATH = "DAT/shape_predictor_68_face_landmarks.dat"
RECOGNITION_MODEL_PATH = "DAT/dlib_face_recognition_resnet_model_v1.dat"
CONFIG_FILE = 'config.json'

def initialize_services():
    """Initialize the required services."""
    try:
        config_manager = ConfigManager(CONFIG_FILE)
        
        # Check if deployment is configured
        if not config_manager.config["deployment"]["type"]:
            st.warning("First-time setup: Please configure deployment settings")
            deployment_type = st.radio("Choose deployment type:", ["Docker", "Cloud"])
            
            if deployment_type == "Docker":
                port = st.number_input("Enter port number:", min_value=1024, max_value=65535, value=6333)
                url = f"http://localhost:{port}"
                api_key = ""
            else:  # Cloud
                url = st.text_input("Enter Qdrant cloud URL:")
                api_key = st.text_input("Enter API key:", type="password")
                
                if not url or not api_key:
                    st.error("Please provide both URL and API key")
                    st.stop()
                
                if not url.startswith(('http://', 'https://')):
                    st.error("Invalid URL format")
                    st.stop()
            
            config_manager.config["deployment"] = {
                "type": deployment_type.lower(),
                "settings": {
                    "url": url,
                    "api_key": api_key,
                    "port": port if deployment_type == "Docker" else None
                }
            }
            config_manager.save_config()
        
        # Get deployment settings
        url, api_key = config_manager.get_deployment_settings()
        
        embedder = FaceEmbedder(PREDICTOR_PATH, RECOGNITION_MODEL_PATH)
        db_manager = DatabaseManager(url, api_key, CONFIG_FILE)
        return embedder, db_manager, config_manager
    
    except Exception as e:
        st.error(f"Failed to initialize services: {str(e)}")
        st.stop()

def process_folder(folder_path: str, embedder: FaceEmbedder, 
                  db_manager: DatabaseManager) -> Dict[str, Any]:
    """Process a folder of images and store in database."""
    try:
        # Create placeholder for progress
        progress_text = st.empty()
        
        # Count total images
        total_images = len([f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        progress_text.text(f"Found {total_images} images to process...")
        
        # Generate embeddings
        embedding_results = embedder.process_image_folder(folder_path)
        
        # Create new collection
        collection_name = datetime.now().strftime("%Y%m%d%H%M%S")
        creation_status = db_manager.create_collection(collection_name)
        
        if creation_status != "Success":
            return {"status": "error", "message": creation_status}
        
        # Store embeddings
        if embedding_results['embeddings']:
            progress_text.text("Storing embeddings in database...")
            storage_results = db_manager.store_embeddings(
                collection_name,
                embedding_results['embeddings'],
                embedding_results['filenames']
            )
            
            # Update config
            config = db_manager.load_config()
            config[folder_path] = collection_name
            db_manager.save_config(config)
            
            # Store the collection name in session state
            st.session_state.current_collection = collection_name
            
            # Clear progress text
            progress_text.empty()
            
            return {
                "status": "success",
                "collection_name": collection_name,
                "processed": embedding_results['stats']['processed'],
                "failed": embedding_results['stats']['failed'],
                "stored": storage_results['success_count'],
                "store_errors": storage_results['error_count']
            }
        
        progress_text.empty()
        return {"status": "error", "message": "No embeddings generated"}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def main():
    st.set_page_config(page_title="Face Similarity Search", layout="wide")
    st.title("Face Similarity Search")

    # Initialize session state for current collection if not exists
    if 'current_collection' not in st.session_state:
        st.session_state.current_collection = None
    
    if 'database_populated' not in st.session_state:
        st.session_state.database_populated = False

    # First-time setup or deployment selection
    if 'deployment_configured' not in st.session_state:
        st.subheader("Choose Deployment Type")
        deployment_type = st.radio("Select database deployment:", ["Docker", "Cloud"])
        
        if deployment_type == "Docker":
            port = st.number_input("Enter port number:", min_value=1024, max_value=65535, value=6333)
            url = f"http://localhost:{port}"
            api_key = ""
        else:  # Cloud
            url = st.text_input("Enter Qdrant cloud URL:")
            api_key = st.text_input("Enter API key:", type="password")
            
            if not url or not api_key:
                st.error("Please provide both URL and API key")
                st.stop()
            
            if not url.startswith(('http://', 'https://')):
                st.error("Invalid URL format")
                st.stop()
        
        if st.button("Save Configuration"):
            config_manager = ConfigManager(CONFIG_FILE)
            config_manager.config["deployment"] = {
                "type": deployment_type.lower(),
                "settings": {
                    "url": url,
                    "api_key": api_key,
                    "port": port if deployment_type == "Docker" else None
                }
            }
            config_manager.save_config()
            st.session_state.deployment_configured = True
            st.rerun()  # Changed from experimental_rerun() to rerun()
    else:
        # Initialize services
        embedder, db_manager, config_manager = initialize_services()

        # Check if any collections exist
        config = db_manager.load_config()
        collections = [value for key, value in config.items() 
                      if key not in ["deployment", "collections"]]
        if collections:
            st.session_state.database_populated = True

        # Create tabs
        tab1, tab2 = st.tabs(["Upload Database", "Search Faces"])

        with tab1:
            st.header("Upload Face Database")
            
            # Text input for folder path
            folder_path = st.text_input("Enter folder path:")
            
            if folder_path:
                if st.button("Process Folder"):
                    if not os.path.exists(folder_path):
                        st.error("Folder path does not exist")
                    else:
                        with st.spinner("Processing images..."):
                            results = process_folder(folder_path, embedder, db_manager)
                            
                            if results["status"] == "success":
                                st.session_state.database_populated = True
                                st.success(f"""
                                    Processing complete:
                                    - Images processed: {results['processed']}
                                    - Processing errors: {results['failed']}
                                    - Embeddings stored: {results['stored']}
                                    - Storage errors: {results['store_errors']}
                                    - Collection name: {results['collection_name']}
                                """)
                                st.info("You can now switch to the 'Search Faces' tab to search for similar faces.")
                            else:
                                st.error(f"Error: {results['message']}")

        with tab2:
            if not st.session_state.database_populated:
                st.warning("Please upload and process images in the 'Upload Database' tab first.")
                st.stop()
            
            st.header("Search Similar Faces")
            
            # Get available collections
            config = db_manager.load_config()
            collections = []
            for key, value in config.items():
                if key != "deployment" and key != "collections":
                    collections.append(value)
            collections = list(set(collections))  # Get unique values
            
            if not collections:
                st.warning("No collections available. Please process a folder first.")
            else:
                # Use the current_collection from session state if available
                default_index = 0
                if st.session_state.current_collection in collections:
                    default_index = collections.index(st.session_state.current_collection)
                
                # Allow user to select collection
                selected_collection = st.selectbox(
                    "Select collection to search:",
                    collections,
                    index=default_index
                )
                
                # Update session state with selected collection
                st.session_state.current_collection = selected_collection
                
                # Number of matches to return
                num_matches = st.slider("Number of matches to return:", 1, 20, 5)
                
                uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
                
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", width=300)
                    
                    if st.button("Search"):
                        with st.spinner("Searching for similar faces..."):
                            embeddings = embedder.get_face_embeddings(image)
                            
                            if not embeddings:
                                st.warning("No faces detected in the uploaded image")
                            else:
                                # Store the embedding immediately
                                storage_result = db_manager.store_embeddings(
                                    selected_collection,
                                    [embeddings[0]],  # Store only the first face if multiple detected
                                    [uploaded_file.name]
                                )
                                
                                # Search for similar faces
                                results = db_manager.search_similar_faces(
                                    selected_collection, 
                                    embeddings[0].tolist(),
                                    limit=num_matches
                                )
                                
                                if results:
                                    st.subheader(f"Top {num_matches} Similar Faces Found:")
                                    cols = st.columns(min(len(results), 5))  # Max 5 images per row
                                    for idx, result in enumerate(results):
                                        col_idx = idx % 5
                                        if col_idx == 0 and idx > 0:
                                            cols = st.columns(min(len(results) - idx, 5))
                                        try:
                                            img_path = os.path.join(
                                                next(k for k, v in config.items() 
                                                    if v == selected_collection), 
                                                result["image"]
                                            )
                                            if os.path.exists(img_path):
                                                match_img = Image.open(img_path)
                                                cols[col_idx].image(
                                                    match_img, 
                                                    caption=f"Match {idx + 1}"
                                                )
                                            else:
                                                cols[col_idx].error(
                                                    f"Image not found: {result['image']}"
                                                )
                                        except Exception as e:
                                            cols[col_idx].error(
                                                f"Error loading image: {result['image']}"
                                            )
                                else:
                                    st.info("No similar faces found")

        # Add option to change deployment
        if st.sidebar.button("Change Deployment"):
            del st.session_state.deployment_configured
            st.session_state.current_collection = None
            st.session_state.database_populated = False
            st.rerun()  # Changed from experimental_rerun() to rerun()

if __name__ == "__main__":
    main()
