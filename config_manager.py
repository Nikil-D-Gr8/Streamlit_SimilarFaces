import json
import os
from typing import Dict, Optional, Tuple
import re

class ConfigManager:
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load existing config or create default"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {
            "deployment": {
                "type": "",
                "settings": {
                    "url": "",
                    "api_key": "",
                    "port": 6333
                }
            },
            "collections": {}
        }

    def save_config(self) -> None:
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def setup_deployment(self) -> Tuple[str, str]:
        """Interactive deployment setup"""
        while True:
            deploy_type = input("Choose deployment type (docker/cloud): ").lower().strip()
            if deploy_type not in ['docker', 'cloud']:
                print("Invalid choice. Please enter 'docker' or 'cloud'")
                continue

            if deploy_type == 'docker':
                while True:
                    try:
                        port = input("Enter port number (default: 6333): ").strip()
                        port = 6333 if not port else int(port)
                        if not (1024 <= port <= 65535):
                            raise ValueError("Port must be between 1024 and 65535")
                        url = f"http://localhost:{port}"
                        self.config["deployment"] = {
                            "type": "docker",
                            "settings": {
                                "url": url,
                                "api_key": "",
                                "port": port
                            }
                        }
                        self.save_config()
                        return url, ""
                    except ValueError as e:
                        print(f"Invalid port number: {e}")

            else:  # cloud
                while True:
                    url = input("Enter Qdrant cloud URL: ").strip()
                    if not url:
                        print("URL cannot be empty")
                        continue
                    if not re.match(r'https?://[^\s/$.?#].[^\s]*$', url):
                        print("Invalid URL format")
                        continue
                    
                    api_key = input("Enter API key: ").strip()
                    if not api_key:
                        print("API key cannot be empty")
                        continue

                    self.config["deployment"] = {
                        "type": "cloud",
                        "settings": {
                            "url": url,
                            "api_key": api_key,
                            "port": None
                        }
                    }
                    self.save_config()
                    return url, api_key

    def get_deployment_settings(self) -> Tuple[str, str]:
        """Get current deployment settings"""
        settings = self.config["deployment"]["settings"]
        return settings["url"], settings["api_key"]

    def update_collection_mapping(self, folder_path: str, collection_name: str) -> None:
        """Update the folder to collection mapping"""
        self.config["collections"][folder_path] = collection_name
        self.save_config()

    def get_collection_name(self, folder_path: str) -> Optional[str]:
        """Get collection name for a folder"""
        return self.config["collections"].get(folder_path)