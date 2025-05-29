import os
import json
import requests
import shutil
from typing import List, Dict
import asyncio
import aiohttp
import ssl
import certifi
from watermark_detection import run_inference as detect_watermarks

class FigmaPipeline:
    def __init__(self, figma_file_key: str, figma_access_token: str, batch_size: int = 10):
        self.figma_file_key = figma_file_key
        self.figma_access_token = figma_access_token
        self.batch_size = batch_size
        self.input_dir = "input_images"
        self.output_dir = "output_images"
        self._setup_directories()
        
        # Configure SSL context
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

    def _setup_directories(self):
        """Create input and output directories if they don't exist"""
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def _clear_input_directory(self):
        """Clear all files from the input directory"""
        for filename in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    async def _download_image(self, session: aiohttp.ClientSession, image_url: str, filename: str) -> str:
        """Download a single image asynchronously"""
        try:
            async with session.get(image_url, ssl=self.ssl_context) as response:
                if response.status == 200:
                    filepath = os.path.join(self.input_dir, filename)
                    with open(filepath, 'wb') as f:
                        f.write(await response.read())
                    return filepath
                else:
                    print(f"Failed to download {image_url}: {response.status}")
                    return None
        except Exception as e:
            print(f"Error downloading {image_url}: {e}")
            return None

    async def _process_batch(self, image_urls: List[str], batch_num: int) -> List[str]:
        """Process a batch of images asynchronously"""
        connector = aiohttp.TCPConnector(ssl=self.ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for idx, url in enumerate(image_urls):
                filename = f"image_batch{batch_num}_{idx}.png"
                task = self._download_image(session, url, filename)
                tasks.append(task)
            
            downloaded_paths = await asyncio.gather(*tasks)
            return [path for path in downloaded_paths if path is not None]

    def _get_figma_images(self) -> List[str]:
        """Get image URLs from Figma file"""
        headers = {
            "X-Figma-Token": self.figma_access_token
        }
        
        # First get the file data to get node IDs
        file_url = f"https://api.figma.com/v1/files/{self.figma_file_key}"
        print(f"Fetching file data from: {file_url}")
        
        # Configure requests session with SSL verification disabled
        session = requests.Session()
        session.verify = False
        file_response = session.get(file_url, headers=headers)
        
        if file_response.status_code != 200:
            raise Exception(f"Failed to get Figma file: {file_response.status_code} - {file_response.text}")
        
        # Get all image nodes from the file
        data = file_response.json()
        image_nodes = []
        
        def extract_image_nodes(node, path=""):
            current_path = f"{path}/{node.get('name', 'unnamed')}"
            print(f"Checking node: {current_path} (type: {node.get('type')})")
            
            # Check if this node is an image or contains images
            if node.get('type') in ['FRAME', 'COMPONENT', 'INSTANCE', 'IMAGE', 'RECTANGLE']:
                if node.get('fills'):
                    for fill in node.get('fills', []):
                        if fill.get('type') == 'IMAGE':
                            print(f"Found image node: {current_path}")
                            image_nodes.append(node['id'])
                            break
            
            # Recursively check children
            if 'children' in node:
                for child in node['children']:
                    extract_image_nodes(child, current_path)
        
        print("\nStarting node extraction...")
        extract_image_nodes(data['document'])
        print(f"\nFound {len(image_nodes)} image nodes")
        
        if not image_nodes:
            print("No image nodes found in the file")
            return []
            
        # Get image URLs for all nodes
        images_url = f"https://api.figma.com/v1/images/{self.figma_file_key}"
        params = {
            'ids': ','.join(image_nodes),
            'format': 'png',
            'scale': 1
        }
        
        print(f"\nFetching image URLs from: {images_url}")
        print(f"Node IDs: {params['ids']}")
        
        images_response = session.get(images_url, headers=headers, params=params)
        
        if images_response.status_code != 200:
            raise Exception(f"Failed to get image URLs: {images_response.status_code} - {images_response.text}")
        
        images_data = images_response.json()
        image_urls = list(images_data.get('images', {}).values())
        print(f"\nRetrieved {len(image_urls)} image URLs")
        
        return image_urls

    async def run_pipeline(self):
        """Run the complete pipeline"""
        try:
            # Get all image URLs from Figma
            image_urls = self._get_figma_images()
            
            # Process images in batches
            for i in range(0, len(image_urls), self.batch_size):
                batch = image_urls[i:i + self.batch_size]
                print(f"Processing batch {i//self.batch_size + 1}")
                
                # Download batch of images
                downloaded_paths = await self._process_batch(batch, i//self.batch_size + 1)
                
                if downloaded_paths:
                    # Run inference on the batch
                    self.run_inference(downloaded_paths)
                    
                    # Clear input directory after processing
                    self._clear_input_directory()
                
                print(f"Completed batch {i//self.batch_size + 1}")
                
        except Exception as e:
            print(f"Pipeline error: {e}")

    def run_inference(self, image_paths: List[str]):
        """Run watermark detection on the given images"""
        detect_watermarks(image_paths)

async def main():
    # Replace these with your actual Figma credentials
    FIGMA_FILE_KEY = "your_figma_file_key"
    FIGMA_ACCESS_TOKEN = "your_figma_access_token"
    
    pipeline = FigmaPipeline(
        figma_file_key=FIGMA_FILE_KEY,
        figma_access_token=FIGMA_ACCESS_TOKEN,
        batch_size=10
    )
    
    await pipeline.run_pipeline()

if __name__ == "__main__":
    asyncio.run(main()) 