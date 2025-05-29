from ultralytics import YOLO
import os
import json
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

model_path = 'watermarks.pt'
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"

model = YOLO(model_path)

def image_enhancer(image_path, threshold=70):
	image = Image.open(image_path).convert("RGB")
	gray_image = image.convert("L")
	enhancer = ImageEnhance.Contrast(gray_image)
	contrast_image = enhancer.enhance(0.85)
	sharpened_image = contrast_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
	sharpened_image = sharpened_image.point(lambda x: x if x > threshold else 0)
	return sharpened_image.convert("RGB")

def run_inference(image_paths: list):
    try:
        images = [image_enhancer(image_path) for image_path in image_paths]
        results = [model.predict(image, conf=0.004, iou=0.) for image in images]
        
        saved_paths = [p.replace(INPUT_DIR, OUTPUT_DIR) for p in image_paths]
        
        watermark_status = []
        for idx, result in enumerate(results):
            result = result[0]

            image = Image.open(image_paths[idx])
            if len(result.boxes) > 0:
                watermark_status.append(
                    {
                        "image": saved_paths[idx],
                        "status": True
                    }
                )
                for box in result.boxes:
                    coordinates = box.xyxy.tolist()
                    draw = ImageDraw.Draw(image)
                    draw.rectangle(coordinates[0], outline="red", width=3)
                image.save(saved_paths[idx])
            else:
                watermark_status.append(
                    {
                        "image": saved_paths[idx],
                        "status": False
                    }
                )
                image.save(saved_paths[idx])

        # Load existing results if file exists
        existing_results = []
        if os.path.exists("result.json"):
            try:
                with open("result.json", "r", encoding='utf-8') as f:
                    existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = []

        # Append new results
        all_results = existing_results + watermark_status

        # Save combined results
        with open("result.json", "w", encoding='utf-8') as f:
            json.dump(all_results, f, indent=4)

    except Exception as e:
        print(e)
