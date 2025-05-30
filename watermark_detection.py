from ultralytics import YOLO
import os
import json
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import msvcrt
import traceback

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

def update_json_file(new_results):
	"""Atomically update the JSON file with new results"""
	json_path = "result.json"
	temp_path = "result.json.tmp"
	
	try:
		# Load existing results if file exists
		existing_results = []
		if os.path.exists(json_path):
			try:
				with open(json_path, "r", encoding='utf-8') as f:
					# Lock the file for reading
					msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
					existing_results = json.load(f)
					# Unlock the file
					msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
			except json.JSONDecodeError:
				print("Warning: Existing result.json was invalid, starting fresh")
			except Exception as e:
				print(f"Error reading existing results: {str(e)}")
				traceback.print_exc()

		# Combine results
		all_results = existing_results + new_results

		# Write to temporary file first
		with open(temp_path, "w", encoding='utf-8') as f:
			# Lock the file for writing
			msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
			json.dump(all_results, f, indent=4)
			# Unlock the file
			msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

		# Atomic rename
		os.replace(temp_path, json_path)
		return True
	except Exception as e:
		print(f"Error updating JSON file: {str(e)}")
		traceback.print_exc()
		if os.path.exists(temp_path):
			try:
				os.remove(temp_path)
			except:
				pass
		return False

def run_inference(image_paths: list):
	try:
		# Ensure output directory exists
		os.makedirs(OUTPUT_DIR, exist_ok=True)
		
		# Process images
		images = [image_enhancer(image_path) for image_path in image_paths]
		results = [model.predict(image, conf=0.003, iou=0.) for image in images]
		
		saved_paths = [p.replace(INPUT_DIR, OUTPUT_DIR) for p in image_paths]
		
		watermark_status = []
		for idx, result in enumerate(results):
			try:
				result = result[0]
				image = Image.open(image_paths[idx])
				
				# Create output directory if it doesn't exist
				os.makedirs(os.path.dirname(saved_paths[idx]), exist_ok=True)
				
				if len(result.boxes) > 0:
					watermark_status.append({
						"image": saved_paths[idx],
						"status": True
					})
					for box in result.boxes:
						coordinates = box.xyxy.tolist()
						draw = ImageDraw.Draw(image)
						draw.rectangle(coordinates[0], outline="red", width=3)
				else:
					watermark_status.append({
						"image": saved_paths[idx],
						"status": False
					})
				
				# Save the image
				image.save(saved_paths[idx])
				print(f"Saved image to {saved_paths[idx]}")
				
			except Exception as e:
				print(f"Error processing image {image_paths[idx]}: {str(e)}")
				traceback.print_exc()
				continue

		# Update JSON file with new results
		if watermark_status:
			if not update_json_file(watermark_status):
				print("Warning: Failed to update result.json with new results")
		else:
			print("Warning: No results to save to JSON file")

	except Exception as e:
		print(f"Error in run_inference: {str(e)}")
		traceback.print_exc()
