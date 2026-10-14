import os
import base64
import json
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# ==== Configuration ====
API_URL = "http://localhost:8080/validate"
# API_URL ="https://ai-validator-654942414948.europe-west9.run.app/validate"
# API_URL = "https://ai-validator-bqvwxg2cvq-od.a.run.app/validate"
INPUT_FOLDER = "/home/tr_user/surveye/ai_validator/data/lpr"     # Folder containing input images
OUTPUT_FOLDER = "/home/tr_user/surveye/ai_validator/data/output"    # Folder to save annotated images

# Example bounding box
bbox = {"x": 1211, "y": 743, "width": 109, "height": 38}

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==== Process all images in the input folder ====
for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue  # skip non-image files

    image_path = os.path.join(INPUT_FOLDER, filename)
    print(f"\n📷 Processing: {filename}")

    # === 1. Encode image ===
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    # === 2. Prepare payload ===
    payload = {
        "image_base64": image_base64,
        "bbox": None,
        "task": "lpr",
        "validator": "gemini"
    }

    response_healthz = requests.post("https://ai-validator-bqvwxg2cvq-od.a.run.app/healthz", headers={"Content-Type": "application/json"},verify=False)
    print("Health Check Response:", response_healthz.text)
    exit(0)  # Exit after health check for testing
    # === 3. Send request ===
    response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload),verify=False)
    
    if response.status_code != 200:
        print(f"❌ Request failed for {filename}: {response.text}")
        continue
    
    result = response.json()
    print("✅ API Response:", json.dumps(result, indent=2))

    # === 4. Draw bounding box & label ===
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Draw rectangle
    # x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    # draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
    
    # Extract label
    label = list(result.values())[0]["label"]
    
    # Try to load a bigger font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)  # Increase from 50 → 80
    except IOError:
        print("⚠️ Using default font (size not adjustable).")
        font = ImageFont.load_default()

    # Draw text
    draw.text((0, 20), label, font=font, fill="yellow")
    
    # === 5. Save annotated image ===
    output_path = os.path.join(OUTPUT_FOLDER, f"annotated_{filename}")
    img.save(output_path)
    print(f"💾 Saved annotated image to {output_path}")
