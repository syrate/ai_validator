import base64
import json
import requests
from pathlib import Path
import os 
API_URL = "https://ai-validator-654942414948.europe-west9.run.app/validate"
# API_URL = "http://localhost:8080/validate"
# Current file's dir
project_root = Path(__file__).resolve().parent.parent
print("Project root:", project_root)


IMAGE_PATH = os.path.join(project_root, "data","lpr2", "N24M10S727_2460004_5_220850_c0.00_r0.00_p0.00_y0.00_b0.00_w0_0.00.jpg")
# IMAGE_PATH = os.path.join(project_root, "data","lpr2", "64cd543d-947e-422e-96b3-990e8de11fce.jpeg")

print("Image path:", IMAGE_PATH)
with open(IMAGE_PATH, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "image_base64": image_base64,
    "task": "lpr",
    "validator": "gemini"
}

# Add API key from environment if available
api_key = os.environ.get("VALIDATOR_API_KEY")
headers = {"Content-Type": "application/json"}
if api_key:
    headers["X-API-Key"] = api_key
response = requests.post(API_URL, headers=headers, data=json.dumps(payload), verify=False)
print(response.json())