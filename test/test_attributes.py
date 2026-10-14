# test_nvidia_attributes.py
import base64
import json
import requests
from pathlib import Path
import os
project_root = Path(__file__).resolve().parent.parent
print("Project root:", project_root)
IMAGE_PATH = os.path.join(project_root, "data","attributes", "13a1e950-d6b2-4251-b482-c2d0b56be190.jpeg")

# URL = "http://localhost:8000/validate/attributes"
URL = "https://ai-validator-654942414948.europe-west9.run.app/validate/attributes"

def send_request(detection_type="body", bbox=None):
    with open(IMAGE_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image_base64": b64,
        "detection_type": detection_type
    }
    if bbox:
        payload["bbox"] = bbox  # example: {"x":100, "y":50, "width":200, "height":200}

    api_key = os.environ.get("VALIDATOR_API_KEY")
    print("Using API key:", api_key is not None)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    r = requests.post(URL, json=payload, headers=headers, timeout=120*2)
    print("status:", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)

if __name__ == "__main__":
    # test the body extractor
    send_request("body")
    # test the car extractor with an optional bbox
    # send_request("face")