# VLM Validator Service

This project provides a FastAPI-based service for validating outputs from detection models (e.g., license plate, age, gender) using multiple AI validators (OpenAI, Gemini, Nvidia).

## Features

- Validate detection results using multiple AI backends (Gemini, NVIDIA, OpenAI)
- Supports image cropping and bounding box handling
- Extensible validator architecture
- API key authentication for all endpoints

## AI Models
- [✅] Gemini
- [🚧] OpenAI
- [✅] NVIDIA NIM microservices (attributes extraction)

## Validators
- [✅] License Plate Recognition
- [✅] Visual Attributes (car, body, face) via NVIDIA
- [🚧] Face Attributes (age, gender)
- [🚧] Vehicle Attributes (color, type, brand)
## NVIDIA Attributes Endpoint

### Endpoint

`POST /validate/attributes`

### Request Body

```
{

## NVIDIA Attributes Endpoint

### Endpoint

`POST /validate/attributes`

### API Key Requirement

All requests must include an API key in the header:

```
X-API-Key: <your_api_key>
```

Set your API key as an environment variable before running client scripts:

```bash
export API_KEY=your_actual_key
```

### Request Body

```json
{
   "image_base64": "<base64-encoded-image>",
   "detection_type": "car"  // or "body" or "face"
}
```

Optionally, you can provide a bounding box:

```json
{
   "image_base64": "<base64-encoded-image>",
   "detection_type": "body",
   "bbox": { "x": 100, "y": 50, "width": 200, "height": 200 }
}
```

### Example Python Request

```python
import base64
import requests
import os

IMAGE_PATH = "data/lpr/N19M43S584_2433132_5_220498_c0.00_r0.00_p0.00_y0.00_b0.00_w0_0.00.jpg"
API_URL = "http://localhost:8080/validate/attributes"
api_key = os.environ.get("API_KEY")

with open(IMAGE_PATH, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

   ```
    "image_base64": b64,
    "detection_type": "body"
}
headers = {"Content-Type": "application/json"}
if api_key:
    headers["X-API-Key"] = api_key


print(r.status_code)
print(r.json())
```

### Response

```json
{
   "success": true,
   "content": {
      "detection_type": "body",
      "attributes": {
         "gender": "female",
         "haircut": "long hair",
         "hat": "",
         "security_helmet": "",
         "upper_body_color": "blue",
         "upper_body_type": "short sleeves",
         "lower_body_color": "black",
         "lower_body_type": "trousers",
         "bag": "handbag",
         "umbrella": false
      }
   }
}
```

On error (missing/invalid key or config):
```json
{
   "success": false,
   "error": "Invalid API key or NVIDIA_NIM_ENDPOINT not configured"
}
```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   - Copy `.env.example` to `.env` and fill in your API keys (OpenAI, Gemini, Nvidia, etc.).
   - Set your API key for authentication:
     ```bash
     export API_KEY=your_actual_key
     ```

4. **Start the API server**
   ```bash
   uvicorn src.app.main:app --host 0.0.0.0 --port 8000
   ```

## Docker

1. **build image**
   ```bash
   git clone <your-repo-url>
   cd ai_validator
   docker build -t ai-validator:dev .
   ```

2. **Run container**
   ```bash
   docker run -it --rm -p 8080:8080 -v .env:/app/.env  --name ai-validator ai-validator:dev
   ```


## Google Cloud Plateform

1. **Install gcloud**
   ```bash
   # Add the Google Cloud SDK distribution URI as a package source
   sudo apt-get install apt-transport-https ca-certificates gnupg curl -y
   echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
   | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list

   # Import the Google Cloud public key
   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
   | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

   # Update and install the CLI
   sudo apt-get update && sudo apt-get install google-cloud-cli -y
   # Once installed run to initialize gcloud
   gcloud init
   ```
2. **Make sure the following services are activated**
   ```bash
   gcloud services enable \
      artifactregistry.googleapis.com \
      run.googleapis.com \
      cloudbuild.googleapis.com \
      secretmanager.googleapis.com
   ```
3. **Push the image to  Registry**
   ```bash
   gcloud auth configure-docker us-central1-docker.pkg.dev

   # We have two Options 
   # Option 1 :
   docker tag ai-validator:dev us-central1-docker.pkg.dev/surveye-468818/serveye/ai-validator:dev
   docker push us-central1-docker.pkg.dev/surveye-468818/serveye/ai-validator:dev 

   # Option 2 :
   # if we want to call also  use builds submit which will build directly to cloud build
   cd <PROJECT_DIR> # Make sure you have Dockefile in it
   gcloud builds submit --tag us-central1-docker.pkg.dev/surveye-468818/serveye/ai-validator:dev
   ```

4. **Store gemini API key as secret**
   ```bash
      echo -n "<YOUR_GEMINI_API_KEY>" | \
      gcloud secrets create APP_GOOGLE_API_KEY --data-file=-
   ```

5. **Deploy to cloud run**
   ```bash
   gcloud run deploy ai-validator \
     --image=us-central1-docker.pkg.dev/surveye-468818/serveye/ai-validator:dev \
     --region=europe-west9 \
     --platform=managed \
     --allow-unauthenticated \
     --set-secrets=APP_GOOGLE_API_KEY=APP_GOOGLE_API_KEY:latest
   ```
   ## API Usage

   ### Endpoint

   `POST /validate`

   ### API Key Requirement

   All requests must include an API key in the header:

   ```
   X-API-Key: <your_api_key>
   ```

   Set your API key as an environment variable before running client scripts:

   ```bash
   export API_KEY=your_actual_key
   ```

   ### Request Body
   If bbox is provided, the license plate is cropped from the full image

   ```json
   {
      "image_base64": "<base64-encoded-image>",
      "bbox": { "x": 1211, "y": 743, "width": 109, "height": 38 },
      "task": "lpr",  
      "validator": "gemini" 
   }
   ```

   You can also call the endpoint by providing the cropped license plate directly without any bbox

   ```json
   {
      "image_base64": "<base64-encoded-image>",
      "task": "lpr",  
      "validator": "gemini" 
   }
   ```

   ### Example Python Request

   ```python
   import base64
   import json
   import requests
   from pathlib import Path
   import os
   API_URL = "https://ai-validator-654942414948.europe-west9.run.app/validate"
   api_key = os.environ.get("API_KEY")

   # Current file's dir
   project_root = Path(__file__).resolve().parent.parent
   IMAGE_PATH = os.path.join(project_root, "data","lpr2", "N19M55S184_2433166_5_220526_c0.00_r0.00_p0.00_y0.00_b0.00_w0_0.00.jpg")

   with open(IMAGE_PATH, "rb") as f:
         image_base64 = base64.b64encode(f.read()).decode("utf-8")

import os 
         "image_base64": image_base64,
         "task": "lpr",
         "validator": "gemini"
   }
   headers = {"Content-Type": "application/json"}
   if api_key:
         headers["X-API-Key"] = api_key

   response = requests.post(API_URL, headers=headers, data=json.dumps(payload), verify=False)
   print(response.json())
   ```
API_URL = "https://ai-validator-654942414948.europe-west9.run.app/validate"

# Current file's dir
project_root = Path(__file__).resolve().parent.parent
IMAGE_PATH = os.path.join(project_root, "data","lpr2", "N19M55S184_2433166_5_220526_c0.00_r0.00_p0.00_y0.00_b0.00_w0_0.00.jpg")

with open(IMAGE_PATH, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "image_base64": image_base64,
    "task": "lpr",
    "validator": "gemini"
}

response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload),verify=False)
print(response.json())
```

### Response

```json
{
  "gemini": {
    "label": "AA 757 GZ",
    "confidence": 95,
    "detail": "License plate detected"
  }
}
```

## Testing

Run unit tests with:

```bash
pytest --maxfail=1 --disable-warnings -q
```

## Project Structure

- `src/app/main.py` – FastAPI app and endpoint
- `src/app/validators/` – Validator implementations (including `nvidia_validator.py` for attributes)
- `src/app/routers/` – Router implementations
- `src/app/deps/auth.py` – API key authentication dependency
- `src/app/models.py` – Pydantic models
- `src/app/settings.py` – Configuration loader (API key, endpoints)
- `test/` – Test scripts and sample requests (all require API key)

##