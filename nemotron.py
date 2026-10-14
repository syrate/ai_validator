import requests
import os
import base64
import sys
import json

# ---------- CONFIG ----------
invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
kApiKey = "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"
stream = False
# ----------------------------

# --- SUPPORTED MEDIA TYPES ---
kSupportedList = {
    "png": ["image/png", "image_url"],
    "jpg": ["image/jpeg", "image_url"],
    "jpeg": ["image/jpeg", "image_url"],
    "webp": ["image/webp", "image_url"],
    "mp4": ["video/mp4", "video_url"],
    "webm": ["video/webm", "video_url"],
    "mov": ["video/mov", "video_url"]
}

def get_extension(filename):
    return os.path.splitext(filename)[1][1:].lower()

def mime_type(ext):
    return kSupportedList[ext][0]

def media_type(ext):
    return kSupportedList[ext][1]

def encode_media_base64(media_file):
    """Encode media file to base64 string"""
    with open(media_file, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ----------- SERVEYE PROMPT -----------
SERVEYE_PROMPT = """
You are ServEye — an intelligent video analytics agent designed to extract detailed visual attributes from detected entities (car, person, or face) in surveillance footage.
Your analysis must consider that the environment and people are located in Senegal, West Africa (especially Dakar), so your color, clothing, and contextual interpretations should reflect local appearances and lighting conditions.

You will receive a detection_type: either "car", "body", or "face".
You must extract the corresponding attributes and return them strictly in JSON format.
If an attribute cannot be determined with confidence, return an empty string ("") for that field.

⚠️ IMPORTANT RULES:
- Return **JSON ONLY** — no text or explanations.
- Include **ONLY the keys listed below** for the given detection type.
- Do NOT add or rename keys, even if you detect additional information.
- For any attribute that cannot be determined, return an empty string ("") or false.
- The JSON must include `"detection_type"` and `"attributes"` objects exactly.

=== Extraction Rules ===
1. If detection_type = "car"
   - vehicle_type: one of ["car", "SUV", "minivan", "minibus", "bus", "pickup", "truck"]
   - brand: vehicle brand (Toyota, Peugeot, Renault, Hyundai, etc.)
   - color: main visible color of the vehicle body
   - direction: "front", "rear", "left", or "right"

2. If detection_type = "body"
   - gender: "male" / "female" / ""
   - haircut: "short hair" / "long hair" / ""
   - hat: Yes or No    , if unknown return ""
   - upper_body_color: color of the top clothing  , if multiple colors, return "multi-color"
   - upper_body_type: "long sleeves" / "short sleeves" / ""
   - lower_body_color: color of the bottom clothing , if multiple colors, return "multi-color"
   - lower_body_type: "trousers" / "shorts" / "skirt" / ""
   - bag: "backpack" / "handbag" / "trolley" / ""
   - umbrella: Yes or No    , if unknown return ""

3. If detection_type = "face"
   - gender: "male" / "female" / ""
   - age: approximate numeric age or range (e.g., "25–30")
   - "scarf": Yes or No, if unknown return ""
   - skin_tone: brief descriptor (e.g., "light brown", "dark", "medium brown")
   - haircut: one of ["bald", "few hair", "short hair", "long hair", ""]
   - beard_type: "no beard" / "beard not obvious" / "beard"
   - hat_color: color of the hat (e.g., "white", "blue", "black", "")
   - mask_type: "Yes or No", if unknown return ""
   - glass_type: "Yes or No", if unknown return ""

=== Output Format (JSON) ===
{
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

Rules:
- Always include "detection_type" and "attributes".
- Use "" for missing attributes, not null.
- Keep all keys even if their values are empty.
- Respond with JSON only, no text explanations.
"""
# --------------------------------------

def chat_with_media(infer_url, media_files, detection_type, stream=False):
    assert isinstance(media_files, list), f"{media_files}"
    has_video = False

    # Build user content (text + media)
    content = [{"type": "text", "text": f"Extract attributes for detection_type='{detection_type}'"}]
    for media_file in media_files:
        ext = get_extension(media_file)
        assert ext in kSupportedList, f"{media_file} format is not supported"

        media_type_key = media_type(ext)
        if media_type_key == "video_url":
            has_video = True

        print(f"Encoding {media_file} as base64...")
        base64_data = encode_media_base64(media_file)

        media_obj = {
            "type": media_type_key,
            media_type_key: {"url": f"data:{mime_type(ext)};base64,{base64_data}"}
        }
        content.append(media_obj)

    if has_video:
        assert len(media_files) == 1, "Only single video supported."

    headers = {
        "Authorization": f"Bearer {kApiKey}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    messages = [
        {"role": "system", "content": SERVEYE_PROMPT.strip() + "/think"},
        {"role": "user", "content": content},
    ]

    payload = {
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "messages": messages,
        "stream": stream,
        "model": "nvidia/nemotron-nano-12b-v2-vl",
    }

    print("Sending request to NVIDIA API...")
    response = requests.post(infer_url, headers=headers, json=payload, stream=stream)

    if stream:
        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))
    else:
        res = response.json()
        print(json.dumps(res, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python serveye_attribute_extract.py <media_file> <detection_type>")
        sys.exit(1)

    media_file = sys.argv[1]
    detection_type = sys.argv[2]  # car, body, or face
    chat_with_media(invoke_url, [media_file], detection_type, stream)
