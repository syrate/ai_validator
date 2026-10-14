import os
import requests
from PIL import Image
import io
import base64
import re
import json
from typing import Any, Dict, Optional

from openai import OpenAI
from pydantic import BaseModel, Field
import openai
from src.app.validators.base_validator import BaseValidator
from src.app.settings import settings
import instructor
from instructor.retry import InstructorRetryException
import time
nvidia_api_key = settings.nvidia_nim_api_key

# Pydantic attribute models (keep current implementation)
class CarAttributes(BaseModel):
    type_de_vehicule: str = Field("", alias="type_de_vehicule")
    marque: str = ""
    couleur: str = ""



class BodyAttributes(BaseModel):
    couleur_haut: str = ""
    type_haut: str = ""
    couleur_bas: str = ""
    type_bas: str = ""
    sac: str = ""
    parapluie: str = ""



class FaceAttributes(BaseModel):
    sexe: str = ""
    age: str = ""
    foulard: str = ""
    teint_de_peau: str = ""
    coiffure: str = ""
    barbe: str = ""
    moustache: str = ""
    chapeau: str = ""
    capuche: str = ""
    masque: str = ""
    lunettes: str = ""


# Response wrapper models for structured parsing
class CarResponse(BaseModel):
    detection_type: str
    attributes: CarAttributes



class BodyResponse(BaseModel):
    detection_type: str
    attributes: BodyAttributes



class FaceResponse(BaseModel):
    detection_type: str
    attributes: FaceAttributes



# Inlined SERVEYE_PROMPT (copied from nemotron.py) to avoid importing that module.
SERVEYE_PROMPT_ATTRIBUTES = """
Vous êtes **SurvEye** — un agent intelligent d’analyse vidéo conçu pour extraire des attributs visuels détaillés à partir d’entités détectées (car, body ou face) dans des séquences de vidéosurveillance.

Votre analyse doit tenir compte du contexte environnemental et culturel du Sénégal, en Afrique de l’Ouest (notamment à Dakar). Vos interprétations des couleurs, vêtements et apparences doivent donc refléter les styles, l’éclairage et les conditions locales.

Vous recevrez un paramètre nommé **detection_type**, dont la valeur sera "car", "body" ou "face".  
Vous devez extraire les attributs correspondants et les retourner **strictement au format JSON**.

Si un attribut ne peut pas être déterminé avec certitude, retournez une chaîne vide ("") pour ce champ.

⚠️ **RÈGLES IMPORTANTES :**
- Répondez **UNIQUEMENT en JSON** — sans texte ni explication.
- Incluez **UNIQUEMENT** les clés indiquées pour le type de détection correspondant.
- N’extrayez les attributs que pour le **detection_type** fourni. Ignorez les autres entités présentes.
- Ne renommez, n’ajoutez ni ne supprimez aucune clé.
- Pour tout attribut incertain ou non détecté, retournez "" ou false.
- Le JSON doit toujours inclure les champs `"detection_type"` et `"attributs"`.
- Utilisez "" à la place de null pour les valeurs manquantes.
- Conservez toutes les clés, même si leurs valeurs sont vides.
- Répondez uniquement avec un JSON valide.

=== Règles d’Extraction ===

1. Si detection_type = "car"
    - type_de_vehicule : parmi ["berline", "suv", "camion", "fourgon", "bus", "moto", "vélo", "autre"]
    - marque : marque principale du véhicule
    - couleur : couleur principale de la carrosserie (si plusieurs, "multi-couleur")

2. Si detection_type = "body"
    - couleur_haut : couleur principale du vêtement supérieur (si plusieurs, "multi-couleur")
    - type_haut : type de vêtement supérieur (ex. "chemise", "t-shirt", "robe")
    - couleur_bas : couleur principale du vêtement inférieur (si plusieurs, "multi-couleur")
    - type_bas : type de vêtement inférieur (ex. "pantalon", "jupe", "robe")
    - sac : "Oui" / "Non" ("" si inconnu)
    - parapluie : "Oui" / "Non" ("" si inconnu)

3. Si detection_type = "face"
    - sexe : "homme" / "femme"
    - âge : parmi ["<11", "11-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    - foulard : "Oui" / "Non" ("" si inconnu)
    - teint_de_peau : parmi ["clair", "brun moyen", "foncé"]
    - coiffure : parmi ["chauve", "peu de cheveux", "cheveux courts", "cheveux longs", ""]
    - barbe : "Oui" / "Non" ("" si inconnu)
    - moustache : "Oui" / "Non" ("" si inconnu)
    - chapeau : "Oui" / "Non" ("" si inconnu)
    - capuche : "Oui" / "Non" ("" si inconnu)
    - masque : "Oui" / "Non" ("" si inconnu)
    - lunettes : "Oui" / "Non" ("" si inconnu)
=== Format de Réponse JSON ===
{
  "detection_type": "<car|body|face>",
  "attributs": { ... }
}

"""

class NvidiaValidator(BaseValidator):
    def __init__(self):
        self.endpoint = settings.nvidia_nim_endpoint
        self.api_key = settings.nvidia_nim_api_key
        self.vlm_url = settings.vlm_url
        self.vlm_api_key = settings.vlm_api_key
        self.openai = OpenAI(base_url=self.endpoint, api_key=self.api_key)
        self.instructor_client = instructor.from_openai(self.openai)
        self.retry = [30,60,120,240]

    def _extract_content_text_from_completion(self, r: Any) -> Optional[str]:
        """Safely extract assistant textual content from various SDK response shapes."""
        try:
            if hasattr(r, "choices") and r.choices:
                choice = r.choices[0]
                message = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
                if message is None and isinstance(choice, dict):
                    message = choice.get("message")
                if isinstance(message, dict):
                    return message.get("content")
                return getattr(message, "content", None)
            elif isinstance(r, dict):
                if "choices" in r and r["choices"]:
                    return r["choices"][0].get("message", {}).get("content")
                return r.get("content")
        except Exception:
            return None
        return None

    def safe_chat_completion(self, client, **kwargs):
        """Make a chat completion with exponential backoff retry on rate limit and connection errors.
        
        Catches both:
        - Direct OpenAI exceptions (RateLimitError, APIConnectionError, APITimeoutError)
        - InstructorRetryException (when instructor exhausts its internal retries)
        """
        for attempt in range(5):
            try:
                return client.chat.completions.create(**kwargs)
            except InstructorRetryException as e:
                # Instructor's internal retries are exhausted
                # Extract the root cause from the exception details
                error_msg = str(e)
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    if attempt < 4:
                        print(f"InstructorRetryException: Rate limit (attempt {attempt + 1}/5) — retrying in {self.retry[attempt]}s...")
                        time.sleep(self.retry[attempt])
                    else:
                        print(f"InstructorRetryException: Rate limit on final attempt (attempt {attempt + 1}/5)")
                        raise
                else:
                    # Not a rate limit, don't retry other instructor errors
                    print(f"InstructorRetryException (non-rate-limit): {error_msg}")
                    raise
            except openai.RateLimitError as e:
                if attempt < 4:  # don't sleep after last attempt
                    print(f"RateLimitError (attempt {attempt + 1}/5) — retrying in {self.retry[attempt]}s...")
                    time.sleep(self.retry[attempt])
                else:
                    print(f"RateLimitError on final attempt (attempt {attempt + 1}/5)")
                    raise
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                if attempt < 4:
                    print(f"API connection/timeout error (attempt {attempt + 1}/5) — retrying in {self.retry[attempt]}s...")
                    time.sleep(self.retry[attempt])
                else:
                    print(f"API connection/timeout error on final attempt")
                    raise
        # should not reach here, but if we do raise explicitly
        raise openai.RateLimitError("Max retries (5) exceeded for chat.completions.create")

    def _normalize_using_pydantic(self, det: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Use pydantic models to drop unknown keys and ensure expected keys exist."""
        allowed_attrs = {
            "car": ["type_de_vehicule", "marque", "couleur"],
            "body": ["sexe", "coiffure", "chapeau", "couleur_haut", "type_haut", "couleur_bas", "type_bas", "sac", "parapluie"],
            "face": ["sexe", "age", "foulard", "teint_de_peau", "coiffure", "barbe", "moustache", "chapeau", "capuche", "masque", "lunettes"],
        }

        try:
            if det == "car":
                norm = CarAttributes.parse_obj(attrs)
                final = norm.dict(by_alias=True)
            elif det == "body":
                norm = BodyAttributes.parse_obj(attrs)
                final = norm.dict()
            elif det == "face":
                norm = FaceAttributes.parse_obj(attrs)
                final = norm.dict()
            else:
                final = {k: (attrs.get(k, "") if attrs.get(k, "") is not None else "") for k in allowed_attrs.get(det, [])}
        except Exception:
            final = {k: (attrs.get(k, "") if attrs.get(k, "") is not None else "") for k in allowed_attrs.get(det, [])}

        # ensure all allowed keys present
        for k in allowed_attrs.get(det, []):
            final.setdefault(k, "")
        return final

    def _call_nemotron(self, image: Image.Image, detection_type: str) -> dict:
        """Call the model with retry logic using create(..., response_model=...).

        Handles rate limits and connection errors with exponential backoff.
        Uses Pydantic response models to request structured output.
        """
        if not self.endpoint:
            return {"success": False, "error": "NVIDIA_NIM_ENDPOINT not configured"}

        try:
            # encode image as base64
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            user_content = [
                {"type": "text", "text": f"Extract attributes for detection_type='{detection_type}'"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
            messages = [
                {"role": "system", "content": SERVEYE_PROMPT_ATTRIBUTES.strip()},
                {"role": "user", "content": user_content},
            ]

            # Choose response model according to requested detection_type
            resp_model_map = {
                "car": CarResponse,
                "body": BodyResponse,
                "face": FaceResponse,
            }
            resp_model = resp_model_map.get(detection_type, BodyResponse)
            
            # Use safe_chat_completion for retry logic on rate limits
            print(f"Calling NVIDIA Nemotron with response_model: {resp_model.__name__} (detection_type='{detection_type}')\n")
            
            r = self.safe_chat_completion(
                self.instructor_client,
                model="nvidia/nemotron-nano-12b-v2-vl",
                messages=messages,
                max_tokens=2048,
                temperature=0.3,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stream=False,
                response_model=resp_model,
            )
            print(f"Successfully received response from NVIDIA: {r}")
            return {"success": True, "content": r}

        except openai.RateLimitError as e:
            return {"success": False, "error": f"Rate limit exceeded after retries: {str(e)}"}
        except InstructorRetryException as e:
            # Instructor's retries exhausted — likely rate limit or validation error
            return {"success": False, "error": f"Instructor retry exhausted: {str(e)}"}
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            return {"success": False, "error": f"API connection/timeout error: {str(e)}"}
        except openai.APIError as e:
            return {"success": False, "error": f"OpenAI API error: {str(e)}"}
        except Exception as e:
            import traceback
            print(f"Unexpected error in _call_nemotron: {traceback.format_exc()}")
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    def __call_qwen(self, image: Image.Image, detection_type: str) -> dict:
        """
        Call Qwen3-VL (Cloud Run) endpoint to extract attributes.

        This function mirrors the behaviour of vlm_client.get_attributes:
        - encodes the image as data:image/jpeg;base64,...
        - sends a small prompt depending on detection_type
        - cleans markdown fences from the model response and attempts to parse JSON
        - returns {"success": bool, "content": {"detection_type": ..., "attributes": {...}}}
        """
        # Determine Cloud Run URL / API key from env
        # vlm_url = os.getenv("VLM_URL")
        # api_key = os.getenv("VLM_API_KEY")
        if not self.vlm_url:
            return {"success": False, "error": "VLM endpoint not configured (set VLM_URL or VLM_CLOUD_RUN_URL)"}

        # encode image as base64
        try:
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            return {"success": False, "error": f"Image encoding failed: {e}"}

        # simple prompt builder (kept small and compatible with vlm_client._build_prompt)
        if detection_type == "face":
            prompt = (
                "Analysez ce visage et extrayez les attributs suivants.\n"
                "Répondez UNIQUEMENT en FRANÇAIS avec un objet JSON valide, sans balises Markdown ni explication :\n"
                "{\n"
                '  "sexe": "homme ou femme",\n'
                '  "age": "tranche d\'âge comme 25-34",\n'
                '  "teint_de_peau": "clair, brun moyen ou foncé",\n'
                '  "coiffure": "description de la coiffure",\n'
                '  "barbe": "Oui ou Non",\n'
                '  "moustache": "Oui ou Non",\n'
                '  "chapeau": "Oui ou Non",\n'
                '  "capuche": "Oui ou Non",\n'
                '  "masque": "Oui ou Non",\n'
                '  "lunettes": "Oui ou Non"\n'
                "}"
            )
        elif detection_type == "body":
            prompt = (
                "Analysez la tenue de cette personne et extrayez les attributs suivants.\n"
                "Répondez UNIQUEMENT en FRANÇAIS avec un objet JSON valide, sans balises Markdown ni explication :\n"
                "{\n"
                '  "couleur_haut": "couleur du vêtement supérieur",\n'
                '  "type_haut": "type de vêtement supérieur (ex. chemise, t-shirt, robe)",\n'
                '  "couleur_bas": "couleur du vêtement inférieur",\n'
                '  "type_bas": "type de vêtement inférieur (ex. pantalon, jupe)",\n'
                '  "sac": "Oui ou Non",\n'
                '  "parapluie": "Oui ou Non"\n'
                "}"
            )
        elif detection_type == "car":
            prompt = (
                "Analysez ce véhicule et extrayez les attributs suivants.\n"
                "Répondez UNIQUEMENT en FRANÇAIS avec un objet JSON valide, sans balises Markdown ni explication :\n"
                "{\n"
                '  "type_de_vehicule": "type du véhicule (ex. berline, suv, camion, moto, autre)",\n'
                '  "marque": "marque principale du véhicule",\n'
                '  "couleur": "couleur principale de la carrosserie"\n'
                "}"
            )
        else:
            prompt = "Analysez cette image et renvoyez une description JSON en français."

        payload = {
            "model": "Qwen/Qwen3-VL-2B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1
        }

        headers = {"Content-Type": "application/json"}
        if self.vlm_api_key:
            headers["Authorization"] = f"Bearer {self.vlm_api_key}"

        try:
            resp = requests.post(self.vlm_url, json=payload, headers=headers, timeout=300)
            resp.raise_for_status()
            resp_json = resp.json()

            if "choices" not in resp_json or len(resp_json["choices"]) == 0:
                print("❌ Unexpected VLM response format:", resp_json)
                return {"success": False, "error": "Unexpected VLM response format"}

            raw_content = resp_json["choices"][0].get("message", {}).get("content", "")
            # clean fenced code blocks like ```json ... ```
            cleaned = re.sub(r"```(?:json)?\s*", "", str(raw_content))
            cleaned = cleaned.replace("```", "").strip()

            try:
                attrs = json.loads(cleaned)
                return {
                    "success": True,
                    "content": {
                        "detection_type": detection_type,
                        "attributes": attrs
                    }
                }
            except json.JSONDecodeError:
                print("❌ Failed to parse JSON from VLM response. Cleaned content:\n", cleaned)
                return {
                    "success": False,
                    "content": {
                        "detection_type": detection_type,
                        "attributes": {}
                    }
                }

        except requests.exceptions.Timeout:
            return {"success": False, "error": "VLM request timed out"}
        except requests.exceptions.HTTPError as e:
            body = getattr(e, "response", None)
            body_txt = body.text[:200] if body is not None else ""
            print(f"HTTP error calling VLM: {e} — Response: {body_txt}")
            return {"success": False, "error": f"HTTP error calling VLM: {e}"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request error calling VLM: {e}"}
    
    
    def attributes_car(self, image: Image.Image) -> dict:
        return self.__call_qwen(image, "car")

    def attributes_body(self, image: Image.Image) -> dict:
        return self.__call_qwen(image, "body")

    def attributes_face(self, image: Image.Image) -> dict:
        return self.__call_qwen(image, "face")

    def validate(self, image: Image.Image, class_name: str) -> dict:
        if class_name == "car":
            return self.attributes_car(image)
        if class_name == "body":
            return self.attributes_body(image)
        if class_name == "face":
            return self.attributes_face(image)
        return {"success": False, "error": f"Unsupported class_name: {class_name}"}