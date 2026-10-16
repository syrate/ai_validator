# import os
# from PIL import Image
# import io
# import google.generativeai as gemini
# import json
import os
import re
import cv2
import json
import ultralytics
from anyio import to_thread
from google import genai
from google.genai import types
from PIL import Image
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.plotting import Annotator, colors
from src.app.validators.base_validator import BaseValidator
from src.app.validators.nvidia_validator import SERVEYE_PROMPT_ATTRIBUTES
from src.app.settings import settings
google_api_key = settings.google_api_key


class GeminiValidator(BaseValidator):
    def __init__(self):
        # super().__init__(base_model="gemini-2.5-pro")
        self.client = genai.Client(api_key=google_api_key)
        self.base_model ="gemini-2.5-pro"

    def _call_gemini_attributes(self, image: Image.Image, detection_type: str) -> dict:
        """Call Gemini to extract car/body/face attributes.

        Mirrors NvidiaValidator's __call_qwen contract so both are interchangeable
        behind VALIDATORS["attributes"]: returns
        {"success": bool, "content": {"detection_type": ..., "attributes": {...}}}
        on success, or {"success": False, "error": ...} on failure.
        """
        prompt = f"{SERVEYE_PROMPT_ATTRIBUTES.strip()}\n\ndetection_type: \"{detection_type}\""

        try:
            response = self.client.models.generate_content(
                model=self.base_model,
                contents=[prompt, image],
                config=types.GenerateContentConfig(temperature=0.3),
            )
            raw_output = response.text
            if not raw_output:
                return {"success": False, "error": "Empty response from Gemini"}

            cleaned = re.sub(r"```(?:json)?\s*", "", raw_output)
            cleaned = cleaned.replace("```", "").strip()

            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                return {"success": False, "error": f"Invalid JSON from Gemini: {cleaned[:200]}"}

            attributes = parsed.get("attributs", parsed.get("attributes", {}))
            return {
                "success": True,
                "content": {
                    "detection_type": parsed.get("detection_type", detection_type),
                    "attributes": attributes,
                },
            }
        except Exception as e:
            return {"success": False, "error": f"Unexpected error calling Gemini: {e}"}

    def generate_safety_recommendations(self, metrics: dict) -> dict:
        """Call Gemini to turn aggregated safety-performance metrics into short,
        actionable recommendations. `metrics` is the same payload the Java backend
        computes for the Safety Module radar chart (per-category violation counts
        and scores for the current vs previous period).

        Returns {"success": bool, "recommendations": [{"title":..., "description":...}]}
        on success, or {"success": False, "error": ...} on failure.
        """
        prompt = (
            "You are a workplace safety analyst for a video-surveillance platform. "
            "Given the following real safety performance metrics for the period "
            f"{metrics.get('period')}, write 3 to 6 short, specific, actionable "
            "recommendations in French for a site safety manager. Each recommendation "
            "must reference a concrete category and its violation count/trend from the "
            "data below - do not invent generic advice unrelated to the data. If a "
            "category has zero violations, do not mention it.\n\n"
            f"Overall safety score: {metrics.get('overallScore')}/100 "
            f"(previous period: {metrics.get('previousOverallScore')}/100)\n\n"
            "Per-category data (category, label, current violations, previous violations, "
            "current score, previous score):\n"
        )
        for cat in metrics.get("categories", []):
            prompt += (
                f"- {cat.get('label')}: {cat.get('currentViolations')} violations "
                f"(previous period: {cat.get('previousViolations')}), "
                f"score {cat.get('currentScore')}/100 "
                f"(previous: {cat.get('previousScore')}/100)\n"
            )
        prompt += (
            "\nReturn ONLY a JSON array, no markdown, no commentary, each element shaped as:\n"
            '{"title": "<short title, max 6 words>", "description": "<one sentence>"}'
        )

        try:
            response = self.client.models.generate_content(
                model=self.base_model,
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0.4),
            )
            raw_output = response.text
            if not raw_output:
                return {"success": False, "error": "Empty response from Gemini"}

            cleaned = re.sub(r"```(?:json)?\s*", "", raw_output)
            cleaned = cleaned.replace("```", "").strip()

            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                return {"success": False, "error": f"Invalid JSON from Gemini: {cleaned[:200]}"}

            if not isinstance(parsed, list):
                return {"success": False, "error": "Gemini did not return a JSON array"}

            recommendations = [
                {"title": item.get("title", ""), "description": item.get("description", "")}
                for item in parsed
                if isinstance(item, dict) and item.get("title") and item.get("description")
            ]
            return {"success": True, "recommendations": recommendations}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error calling Gemini: {e}"}

    def attributes_car(self, image: Image.Image) -> dict:
        return self._call_gemini_attributes(image, "car")

    def attributes_body(self, image: Image.Image) -> dict:
        return self._call_gemini_attributes(image, "body")

    def attributes_face(self, image: Image.Image) -> dict:
        return self._call_gemini_attributes(image, "face")

    async def validate(self, image: Image.Image,task: str) -> dict:
        """
        Dispatch validation based on the task: 'lpr', 'gender', or 'age'.
        """
        if task == "lpr":
            return await self._validate_lpr(image)
        # elif task == "gender":
        #     return self._validate_gender(image, class_name)
        # elif task == "age":
        #     return self._validate_age(image, class_name)
        else:
            raise ValueError(f" Unsupported task: {task}. Allowed: lpr, gender, age")
        
    async def _validate_lpr(self, image: Image.Image) -> dict:

        prompt = """
        You are tasked with detecting and extracting 2D bounding boxes for all visible vehicle license plates in the image.

        IMPORTANT:
        - All vehicles and license plates in these images are from Senegal.
        - Only check that the plate number matches the Senegalese license plate format.
        - DO NOT reject or skip any plate based on country, language, or alphabet—just the format.

        Senegalese license plates use only uppercase Latin letters (A-Z) and digits (0-9).
        They NEVER contain Cyrillic, lowercase letters, accents, or special symbols.

        Valid formats:
        - AA123BB  (two letters, three digits, two letters)
        - AA1234BB (two letters, four digits, two letters)
        - DK1234A  (two letters, four digits, one letter)
        - DK1234BB (two letters, four digits, two letters)

        Rules:
        1. Use OCR to extract the plate number.
        2. Return a result ONLY if the plate matches this regex: ^[A-Z]{2}[0-9]{3,4}[A-Z]{1,2}$
        3. Remove any spaces or special characters.
        4. Ignore unclear or unreadable plates.
        5. Do not output any country, explanation, or commentary—just the result.
        """
        
        output_prompt = """
        Return ONLY a JSON array where each element is:
        {
        "box_2d": [x_min, y_min, x_max, y_max],
        "label": "<PLATE>",
        "confidence": <1-100>
        }
        - 'label' must match: ^[A-Z]{2}[0-9]{3,4}[A-Z]{1,2}$
        - Remove spaces and special characters.
        - No extra text, no explanations.
        """

        
        try:
            # Step 1: Run inference
            # results = await to_thread.run_sync(self.run_inference,self.client, image, self.base_model, prompt + output_prompt)
            results = await self.run_inference(self.client, image, self.base_model, prompt + output_prompt)

            # Step 2: Validate response structure
            if not isinstance(results, dict) or not results.get("success", True):
                return {"success": False, "error": results.get("error", "Unknown LLM error"), "detections": []}

            raw_output = results.get("output")
            if not raw_output:
                return {"success": False, "error": "Empty response from LLM", "detections": []}

            # Step 3: Parse JSON safely
            try:
                cln_results = self.clean_results(raw_output)
            except (json.JSONDecodeError, TypeError) as e:
                return {"success": False, "error": f"Invalid JSON from LLM: {e}", "detections": []}
            print("Cleaned results:", cln_results)
            
            # Step 4: Validate detection list format
            if not isinstance(cln_results, list):
                return {"success": False, "error": "LLM returned non-list result", "detections": []}

            # ✅ Handle case: No detections
            if len(cln_results) == 0:
                return {"success": True, "detections": [], "annotated_image": image}

            # Step 5: Validate detection objects
            if not all(isinstance(item, dict) and {"box_2d", "label", "confidence"}.issubset(item.keys()) for item in cln_results):
                return {"success": False, "error": "Malformed detection objects from LLM", "detections": []}

            # Step 6: Annotate image with detections
            h, w = image.size
            annotator = Annotator(image)
            for idx, item in enumerate(cln_results):
                y1, x1, y2, x2 = item["box_2d"]
                y1, x1, y2, x2 = (y1 / 1000 * h, x1 / 1000 * w, y2 / 1000 * h, x2 / 1000 * w)
                annotator.box_label([x1, y1, x2, y2], label=item["label"], color=colors(idx, True))

            # ✅ Return successful result
            return {"success": True, "detections": cln_results, "annotated_image": annotator.result()}

        except Exception as e:
            # ✅ Catch unexpected runtime errors
            return {"success": False, "error": f"Unexpected error: {str(e)}", "detections": []}
    
    
    
    def _validate_lpd(self, image: Image.Image) -> dict:
        prompt = """
        Detect 2D bounding boxes for all visible license plates.
        """
        
        output_prompt = """
        Return only 'box_2d',  'label' (plate number from OCR) , and 'confidence' (1-100).
        Ignore unclear plates. 
        No extra text.
        Use OCR to extract the plate number.
        Make sure there in no space in the plate number, and no special characters, only number and letters only
           
        """
        
        h, w = image.size
        results = self.inference(image, prompt + output_prompt)
        
        cln_results = json.loads(self.clean_results(results))
        print("Cleaned results:", cln_results)
        annotator = Annotator(image)
        for idx, item in enumerate(cln_results):
            y1, x1, y2, x2 = item["box_2d"]
            y1, x1, y2, x2 = (y1 / 1000 * h, x1 / 1000 * w, y2 / 1000 * h, x2 / 1000 * w)
            
            # if x1 > x2:
            #     x1, x2 = x2, x1  # Swap x-coordinates if needed
            # if y1 > y2:
            #     y1, y2 = y2, y1  # Swap y-coordinates if needed
            
            annotator.box_label([x1, y1, x2, y2], label=item["label"], color=colors(idx, True))

        return {"detections": cln_results[0], "annotated_image": annotator.result()}

    