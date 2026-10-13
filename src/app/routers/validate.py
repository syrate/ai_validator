import os

from fastapi import APIRouter, HTTPException, Depends
from anyio import to_thread
from src.app.models import ValidateRequest, ValidateResponse, ValidatorResult, NvidiaAttributesRequest
from src.app.utils.image_utils import decode_image, crop_image
from src.app.validators.openai_validator import OpenAIValidator
from src.app.validators.gemini_validator import GeminiValidator
from src.app.validators.nvidia_validator import NvidiaValidator
from src.app.deps.auth import api_key_auth

router = APIRouter()

# Which VLM backend handles /validate/attributes (car/body/face extraction).
# "qwen" (default, unchanged behavior) or "gemini".
ATTRIBUTES_VLM_PROVIDER = os.getenv("ATTRIBUTES_VLM_PROVIDER", "qwen").lower()

_openai_validator = OpenAIValidator()
_gemini_validator = GeminiValidator()
_nvidia_validator = NvidiaValidator()

VALIDATORS = {
    "openai": _openai_validator,
    "gemini": _gemini_validator,
    "nvidia": _nvidia_validator,
    "attributes": _gemini_validator if ATTRIBUTES_VLM_PROVIDER == "gemini" else _nvidia_validator,
}

@router.post("/validate", response_model=ValidateResponse, dependencies=[Depends(api_key_auth)])
async def validate(req: ValidateRequest):
    try:
        # decode/crop on a worker thread if it's CPU-heavy (Pillow)
        try:
            img = await to_thread.run_sync(decode_image, req.image_base64)
            crop = await to_thread.run_sync(crop_image, img, req.bbox) if req.bbox is not None else img
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

        name = req.validator
        if name not in VALIDATORS:
            print(f"Unknown validator: {name}") 
            raise HTTPException(status_code=400, detail=f"Unknown validator: {name}")

        # If validator is sync, offload to thread; if you implement .validate_async, call that instead.
        try:
            # val = await to_thread.run_sync(VALIDATORS[name].validate, crop, req.task)
            val = await VALIDATORS[name].validate(crop, req.task)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Validator execution error: {str(e)}")

        if not isinstance(val, dict) or not val.get("success", True):
            err = val.get("error", "Validator returned an error") if isinstance(val, dict) else "Invalid validator response"
            raise HTTPException(status_code=500, detail=err)

        detections = val.get("detections", [])
        if not isinstance(detections, list):
            raise HTTPException(status_code=500, detail="Validator returned detections in invalid format")

        if len(detections) == 0:
            return ValidateResponse({name: ValidatorResult(label="0", confidence=1, detail="No License plate detected")})

        det = detections[0]
        if not all(k in det for k in ["label", "confidence"]):
            raise HTTPException(status_code=500, detail="Malformed detection result from validator")

        try:
            conf10 = max(1, min(int(det["confidence"]), 100))
        except (ValueError, TypeError):
            raise HTTPException(status_code=500, detail="Invalid confidence value in detection result")

        return ValidateResponse({name: ValidatorResult(label=det["label"], confidence=conf10 , detail="License plate detected")})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")


@router.post("/validate/attributes", dependencies=[Depends(api_key_auth)])
async def validate_nvidia_attributes(req: NvidiaAttributesRequest):
    """Endpoint to extract visual attributes (car/body/face) using NVIDIA backend.

    Returns the raw attributes JSON returned by the Nvidia service wrapped in a
    simple envelope: {"success": True, "attributes": {...}} or an HTTP error.
    """
    try:
        try:
            img = await to_thread.run_sync(decode_image, req.image_base64)
            crop = await to_thread.run_sync(crop_image, img, req.bbox) if req.bbox is not None else img
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

        validator = VALIDATORS.get("attributes")
        if validator is None:
            raise HTTPException(status_code=500, detail="Nvidia validator not configured")

        # Choose specific method per detection_type to allow different handling per type
        method_map = {
            "car": validator.attributes_car,
            "body": validator.attributes_body,
            "face": validator.attributes_face,
        }

        if req.detection_type not in method_map:
            raise HTTPException(status_code=400, detail=f"Unsupported detection_type: {req.detection_type}")

        # call the specific attributes extractor in a worker thread
        result = await to_thread.run_sync(method_map[req.detection_type], crop)

        if not isinstance(result, dict) or not result.get("success", False):
            err = result.get("error", "Validator returned an error") if isinstance(result, dict) else "Invalid validator response"
            raise HTTPException(status_code=500, detail=err)

        # return raw attributes
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")
