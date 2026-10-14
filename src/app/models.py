from pydantic import BaseModel, Field ,RootModel
from typing import List, Literal, Optional, Dict

class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class ValidateRequest(BaseModel):
    image_base64: str
    # bbox: BBox
    bbox: Optional[BBox] = None   # <-- bbox is now optional
    task: Literal["lpr", "lpd", "age", "gender"]   #restrict to only these options
    validator: Literal["openai", "gemini", "nvidia"]   #restrict to only these options


class NvidiaAttributesRequest(BaseModel):
    image_base64: str
    bbox: Optional[BBox] = None
    detection_type: Literal["car", "body", "face"]
    # optional: the validator string isn't necessary here since this endpoint
    # is dedicated to NVIDIA attributes extraction.

class ValidatorResult(BaseModel):
    label: str
    confidence: int = Field(..., ge=1, le=100)
    detail: Optional[str] = None  # Additional info if needed

class ValidateResponse(RootModel[Dict[str, ValidatorResult]]):
    root: Dict[str, ValidatorResult]