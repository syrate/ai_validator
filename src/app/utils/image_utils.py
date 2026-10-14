import base64
import io
from PIL import Image
from src.app.models import BBox


def decode_image(base64_str: str) -> Image.Image:
    print("Decoding image from base64 string")
    data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(data)).convert('RGB')


def crop_image(image: Image.Image, bbox: BBox) -> Image.Image:
    x1 = bbox.x
    y1 = bbox.y
    x2 = bbox.x + bbox.width
    y2 = bbox.y + bbox.height
    return image.crop((x1, y1, x2, y2))