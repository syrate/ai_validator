import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

project_root = Path(__file__).resolve().parent.parent
print("Project root:", project_root)

# Keep the endpoint as requested
URL = "http://localhost:8000/validate/attributes"
# URL = "https://ai-validator-654942414948.europe-west9.run.app/validate/attributes"

def is_image_file(p: Path, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tiff")) -> bool:
    return p.is_file() and p.suffix.lower() in extensions

def encode_image_to_base64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_validator(image_b64: str, detection_type: str, bbox: dict = None, timeout: int = 300) -> dict:
    payload = {
        "image_base64": image_b64,
        "detection_type": detection_type
    }
    if bbox:
        payload["bbox"] = bbox
    headers = {}
    api_key = os.environ.get("VALIDATOR_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        r = requests.post(URL, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def format_attributes_for_overlay(resp: dict) -> str:
    if not resp.get("success"):
        return f"ERROR: {resp.get('error')}"
    content = resp.get("content", {})
    attrs = content.get("attributes") if isinstance(content.get("attributes"), dict) else content
    if not attrs:
        return json.dumps(content, ensure_ascii=False)
    # produce multiline "key: value"
    lines = []
    if isinstance(attrs, dict):
        for k, v in attrs.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append(str(attrs))
    return "\n".join(lines)

def draw_text_overlay(img: Image.Image, text: str, padding=8, text_color=(255, 215, 0), stroke_width=2, stroke_fill=(0, 0, 0)) -> Image.Image:
    """
    Draw multiline text at the bottom-right corner of the image without a background rectangle.
    Uses a colored fill and a stroke (outline) for readability.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    lines = text.splitlines() or [""]
    widths = []
    heights = []
    for line in lines:
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except AttributeError:
            try:
                w, h = font.getsize(line)
            except Exception:
                w = max(10, len(line) * 6)
                h = 12
        widths.append(w)
        heights.append(h)

    # compute block size
    block_w = max(widths) if widths else 0
    block_h = sum(heights) + (len(lines) - 1) * 2

    # position at bottom-right with padding from edges
    img_w, img_h = img.size
    start_x = max(padding, img_w - block_w - padding * 2)
    start_y = max(padding, img_h - block_h - padding * 2)

    x = start_x + padding
    y = start_y + padding
    for line, h in zip(lines, heights):
        try:
            draw.text((x, y), line, font=font, fill=text_color, stroke_width=stroke_width, stroke_fill=stroke_fill)
        except TypeError:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
            for dx, dy in offsets:
                draw.text((x + dx, y + dy), line, font=font, fill=stroke_fill)
            draw.text((x, y), line, font=font, fill=text_color)
        y += h + 2

    return img

def process_folder(input_dir: Path, output_dir: Path, detection_type: str, bbox: dict = None):
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in input_dir.iterdir() if is_image_file(p)])
    if not files:
        print(f"No image files found in {input_dir}")
        return

    for idx, img_path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Processing {img_path.name} ...")
        try:
            b64 = encode_image_to_base64(img_path)
            resp = call_validator(b64, detection_type, bbox=bbox)
            overlay_text = format_attributes_for_overlay(resp)
            print(overlay_text)

            with Image.open(img_path) as img:
                annotated = draw_text_overlay(img, overlay_text)

                out_path = output_dir / img_path.name
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # determine format for saving
                ext = out_path.suffix.lower()
                fmt = None
                if ext in (".jpg", ".jpeg"):
                    fmt = "JPEG"
                    annotated_to_save = annotated.convert("RGB")
                else:
                    fmt = "PNG" if ext == ".png" else None
                    annotated_to_save = annotated

                try:
                    # save using explicit path string and format
                    if fmt:
                        annotated_to_save.save(out_path.as_posix(), format=fmt)
                    else:
                        annotated_to_save.save(out_path.as_posix())
                    # close images to release file handles
                    try:
                        annotated_to_save.close()
                    except Exception:
                        pass
                    try:
                        annotated.close()
                    except Exception:
                        pass

                    print(f"Saved annotated image to {out_path}")
                except Exception as save_err:
                    print(f"Failed to save {out_path}: {save_err}", file=sys.stderr)
        except Exception as e:
            print(f"Failed processing {img_path.name}: {e}", file=sys.stderr)

def parse_bbox(s: str):
    # bbox format: x,y,width,height
    try:
        parts = [int(x) for x in s.split(",")]
        if len(parts) != 4:
            raise ValueError("bbox must have 4 integers")
        return {"x": parts[0], "y": parts[1], "width": parts[2], "height": parts[3]}
    except Exception:
        raise

def main():
    parser = argparse.ArgumentParser(description="Batch validate images and save annotated output.")
    parser.add_argument("--input-dir", "-i", default=str(project_root / "data" / "attributes"), help="Input folder with images")
    parser.add_argument("--output-dir", "-o", default=str(project_root / "out_attributes"), help="Output folder for annotated images")
    parser.add_argument("--detection-type", "-t", default="car", help="detection_type to send (e.g. body, car, face)")
    parser.add_argument("--bbox", help="Optional bbox as x,y,width,height")
    args = parser.parse_args()

    bbox = None
    if args.bbox:
        bbox = parse_bbox(args.bbox)

    process_folder(Path(args.input_dir), Path(args.output_dir), args.detection_type, bbox=bbox)

if __name__ == "__main__":
    main()