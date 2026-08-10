import base64
import json
import logging
from io import BytesIO
from typing import Dict, List

import httpx
from PIL import Image
from attribute_schemas import PERSON_ATTRIBUTE_GROUPS
from attribute_detector import _short_label

# Map for dynamic schema building (currently just person)
ATTRIBUTE_GROUPS = {"person": PERSON_ATTRIBUTE_GROUPS}

logger = logging.getLogger("vlm_detector")


class VLMAttributeDetector:
    """
    Attribute detector using Qwen-VL (or similar) via Ollama.
    Requests structured JSON output matching the attribute schemas.
    """

    def __init__(self, model_name: str = "qwen3-vl:2b", ollama_host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_host = ollama_host
        
        # Build prompt templates dynamically from schemas
        self.category_prompts = {}
        for cat, groups in ATTRIBUTE_GROUPS.items():
            prompt = (
                "You are an expert e-commerce and visual attribute cataloger. "
                "Analyze this image and extract the following attributes as a JSON object. "
                "Only return the JSON object, nothing else.\n\n"
                "Attributes to extract:\n"
            )
            for group in groups:
                if group.positive_idx is not None:
                    # Binary group
                    prompt += f"- \"{group.name}\": boolean (true if present, false if not)\n"
                else:
                    # Multi-class group
                    options = [_short_label(label) for label in group.labels]
                    prompt += f"- \"{group.name}\": string (choose exactly one of: {', '.join(options)})\n"
            
            self.category_prompts[cat] = prompt

    def detect(self, img: Image.Image, category: str = "person") -> Dict[str, dict]:
        """Detect attributes using the VLM."""
        if category not in self.category_prompts:
            raise ValueError(f"Unknown category: {category}")
            
        buffered = BytesIO()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": self.category_prompts[category],
                    "images": [img_b64]
                }
            ],
            "stream": False,
            "format": "json"
        }
        
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{self.ollama_host}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                
            content = data["message"]["content"]
            parsed = json.loads(content)
        except Exception as e:
            logger.error(f"VLM API failed or returned invalid JSON: {e}")
            parsed = {}
            
        # Format matching the output of SigLIP2 detector
        output = {}
        groups = ATTRIBUTE_GROUPS[category]
        for group in groups:
            val = parsed.get(group.name)
            
            if group.positive_idx is not None:
                # Binary attribute
                present = bool(val)
                label_idx = group.positive_idx if present else (1 - group.positive_idx)
                output[group.name] = {
                    "label": group.labels[label_idx],
                    "short": "✓ yes" if present else "✗ no",
                    "score": 1.0 if present else 0.0,
                    "present": present
                }
            else:
                # Multi-class attribute
                matched_label = group.labels[0]
                matched_short = _short_label(group.labels[0])
                
                if isinstance(val, str):
                    val_lower = val.lower().strip()
                    for label in group.labels:
                        short = _short_label(label)
                        if short.lower() == val_lower:
                            matched_label = label
                            matched_short = short
                            break
                            
                output[group.name] = {
                    "label": matched_label,
                    "short": matched_short,
                    "score": 1.0  # VLM doesn't give confidence scores easily in JSON mode
                }
                
        return output

    def detect_batch(self, imgs: List[Image.Image], category: str = "person") -> List[Dict[str, dict]]:
        """Run detection on a batch of images sequentially."""
        return [self.detect(img, category) for img in imgs]
