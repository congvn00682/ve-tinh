"""Run top-k inference for one RGB aerial image."""

from __future__ import annotations

import argparse
import json

import torch
from PIL import Image

from .data import build_transforms
from .models import create_model
from .runtime import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    # Only load checkpoints produced by this project or another trusted source.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = list(checkpoint["class_names"])
    model = create_model(
        checkpoint["architecture"],
        len(class_names),
        load_pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    _, transform = build_transforms(int(checkpoint["image_size"]))
    with Image.open(args.image) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        temperature = float(checkpoint.get("temperature", 1.0))
        probabilities = torch.softmax(logits / temperature, dim=1)[0]

    count = min(args.top_k, len(class_names))
    scores, indices = probabilities.topk(count)
    result = {
        "image": args.image,
        "architecture": checkpoint["architecture"],
        "top_k": [
            {"label": class_names[index], "confidence": float(score)}
            for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist())
        ],
        "warning": "Confidence is not guaranteed to be calibrated under domain shift.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
