import os
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms, models

from predict import build_model, build_transform, predict_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--img-size", type=int, default=448)
    parser.add_argument("--output-file", default="dataset_predictions.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading checkpoint...")
    ckpt = torch.load(args.model, map_location=device)

    threshold = ckpt["threshold"]

    model = build_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    transform = build_transform(args.img_size)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    rows = []

    for root, _, files in os.walk(args.input_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext not in image_exts:
                continue

            img_path = os.path.join(root, file)

            try:
                prob, label = predict_image(
                    model,
                    img_path,
                    transform,
                    device,
                    threshold,
                )

                rows.append((img_path, prob, label))
                print(f"{img_path} -> {label} ({prob:.4f})")

            except Exception as e:
                print(f"[ERROR] Failed on {img_path}: {e}")

    with open(args.output_file, "w") as f:
        f.write("image_path,prob_reject,prediction\n")
        for img_path, prob, label in rows:
            f.write(f"{img_path},{prob:.6f},{label}\n")

    print(f"\n[INFO] Saved predictions to {args.output_file}")


if __name__ == "__main__":
    main()