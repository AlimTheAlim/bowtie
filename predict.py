import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models


# ---------------------------
# Build model same as training
# ---------------------------
def build_model(dropout=0.3):
    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=dropout),
        torch.nn.Linear(in_features, 1)
    )

    return model


# ---------------------------
# Transform (must match eval)
# ---------------------------
def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ---------------------------
# Predict single image
# ---------------------------
def predict_image(model, img_path, transform, device, threshold):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x).view(-1)
        prob = torch.sigmoid(logits).item()

    label = "reject" if prob >= threshold else "accept"

    return prob, label


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--img-size", type=int, default=448)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading checkpoint...")
    ckpt = torch.load(args.model, map_location=device)

    threshold = ckpt["threshold"]
    class_to_idx = ckpt["class_to_idx"]

    print("Threshold:", threshold)
    print("Classes:", class_to_idx)

    model = build_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    transform = build_transform(args.img_size)

    prob, label = predict_image(
        model,
        args.image,
        transform,
        device,
        threshold,
    )

    print("Prob reject:", prob)
    print("Prediction:", label)


if __name__ == "__main__":
    main()