import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

import matplotlib.pyplot as plt
from matplotlib import cm


# ---------------------------
# Build model same as training
# ---------------------------
def build_model(dropout=0.3):
    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1)
    )
    return model


# ---------------------------
# Eval transform
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
# Find last conv layer automatically
# ---------------------------
def find_last_conv_layer(model):
    last_name = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
            last_module = module

    if last_module is None:
        raise RuntimeError("No Conv2d layer found in model.")

    print(f"[INFO] Using last conv layer: {last_name}")
    return last_module


# ---------------------------
# Grad-CAM
# ---------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(self.forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, x):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x).view(-1)
        score = logits[0]  # binary logit for reject class
        score.backward()

        # activations: [1, C, H, W]
        # gradients:   [1, C, H, W]
        grads = self.gradients[0]         # [C, H, W]
        acts = self.activations[0]        # [C, H, W]

        weights = grads.mean(dim=(1, 2))  # [C]
        cam = torch.zeros(acts.shape[1:], device=acts.device)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, torch.sigmoid(score).item()


# ---------------------------
# Overlay heatmap
# ---------------------------
def overlay_heatmap_on_image(pil_img, cam, alpha=0.4):
    img = np.array(pil_img).astype(np.float32) / 255.0
    h, w = img.shape[:2]

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    cam_resized = np.array(cam_img).astype(np.float32) / 255.0

    heatmap = cm.jet(cam_resized)[..., :3]  # RGB
    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return img, cam_resized, overlay


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best_model.pt")
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--img-size", type=int, default=448)
    parser.add_argument("--out", default="gradcam_result.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading checkpoint...")
    ckpt = torch.load(args.model, map_location=device)

    saved_args = ckpt.get("args", {})
    dropout = saved_args.get("dropout", 0.3)
    img_size = saved_args.get("img_size", args.img_size)
    threshold = ckpt.get("threshold", 0.5)

    model = build_model(dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    transform = build_transform(img_size)

    pil_img = Image.open(args.image).convert("RGB")
    x = transform(pil_img).unsqueeze(0).to(device)

    target_layer = find_last_conv_layer(model)
    gradcam = GradCAM(model, target_layer)

    cam, prob_reject = gradcam.generate(x)
    gradcam.remove()

    original, cam_resized, overlay = overlay_heatmap_on_image(pil_img, cam, alpha=0.4)

    pred_label = "reject" if prob_reject >= threshold else "accept"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(original)
    axes[1].imshow(cam_resized, cmap="jet", alpha=0.6)
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\nprob_reject={prob_reject:.3f} | pred={pred_label} | thr={threshold:.2f}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved Grad-CAM to: {args.out}")
    print(f"[INFO] prob_reject={prob_reject:.4f}")
    print(f"[INFO] threshold={threshold:.4f}")
    print(f"[INFO] prediction={pred_label}")


if __name__ == "__main__":
    main()