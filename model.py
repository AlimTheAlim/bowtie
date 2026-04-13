import os
import json
import math
import copy
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# =========================================================
# Utils
# =========================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_class_mapping(imagefolder_ds):
    class_to_idx = imagefolder_ds.class_to_idx
    if "reject" not in class_to_idx or "accept" not in class_to_idx:
        raise ValueError(
            f"Expected class folders named 'accept' and 'reject'. Found: {list(class_to_idx.keys())}"
        )
    return class_to_idx

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    # F2 emphasizes recall more than precision
    beta = 2.0
    if precision == 0 and recall == 0:
        f2 = 0.0
    else:
        f2 = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall + 1e-12)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp + 1e-12)
    return {
        "precision_reject": float(precision),
        "recall_reject": float(recall),
        "f1_reject": float(f1),
        "f2_reject": float(f2),
        "specificity_accept": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

def find_best_threshold(y_true, y_prob, min_precision=None):
    best_t = 0.5
    best_metrics = None
    best_score = -1.0

    for t in np.linspace(0.05, 0.95, 91):
        m = compute_metrics(y_true, y_prob, threshold=t)

        if min_precision is not None and m["precision_reject"] < min_precision:
            continue

        score = m["f2_reject"]  # prioritize reject recall
        if score > best_score:
            best_score = score
            best_t = float(t)
            best_metrics = m

    if best_metrics is None:
        best_metrics = compute_metrics(y_true, y_prob, threshold=0.5)
        best_t = 0.5

    return best_t, best_metrics

# =========================================================
# Loss
# =========================================================
class FocalLossBinary(nn.Module):
    """
    BCEWithLogits-based focal loss for binary classification.
    targets must be float tensor of shape [B].
    """
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1).float().to(logits.device)

        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=pos_weight
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal = self.alpha * ((1 - pt) ** self.gamma) * bce
        return focal.mean()

# =========================================================
# Data
# =========================================================
def build_transforms(img_size=448):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # keep low unless orientation truly irrelevant
        transforms.RandomRotation(degrees=7),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.03, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf

def make_datasets(data_dir, img_size):
    train_tf, eval_tf = build_transforms(img_size)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

    return train_ds, val_ds, test_ds

def make_weighted_sampler(train_ds):
    targets = np.array(train_ds.targets)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler, class_counts.tolist()

# =========================================================
# Model
# =========================================================
def build_model(dropout=0.3):
    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1)
    )
    return model

# =========================================================
# Train / Eval
# =========================================================
@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs = []
    labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).view(-1)
        p = torch.sigmoid(logits).cpu().numpy()

        probs.append(p)
        labels.append(y.numpy())

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    return labels, probs

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss = 0.0

    print(f"[DEBUG] train batches: {len(loader)}", flush=True)

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx % 10 == 0:
            print(f"[DEBUG] batch {batch_idx}/{len(loader)}", flush=True)

        x = x.to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(x).view(-1)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x).view(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)

    return running_loss / len(loader.dataset)

def evaluate_loss(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)
            logits = model(x).view(-1)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)

    return running_loss / len(loader.dataset)

# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="runs/reject_priority")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=448)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--focal-alpha", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pos-weight", type=float, default=4.0)
    parser.add_argument("--min-precision", type=float, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    pin_memory = use_amp

    print(f"[INFO] Device: {device}", flush=True)
    print("[INFO] Building datasets...", flush=True)
    train_ds, val_ds, test_ds = make_datasets(args.data_dir, args.img_size)

    class_to_idx = get_class_mapping(train_ds)
    accept_idx = class_to_idx["accept"]
    reject_idx = class_to_idx["reject"]

    # Force reject=1, accept=0 logic in reporting
    if reject_idx != 1 or accept_idx != 0:
        print(f"[WARN] Class mapping is {class_to_idx}.", flush=True)
        print("[WARN] ImageFolder sorts alphabetically, so accept=0 and reject=1 is expected.", flush=True)
        print("[WARN] If not, rename folders or adjust logic.", flush=True)

    print("[INFO] Building sampler...", flush=True)
    sampler, class_counts = make_weighted_sampler(train_ds)
    print(f"Train class counts [accept, reject-ish by index order]: {class_counts}", flush=True)
    print(f"Class mapping: {class_to_idx}", flush=True)

    print("[INFO] Building dataloaders...", flush=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory
    )

    print("[INFO] Building model...", flush=True)
    model = build_model(dropout=args.dropout).to(device)

    # freeze backbone at the start
    if args.freeze_backbone_epochs > 0:
        print("[INFO] Freezing backbone initially...", flush=True)
        for _, param in model.named_parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    print("[INFO] Building criterion...", flush=True)
    criterion = FocalLossBinary(
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
        pos_weight=args.pos_weight
    ).to(device)

    print("[INFO] Building optimizer/scheduler...", flush=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_state = None
    best_threshold = 0.5
    best_val_score = -1.0
    history = []

    print("[INFO] Starting training loop...", flush=True)

    for epoch in range(1, args.epochs + 1):
        print(f"[INFO] Epoch {epoch}/{args.epochs} started", flush=True)

        if epoch == args.freeze_backbone_epochs + 1 and args.freeze_backbone_epochs > 0:
            print("[INFO] Unfreezing full model.", flush=True)
            for param in model.parameters():
                param.requires_grad = True

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr * 0.5,
                weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - args.freeze_backbone_epochs)
            )

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        print(f"[INFO] Epoch {epoch}: train finished", flush=True)

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        print(f"[INFO] Epoch {epoch}: val loss computed", flush=True)

        y_val, p_val = predict_probs(model, val_loader, device)
        threshold, val_metrics = find_best_threshold(
            y_val, p_val, min_precision=args.min_precision
        )

        score = val_metrics["f2_reject"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "threshold": threshold,
            **val_metrics
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"thr={threshold:.2f} | "
            f"reject_precision={val_metrics['precision_reject']:.4f} | "
            f"reject_recall={val_metrics['recall_reject']:.4f} | "
            f"reject_f2={val_metrics['f2_reject']:.4f} | "
            f"TP={val_metrics['tp']} FP={val_metrics['fp']} FN={val_metrics['fn']} TN={val_metrics['tn']}",
            flush=True
        )

        if score > best_val_score:
            best_val_score = score
            best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_state_dict": best_state,
                    "threshold": best_threshold,
                    "class_to_idx": class_to_idx,
                    "args": vars(args),
                    "best_val_metrics": val_metrics,
                },
                os.path.join(args.out_dir, "best_model.pt")
            )
            print(f"[INFO] Saved new best model at epoch {epoch}", flush=True)

        scheduler.step()

    print("[INFO] Training finished. Evaluating on test...", flush=True)

    # Final eval on test
    model.load_state_dict(best_state)
    y_test, p_test = predict_probs(model, test_loader, device)
    test_metrics = compute_metrics(y_test, p_test, threshold=best_threshold)

    print("\n=== FINAL TEST ===", flush=True)
    print(f"Best threshold: {best_threshold:.2f}", flush=True)
    for k, v in test_metrics.items():
        print(f"{k}: {v}", flush=True)

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump(
            {
                "best_threshold": best_threshold,
                "test_metrics": test_metrics,
            },
            f,
            indent=2
        )

    print("[INFO] Saved history.json and test_metrics.json", flush=True)

if __name__ == "__main__":
    main()