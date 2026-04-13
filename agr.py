import os
import random
from PIL import Image, ImageEnhance, ImageOps

# =========================
# CONFIG
# =========================
DATASET_DIR = r"D:\CSCI-484\bowtie\LA-UR-25-28525-bowtie\dataset"  
AUGS_PER_IMAGE = 4                    # how many new copies per original
AUGMENT_SPLITS = ["train"]            # usually only train
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Safe transforms for tiny defects
MAX_ROTATION = 10                     # degrees
MAX_SHIFT_RATIO = 0.03                # 3% of width/height
MIN_SCALE = 0.95
MAX_SCALE = 1.05
BRIGHTNESS_RANGE = (0.92, 1.08)
CONTRAST_RANGE = (0.92, 1.08)

ENABLE_HFLIP = True
ENABLE_VFLIP = False                  # set True only if vertical orientation does not matter

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# =========================
# HELPERS
# =========================
def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTS


def list_original_images(folder: str):
    """
    Only augment original files, not already-augmented ones.
    """
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if not is_image_file(name):
            continue
        if "_aug" in name:
            continue
        files.append(name)
    return files


def apply_small_rotation(img: Image.Image) -> Image.Image:
    angle = random.uniform(-MAX_ROTATION, MAX_ROTATION)
    return img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)


def apply_brightness_contrast(img: Image.Image) -> Image.Image:
    brightness_factor = random.uniform(*BRIGHTNESS_RANGE)
    contrast_factor = random.uniform(*CONTRAST_RANGE)

    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    return img


def apply_flip(img: Image.Image) -> Image.Image:
    if ENABLE_HFLIP and random.random() < 0.5:
        img = ImageOps.mirror(img)
    if ENABLE_VFLIP and random.random() < 0.2:
        img = ImageOps.flip(img)
    return img


def apply_shift(img: Image.Image) -> Image.Image:
    w, h = img.size
    max_dx = int(w * MAX_SHIFT_RATIO)
    max_dy = int(h * MAX_SHIFT_RATIO)

    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)

    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, 0, dx, 0, 1, dy),
        resample=Image.Resampling.BICUBIC
    )


def apply_scale(img: Image.Image) -> Image.Image:
    """
    Slight zoom in/out, then crop/pad back to original size.
    """
    w, h = img.size
    scale = random.uniform(MIN_SCALE, MAX_SCALE)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    if scale >= 1.0:
        # crop center back to original size
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        return resized.crop((left, top, left + w, top + h))
    else:
        # paste centered on black canvas
        canvas = Image.new(img.mode, (w, h))
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        canvas.paste(resized, (left, top))
        return canvas


def augment_image(img: Image.Image) -> Image.Image:
    """
    Apply a few light augmentations in random order.
    """
    operations = [
        apply_small_rotation,
        apply_brightness_contrast,
        apply_shift,
        apply_scale,
        apply_flip,
    ]
    random.shuffle(operations)

    out = img.copy()
    for op in operations:
        # Flip always decides internally; others are applied
        if op is apply_flip:
            out = op(out)
        else:
            out = op(out)
    return out


def save_augmented(img: Image.Image, original_path: str, aug_index: int):
    folder = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    stem, ext = os.path.splitext(filename)

    new_name = f"{stem}_aug{aug_index}{ext}"
    save_path = os.path.join(folder, new_name)

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    img.save(save_path, quality=95)
    return save_path


# =========================
# MAIN
# =========================
def process_folder(folder: str):
    originals = list_original_images(folder)
    if not originals:
        print(f"[SKIP] No original images in: {folder}")
        return

    print(f"[INFO] Found {len(originals)} originals in {folder}")

    created = 0

    for filename in originals:
        path = os.path.join(folder, filename)

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")

                for i in range(1, AUGS_PER_IMAGE + 1):
                    aug_img = augment_image(img)
                    save_augmented(aug_img, path, i)
                    created += 1

        except Exception as e:
            print(f"[ERROR] Failed on {path}: {e}")

    print(f"[DONE] Created {created} augmented images in {folder}")


def main():
    for split in AUGMENT_SPLITS:
        split_path = os.path.join(DATASET_DIR, split)
        if not os.path.isdir(split_path):
            print(f"[WARN] Missing split folder: {split_path}")
            continue

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                process_folder(class_path)

    print("All done.")


if __name__ == "__main__":
    main()