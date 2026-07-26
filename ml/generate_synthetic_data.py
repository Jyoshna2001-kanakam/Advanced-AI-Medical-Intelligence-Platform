"""
generate_synthetic_data.py
---------------------------
Generates a small synthetic "chest X-ray" style image dataset so the full
training -> inference -> Grad-CAM -> API pipeline can be run and verified
end-to-end without needing to download a large external dataset first.

For REAL use, replace the contents of data/train and data/val with the
public "Chest X-Ray Images (Pneumonia)" dataset (Kermany et al.), keeping
the same folder layout:

    data/
      train/
        NORMAL/
        PNEUMONIA/
      val/
        NORMAL/
        PNEUMONIA/

The synthetic generator below creates grayscale 224x224 images with two
distinct statistical textures (smooth vs. speckled + opacity blobs) so the
model has a genuine (if simplified) pattern to learn, which keeps the demo
training run honest rather than trivial/random.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RNG = np.random.default_rng(42)


def make_normal_image(size=224):
    """Smooth lung-field-like image: soft gradient + mild noise, no opacities."""
    base = RNG.normal(loc=140, scale=8, size=(size, size))
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=4))

    draw = ImageDraw.Draw(img)
    # faint symmetric "rib cage" arcs
    for i in range(4):
        y = 40 + i * 35
        draw.arc([20, y, size - 20, y + 160], start=200, end=340, fill=110, width=2)
    return img


def make_pneumonia_image(size=224):
    """Speckled image with localized dense opacity blobs (simulated infiltrates)."""
    base = RNG.normal(loc=130, scale=14, size=(size, size))
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=2))

    draw = ImageDraw.Draw(img)
    for i in range(4):
        y = 40 + i * 35
        draw.arc([20, y, size - 20, y + 160], start=200, end=340, fill=100, width=2)

    # Add 2-4 bright dense "opacity" blobs concentrated on one side (asymmetry
    # is a real hallmark clinicians look for in pneumonia consolidation)
    n_blobs = RNG.integers(2, 5)
    side_bias = RNG.choice([1, -1])
    for _ in range(n_blobs):
        cx = size // 2 + side_bias * RNG.integers(20, 70)
        cy = RNG.integers(70, size - 50)
        r = RNG.integers(15, 35)
        blob = Image.new("L", (r * 2, r * 2), 0)
        bd = ImageDraw.Draw(blob)
        bd.ellipse([0, 0, r * 2, r * 2], fill=int(RNG.integers(190, 230)))
        blob = blob.filter(ImageFilter.GaussianBlur(radius=r / 3))
        img.paste(Image.composite(blob, img.crop((cx - r, cy - r, cx + r, cy + r)), blob),
                  (cx - r, cy - r))
    return img


def build_dataset(root="data", n_train=120, n_val=30):
    classes = {"NORMAL": make_normal_image, "PNEUMONIA": make_pneumonia_image}
    for split, count in [("train", n_train), ("val", n_val)]:
        for cls, fn in classes.items():
            out_dir = os.path.join(root, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(count):
                img = fn()
                img.convert("RGB").save(os.path.join(out_dir, f"{cls.lower()}_{i:04d}.png"))
        print(f"[ok] {split}: {count} images per class written to {root}/{split}/")


if __name__ == "__main__":
    build_dataset()
    print("Synthetic dataset generation complete.")
