"""
FORGE phase 2c — degradation curve. Runs locally (model + data already here).

Reuses the exact degrade() transforms from phase0's harness (resize, then
JPEG re-encode at a given quality) on the in-distribution test set, so the
web app's degradation chart reflects a real measurement instead of a mock.
"""
import csv
import json
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

PHASE2_DIR = Path(__file__).parent
PHASE1_OUT = PHASE2_DIR.parent / "phase1" / "output"
LOCAL_DATA_ROOT = PHASE1_OUT / "normalized"
OLD_PREFIX = "/kaggle/working/normalized/"
CKPT_PATH = PHASE2_DIR / "output" / "model_best.pt"
CALIB_PATH = PHASE2_DIR / "output" / "calibration.json"
IMG_SIZE = 224  # must match phase2_train.py's training resolution
DEVICE = torch.device("cpu")

LEVELS = [
    ("original", None),
    ("q95", 95),
    ("q75", 75),
    ("q50", 50),
    ("q25", 25),
]


def remap_path(p):
    return str(LOCAL_DATA_ROOT / p[len(OLD_PREFIX):]) if p.startswith(OLD_PREFIX) else p


def degrade(im, jpeg_quality=None):
    if jpeg_quality is None:
        return im
    buf = BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class DegradedDataset(Dataset):
    def __init__(self, rows, jpeg_quality):
        self.rows = rows
        self.jpeg_quality = jpeg_quality
        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        im = Image.open(r["path"]).convert("RGB")
        im = degrade(im, self.jpeg_quality)
        x = self.tf(im)
        y = 1.0 if r["label"] == "fake" else 0.0
        return x, y


def build_model():
    m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, 1)
    m.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    m.eval()
    return m


@torch.no_grad()
def accuracy(model, rows, jpeg_quality, T, batch_size=64):
    loader = DataLoader(DegradedDataset(rows, jpeg_quality), batch_size=batch_size, shuffle=False, num_workers=2)
    correct, n = 0, 0
    for x, y in loader:
        logits = model(x).squeeze(1) / T
        correct += ((logits > 0) == (y > 0.5)).sum().item()
        n += x.size(0)
    return correct / n


def main():
    with open(PHASE1_OUT / "splits.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["path"] = remap_path(r["path"])
    test_rows = [r for r in rows if r["split"] == "test"]

    T = 1.0
    if CALIB_PATH.exists():
        T = json.load(open(CALIB_PATH))["temperature"]

    model = build_model()
    results = []
    for name, quality in LEVELS:
        acc = accuracy(model, test_rows, quality, T)
        results.append((name, acc))
        print(f"{name}: {acc:.3f}")

    with open(PHASE2_DIR / "output" / "DEGRADATION_REPORT.md", "w") as f:
        f.write("# FORGE phase 2c — degradation curve\n\n")
        f.write("Measured on the in-distribution test set (n={}), JPEG re-encoded at each quality before model input.\n\n".format(len(test_rows)))
        f.write("| level | accuracy |\n|---|---:|\n")
        for name, acc in results:
            f.write(f"| {name} | {acc:.3f} |\n")

    with open(PHASE2_DIR / "output" / "degradation.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
