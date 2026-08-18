"""
FORGE -- the practical shipping model. Trains on ALL generator families.

Every earlier phase-2 script deliberately hid two generator families
(StyleGAN3, SFHQ-T2I) from training so the project could measure "what
happens with a generator you've never seen" -- a research question about
honesty, which produced the leave-one-generator-out numbers reported on the
site. That handicap is why StyleGAN3 scored 0.000 there: the model was
blindfolded to it on purpose. On GAN families it IS allowed to train on it
scores 0.99-1.00.

This script drops the blindfold, because the actual deliverable is a tool
that says "AI or real, with a probability" and is right as often as
possible. Splits are recomputed here in memory (ignoring splits.csv's
`split` column) so no 1.6GB dataset re-upload is needed just to change
which rows are held out -- the images are identical, only the assignment
changes.

Still reported per-family, never pooled: every family gets a test slice the
model never trained on, so the per-family accuracy table stays honest. What
changes is that no family is excluded from training entirely.
"""
import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

torch.manual_seed(0)
random.seed(0)

OUT = Path("/kaggle/working")
OLD_PREFIX = "/kaggle/working/normalized/"


def find_data_root():
    hits = list(Path("/kaggle/input").rglob("splits.csv"))
    if not hits:
        raise FileNotFoundError("splits.csv not found anywhere under /kaggle/input")
    return hits[0].parent


DATA_ROOT = find_data_root()


def usable_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        (torch.zeros(1, device="cuda") + 1)
        torch.cuda.synchronize()
        return torch.device("cuda")
    except RuntimeError as e:
        print(f"CUDA present but unusable ({e}); falling back to CPU")
        return torch.device("cpu")


DEVICE = usable_device()
IMG_SIZE = 224
BATCH_SIZE = 32 if DEVICE.type == "cuda" else 16
EPOCHS = 6 if DEVICE.type == "cuda" else 5
LR = 1e-4


def remap_path(p):
    return str(DATA_ROOT / p[len(OLD_PREFIX):]) if p.startswith(OLD_PREFIX) else p


def load_and_resplit():
    """Read the manifest and assign train/val/test 70/15/15 within EVERY
    family -- including the two that earlier phases held out entirely."""
    with open(DATA_ROOT / "splits.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["path"] = remap_path(r["path"])

    by_family = defaultdict(list)
    for r in rows:
        key = r["family"] if r["label"] == "fake" else f"real/{r['source_dataset']}"
        by_family[key].append(r)

    for family, frows in by_family.items():
        random.shuffle(frows)
        n = len(frows)
        n_train, n_val = int(n * 0.70), int(n * 0.15)
        for i, r in enumerate(frows):
            r["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return rows


class FORGEDataset(Dataset):
    def __init__(self, rows, train):
        self.rows = rows
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
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
        return self.tf(im), (1.0 if r["label"] == "fake" else 0.0)


def build_model():
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, 1)
    return m.to(DEVICE)


def run_epoch(model, loader, loss_fn, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).float()
            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += ((logits > 0) == (y > 0.5)).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_by_family(model, rows, batch_size=128):
    model.eval()
    loader = DataLoader(FORGEDataset(rows, train=False), batch_size=batch_size, shuffle=False, num_workers=2)
    preds = []
    for x, _ in loader:
        preds.extend((model(x.to(DEVICE)).squeeze(1) > 0).cpu().int().tolist())

    per_family = defaultdict(lambda: [0, 0])
    for r, p in zip(rows, preds):
        true = 1 if r["label"] == "fake" else 0
        per_family[r["family"]][0] += int(p == true)
        per_family[r["family"]][1] += 1
    return per_family


def main():
    rows = load_and_resplit()
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]

    train_loader = DataLoader(FORGEDataset(train_rows, train=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(FORGEDataset(val_rows, train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_real = sum(1 for r in train_rows if r["label"] == "real")
    n_fake = sum(1 for r in train_rows if r["label"] == "fake")
    pos_weight = torch.tensor([n_real / n_fake], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    families = sorted(set(r["family"] for r in train_rows if r["label"] == "fake"))
    log_lines = [
        f"device={DEVICE}, epochs={EPOCHS}, train={len(train_rows)} (real={n_real} fake={n_fake}), "
        f"val={len(val_rows)}, test={len(test_rows)}, fake families trained on={len(families)}\n"
    ]

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
        line = (f"epoch {epoch+1}/{EPOCHS}  train_acc={train_acc:.3f}  "
                f"val_acc={val_acc:.3f}  ({time.time()-t0:.0f}s)")
        print(line)
        log_lines.append(line + "\n")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT / "model_best.pt")

    model.load_state_dict(torch.load(OUT / "model_best.pt"))
    test_by_family = eval_by_family(model, test_rows)

    report = ["# FORGE -- final shipping model (trained on all generator families)\n\n",
              "".join(f"    {l}" for l in log_lines), "\n"]
    report.append("## Test accuracy per family (test slices never trained on)\n\n")
    report.append("| family | n | accuracy |\n|---|---:|---:|\n")
    for fam, (c, n) in sorted(test_by_family.items()):
        report.append(f"| {fam} | {n} | {c/n:.3f} |\n")

    total_c = sum(c for c, n in test_by_family.values())
    total_n = sum(n for c, n in test_by_family.values())
    report.append(f"\n**Overall test accuracy: {total_c/total_n:.3f}**\n")
    report.append("\nNote: unlike earlier phase-2 runs, no generator family was withheld from "
                  "training. The leave-one-generator-out numbers reported separately still stand "
                  "as the honest answer to 'what about a generator released after this was trained'.\n")

    with open(OUT / "PHASE2_REPORT.md", "w") as f:
        f.writelines(report)
    print("".join(report))


if __name__ == "__main__":
    main()
