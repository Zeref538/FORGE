"""
FORGE phase 2i -- source-diversity ablation. Same training recipe as
phase2_train.py -- the change is upstream, in the dataset: now has TWO
StyleGAN-family sources (deepfake-face-images + 140k-real-and-fake-faces)
capped at 2,500 images EACH instead of one source dumped to 10,000. Tests
whether spreading the same rough budget across two different fingerprints
generalizes to StyleGAN3 better than one dominant source did (which scored
worse than a single small dose: 0.210 vs 0.365).

Attach: johnandreimartinez/forge-normalized-v1 (has splits.csv baked in —
the single source of truth for which image is train/val/test/heldout).

Trains MobileNetV3-Small (ImageNet-pretrained, swapped to a 1-logit head) on
`train`, model-selects on `val`, then reports two numbers that must be read
separately, never pooled: accuracy on `test` (families the model trained on)
vs accuracy on `heldout` (StyleGAN3-Fake + SFHQ-T2I, never seen in training).
The gap between them is the actual result of this project.
"""
import csv
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# a previous run's classification head (randomly initialized fresh each run)
# went unseeded, and with only 1 CPU epoch the random init dominated the
# result — held-out accuracy on one family swung 1.000 -> 0.000 between two
# otherwise-identical runs. Fixing the seed makes runs reproducible; more
# epochs (below) makes the result reflect training, not initialization luck.
torch.manual_seed(0)

OUT = Path("/kaggle/working")
OLD_PREFIX = "/kaggle/working/normalized/"


def find_data_root():
    # Kaggle mounts attached datasets inconsistently between sessions —
    # sometimes flat (/kaggle/input/<slug>/), sometimes nested
    # (/kaggle/input/datasets/<owner>/<slug>/). Locate splits.csv instead of
    # assuming a fixed depth (this exact assumption broke phase1 twice).
    hits = list(Path("/kaggle/input").rglob("splits.csv"))
    if not hits:
        raise FileNotFoundError("splits.csv not found anywhere under /kaggle/input")
    return hits[0].parent


DATA_ROOT = find_data_root()

def usable_device():
    # Kaggle's GPU pool sometimes hands out older Pascal cards (P100, sm_60)
    # that the preinstalled PyTorch build (sm_70+ only) can't run kernels on.
    # A tiny op fails the same way full training would, just instantly —
    # falling back to CPU beats crashing 40s into a real run.
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
# bumped from 224 -> 384: source images are normalized to 512px, and a lot of
# the actual detection signal (frequency-domain / upsampling artifacts) lives
# in exactly the high-frequency detail that aggressive downsampling destroys
IMG_SIZE = 224  # back to the 224px baseline config — isolates the new GAN training data as the only variable
BATCH_SIZE = 32 if DEVICE.type == "cuda" else 16
EPOCHS = 4 if DEVICE.type == "cuda" else 3
LR = 1e-4


def remap_path(p):
    # manifest paths were recorded at write time (/kaggle/working/normalized/...);
    # the packaged dataset mounts them under DATA_ROOT instead.
    return str(DATA_ROOT / p[len(OLD_PREFIX):]) if p.startswith(OLD_PREFIX) else p


def load_rows():
    with open(DATA_ROOT / "splits.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["path"] = remap_path(r["path"])
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
        x = self.tf(im)
        y = 1.0 if r["label"] == "fake" else 0.0
        return x, y


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
    ds = FORGEDataset(rows, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    preds = []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x).squeeze(1)
        preds.extend((logits > 0).cpu().int().tolist())

    per_family_correct = defaultdict(lambda: [0, 0])
    for r, p in zip(rows, preds):
        true = 1 if r["label"] == "fake" else 0
        fam = r["family"]
        per_family_correct[fam][0] += int(p == true)
        per_family_correct[fam][1] += 1
    return per_family_correct


def main():
    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    heldout_rows = [r for r in rows if r["split"] == "heldout"]

    train_loader = DataLoader(FORGEDataset(train_rows, train=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(FORGEDataset(val_rows, train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # train set is 22500 fake vs 5000 real (4.5:1) — unweighted BCE lets the
    # model minimize loss by leaning toward "fake" and eating the errors on
    # the minority real class. pos_weight < 1 discounts the majority (fake,
    # y=1) class's loss contribution to compensate.
    n_real = sum(1 for r in train_rows if r["label"] == "real")
    n_fake = sum(1 for r in train_rows if r["label"] == "fake")
    pos_weight = torch.tensor([n_real / n_fake], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_acc = 0.0
    log_lines = [f"device={DEVICE}, train={len(train_rows)} (real={n_real} fake={n_fake}, pos_weight={pos_weight.item():.3f}), val={len(val_rows)}, test={len(test_rows)}, heldout={len(heldout_rows)}\n"]
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
        line = f"epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  val_loss={val_loss:.4f} val_acc={val_acc:.3f}  ({time.time()-t0:.0f}s)"
        print(line)
        log_lines.append(line + "\n")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT / "model_best.pt")

    model.load_state_dict(torch.load(OUT / "model_best.pt"))

    test_by_family = eval_by_family(model, test_rows)
    # held-out fake families have no real counterpart of their own, so
    # "accuracy" on them alone would just be recall-on-fakes; pair with the
    # real test images (never trained on either) for a real binary accuracy
    real_test_rows = [r for r in test_rows if r["label"] == "real"]
    heldout_by_family = eval_by_family(model, heldout_rows + real_test_rows)

    report = ["# FORGE phase 2i -- source-diversity ablation report\n\n", "".join(f"    {l}" for l in log_lines), "\n"]
    report.append("## Test accuracy (families seen in training)\n\n")
    report.append("| family | n | accuracy |\n|---|---:|---:|\n")
    for fam, (c, n) in sorted(test_by_family.items()):
        report.append(f"| {fam} | {n} | {c/n:.3f} |\n")

    report.append("\n## Held-out accuracy (generators never trained on)\n\n")
    report.append("| family | n | accuracy |\n|---|---:|---:|\n")
    for fam, (c, n) in sorted(heldout_by_family.items()):
        report.append(f"| {fam} | {n} | {c/n:.3f} |\n")

    test_acc = sum(c for c, n in test_by_family.values()) / sum(n for c, n in test_by_family.values())
    heldout_acc = sum(c for c, n in heldout_by_family.values()) / sum(n for c, n in heldout_by_family.values())
    report.append(f"\n**In-distribution test accuracy: {test_acc:.3f}**\n")
    report.append(f"**Held-out (unseen generator) accuracy: {heldout_acc:.3f}**\n")
    report.append(f"**Generalization gap: {test_acc - heldout_acc:.3f}**\n")
    report.append("\nCompare: no-GAN baseline (StyleGAN3 0.000), small single-source dose (0.365), big single-source dose 10k (0.210).\n")

    with open(OUT / "PHASE2_REPORT.md", "w") as f:
        f.writelines(report)
    print("".join(report))


if __name__ == "__main__":
    main()
