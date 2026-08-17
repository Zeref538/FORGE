"""
FORGE phase 2g -- longer-training ablation. Same model, same data (10
generator families, StyleGAN3 + SFHQ-T2I held out) as the current best
baseline (phase2_train.py) -- only variable changed is EPOCHS.

Why: the StyleGAN/StyleGAN2 family was only just added to training, and the
CPU baseline run stopped at 3 epochs (one epoch = one full pass over every
training image) to keep Kaggle sessions short. Checkpointing already keeps
whichever epoch had the best val accuracy, so more epochs can only help or
tie, never hurt the picked checkpoint -- this tests whether 3 epochs left
real accuracy on the table, especially on the newly-added family.
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

torch.manual_seed(0)

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
EPOCHS = 10 if DEVICE.type == "cuda" else 8  # was 4/3 in the baseline -- more practice rounds, same data
LR = 1e-4


def remap_path(p):
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

    n_real = sum(1 for r in train_rows if r["label"] == "real")
    n_fake = sum(1 for r in train_rows if r["label"] == "fake")
    pos_weight = torch.tensor([n_real / n_fake], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_acc = 0.0
    log_lines = [f"device={DEVICE}, epochs={EPOCHS}, train={len(train_rows)} (real={n_real} fake={n_fake}, pos_weight={pos_weight.item():.3f}), val={len(val_rows)}, test={len(test_rows)}, heldout={len(heldout_rows)}\n"]
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
    real_test_rows = [r for r in test_rows if r["label"] == "real"]
    heldout_by_family = eval_by_family(model, heldout_rows + real_test_rows)

    report = ["# FORGE phase 2g -- longer-training ablation report\n\n", "".join(f"    {l}" for l in log_lines), "\n"]
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
    report.append("\nCompare against phase2_train.py's 3-epoch run (in-dist 0.933, heldout 0.447, StyleGAN3 0.365, SFHQ-T2I 0.398).\n")

    with open(OUT / "PHASE2_REPORT.md", "w") as f:
        f.writelines(report)
    print("".join(report))


if __name__ == "__main__":
    main()
