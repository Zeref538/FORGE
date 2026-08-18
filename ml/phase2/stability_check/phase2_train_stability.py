"""
FORGE phase 2j -- epoch-by-epoch stability check. Same model, same data as
whatever forge-normalized-v1 currently holds -- the difference from every
earlier script is that this one evaluates held-out accuracy (StyleGAN3 +
SFHQ-T2I) after EVERY epoch and logs it, instead of only checking the one
checkpoint that happened to win on validation accuracy.

Why: three different data recipes gave StyleGAN3 held-out scores of 0.365,
0.210, and 0.024 -- not a clean trend either direction. A separate run also
showed the SAME data going from 0.365 (epoch 3) to 0.026 (epoch 8). That
suggests held-out accuracy on a family the model never trained on might
swing a lot between epochs regardless of the data recipe, simply because
model-selection (best validation accuracy) never looks at held-out families
at all -- it has no way to know or care whether the checkpoint it keeps is
good or terrible at the one thing we're actually trying to measure. This
run makes that visible by tracking every epoch instead of trusting one.
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
EPOCHS = 10 if DEVICE.type == "cuda" else 8
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
    heldout_rows = [r for r in rows if r["split"] == "heldout"]
    test_rows = [r for r in rows if r["split"] == "test"]
    real_test_rows = [r for r in test_rows if r["label"] == "real"]

    train_loader = DataLoader(FORGEDataset(train_rows, train=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(FORGEDataset(val_rows, train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_real = sum(1 for r in train_rows if r["label"] == "real")
    n_fake = sum(1 for r in train_rows if r["label"] == "fake")
    pos_weight = torch.tensor([n_real / n_fake], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    log_lines = [f"device={DEVICE}, epochs={EPOCHS}, train={len(train_rows)} (real={n_real} fake={n_fake})\n"]
    per_epoch_rows = []
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)

        heldout_by_family = eval_by_family(model, heldout_rows + real_test_rows)
        sg3 = heldout_by_family["real-vs-fake-faces-stylegan3/Fake faces"]
        sfhq = heldout_by_family["sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models"]
        real_acc = heldout_by_family["real"]
        sg3_acc = sg3[0] / sg3[1]
        sfhq_acc = sfhq[0] / sfhq[1]
        real_held_acc = real_acc[0] / real_acc[1]

        line = (f"epoch {epoch+1}/{EPOCHS}  train_acc={train_acc:.3f} val_acc={val_acc:.3f}  "
                f"StyleGAN3={sg3_acc:.3f} SFHQ-T2I={sfhq_acc:.3f} real={real_held_acc:.3f}  ({time.time()-t0:.0f}s)")
        print(line)
        log_lines.append(line + "\n")
        per_epoch_rows.append((epoch + 1, val_acc, sg3_acc, sfhq_acc, real_held_acc))
        torch.save(model.state_dict(), OUT / f"model_epoch{epoch+1}.pt")

    report = ["# FORGE phase 2j -- epoch-by-epoch stability check\n\n", "".join(f"    {l}" for l in log_lines), "\n"]
    report.append("## Held-out accuracy by epoch (not just the best-val checkpoint)\n\n")
    report.append("| epoch | val_acc | StyleGAN3 | SFHQ-T2I | real |\n|---:|---:|---:|---:|---:|\n")
    for ep, val_acc, sg3_acc, sfhq_acc, real_acc in per_epoch_rows:
        report.append(f"| {ep} | {val_acc:.3f} | {sg3_acc:.3f} | {sfhq_acc:.3f} | {real_acc:.3f} |\n")

    sg3_vals = [r[2] for r in per_epoch_rows]
    report.append(f"\n**StyleGAN3 range across epochs: {min(sg3_vals):.3f} - {max(sg3_vals):.3f}**\n")
    report.append(f"**StyleGAN3 at the epoch validation would have picked (highest val_acc): "
                   f"{sorted(per_epoch_rows, key=lambda r: -r[1])[0][2]:.3f}**\n")

    with open(OUT / "PHASE2_REPORT.md", "w") as f:
        f.writelines(report)
    print("".join(report))


if __name__ == "__main__":
    main()
