"""
FORGE -- backbone size comparison. Trains the SAME data/recipe on three
model sizes and reports accuracy alongside the exported ONNX file size,
because those two numbers together are the actual decision.

Why this matters here: MobileNetV3-Small was chosen at the start for one
reason only -- it has to download into a browser, and ~6MB does. It was
never chosen for accuracy. The published work this project compares against
(Ojha et al. 2023) used ResNet-50, roughly 10x larger. This run measures
what that size difference is actually worth on THIS data, so the
browser-vs-accuracy tradeoff is made on numbers instead of assumption.

  mobilenet_v3_small  ~2.5M params   (current, ~6MB ONNX)
  efficientnet_b0     ~5.3M params   (middle ground)
  resnet50           ~25.6M params   (what the literature used)

Trains on ALL generator families (no leave-one-out handicap) -- this is for
the practical tool. Per-family test accuracy is still reported separately,
never pooled into a single flattering number.
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
ON_GPU = DEVICE.type == "cuda"
EPOCHS = 5 if ON_GPU else 3
LR = 1e-4

# resnet50 activations are much larger per image; keep batches smaller for it
BATCH = {
    "mobilenet_v3_small": 32 if ON_GPU else 16,
    "efficientnet_b0": 32 if ON_GPU else 16,
    "resnet50": 24 if ON_GPU else 8,
}


def remap_path(p):
    return str(DATA_ROOT / p[len(OLD_PREFIX):]) if p.startswith(OLD_PREFIX) else p


def load_and_resplit():
    with open(DATA_ROOT / "splits.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["path"] = remap_path(r["path"])
    by_family = defaultdict(list)
    for r in rows:
        key = r["family"] if r["label"] == "fake" else f"real/{r['source_dataset']}"
        by_family[key].append(r)
    for frows in by_family.values():
        random.shuffle(frows)
        n = len(frows)
        n_train, n_val = int(n * 0.70), int(n * 0.15)
        for i, r in enumerate(frows):
            r["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return rows


class FORGEDataset(Dataset):
    def __init__(self, rows, train):
        self.rows = rows
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), norm,
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), norm,
            ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        im = Image.open(r["path"]).convert("RGB")
        return self.tf(im), (1.0 if r["label"] == "fake" else 0.0)


def build(name):
    """Swap each architecture's final classification layer for a single
    output (one number = how fake-looking, before the sigmoid)."""
    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 1)
    else:
        raise ValueError(name)
    return m.to(DEVICE)


def run_epoch(model, loader, loss_fn, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).float()
            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
            correct += ((logits > 0) == (y > 0.5)).sum().item()
            n += x.size(0)
    return total / n, correct / n


@torch.no_grad()
def eval_by_family(model, rows, batch_size):
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

    n_real = sum(1 for r in train_rows if r["label"] == "real")
    n_fake = sum(1 for r in train_rows if r["label"] == "fake")
    pos_weight = torch.tensor([n_real / n_fake], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    report = ["# FORGE -- backbone size comparison\n\n",
              f"    device={DEVICE}, epochs={EPOCHS}, train={len(train_rows)} "
              f"(real={n_real} fake={n_fake}), val={len(val_rows)}, test={len(test_rows)}\n\n"]
    summary = []
    per_family_tables = []

    for name in ["mobilenet_v3_small", "efficientnet_b0", "resnet50"]:
        bs = BATCH[name]
        train_loader = DataLoader(FORGEDataset(train_rows, True), batch_size=bs, shuffle=True, num_workers=2)
        val_loader = DataLoader(FORGEDataset(val_rows, False), batch_size=bs, shuffle=False, num_workers=2)

        model = build(name)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_val, ckpt = 0.0, OUT / f"{name}_best.pt"
        t_start = time.time()
        for epoch in range(EPOCHS):
            t0 = time.time()
            _, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
            _, val_acc = run_epoch(model, val_loader, loss_fn)
            line = (f"    [{name}] epoch {epoch+1}/{EPOCHS} train_acc={train_acc:.3f} "
                    f"val_acc={val_acc:.3f} ({time.time()-t0:.0f}s)")
            print(line, flush=True)
            report.append(line + "\n")
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), ckpt)
        train_time = time.time() - t_start

        model.load_state_dict(torch.load(ckpt))
        by_fam = eval_by_family(model, test_rows, bs)
        acc = sum(c for c, _ in by_fam.values()) / sum(n for _, n in by_fam.values())

        # export to ONNX -- the size here is what a browser would download
        onnx_path = OUT / f"forge_{name}.onnx"
        model_cpu = model.to("cpu").eval()
        torch.onnx.export(
            model_cpu, torch.randn(1, 3, IMG_SIZE, IMG_SIZE), str(onnx_path),
            input_names=["image"], output_names=["logit"],
            dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
        size_mb = onnx_path.stat().st_size / 1e6
        model.to(DEVICE)

        summary.append((name, n_params, acc, size_mb, train_time))
        rows_tbl = [f"\n### {name}\n\n| family | n | accuracy |\n|---|---:|---:|\n"]
        for fam, (c, n) in sorted(by_fam.items()):
            rows_tbl.append(f"| {fam} | {n} | {c/n:.3f} |\n")
        per_family_tables.append("".join(rows_tbl))
        print(f"[{name}] done: acc={acc:.3f} size={size_mb:.1f}MB", flush=True)

    report.append("\n## Summary -- accuracy vs download size\n\n")
    report.append("| backbone | params | overall test acc | ONNX size | train time |\n")
    report.append("|---|---:|---:|---:|---:|\n")
    for name, n_params, acc, size_mb, t in summary:
        report.append(f"| {name} | {n_params/1e6:.1f}M | {acc:.3f} | {size_mb:.1f} MB | {t/60:.0f} min |\n")
    report.append("\n(ONNX size is what a browser must download before the first "
                  "prediction. ~6MB is instant; ~100MB is not.)\n")
    report.extend(per_family_tables)

    with open(OUT / "BACKBONE_REPORT.md", "w") as f:
        f.writelines(report)
    print("".join(report))


if __name__ == "__main__":
    main()
