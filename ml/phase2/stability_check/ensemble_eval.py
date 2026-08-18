"""
FORGE phase 2k -- can we tame the epoch-to-epoch noise instead of just
reporting it? Runs locally (torch installed on this machine already, no
Kaggle round-trip -- and the 8 checkpoints from the clean stability check
are already on disk).

Two standard fixes for "training landed on a noisy checkpoint", tested
against the same held-out families used everywhere else in this project:

1. Prediction averaging (a simple ensemble): run every held-out image
   through all 8 saved epoch checkpoints, average their probabilities,
   then threshold. If different epochs are wrong in different ways, this
   smooths out the noise instead of betting on one lucky/unlucky epoch.

2. Weight averaging (Stochastic Weight Averaging, Izmailov et al. 2018):
   average the checkpoints' WEIGHTS together into a single model, instead
   of averaging predictions. Requires recomputing BatchNorm's running
   statistics afterward (its running mean/var don't average correctly on
   their own), done here with a pass over a sample of training images.

Both are tested using checkpoints from a range of windows (all 8, just the
last 4, etc.) since including the worst early epochs might hurt more than
help.
"""
import copy
import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.optim.swa_utils import update_bn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

random.seed(0)
torch.manual_seed(0)

HERE = Path(__file__).parent
PHASE1_OUT = HERE.parent.parent / "phase1" / "output"
CKPT_DIR = HERE / "output"
OLD_PREFIX = "/kaggle/working/normalized/"
IMG_SIZE = 224
DEVICE = torch.device("cpu")
BN_RECAL_SAMPLE = 2000  # subset of train images for BatchNorm recalibration -- full 17500 would be slow locally and isn't needed for stable BN stats


def remap_path(p):
    return str(PHASE1_OUT / "normalized" / p[len(OLD_PREFIX):]) if p.startswith(OLD_PREFIX) else p


def load_rows():
    with open(PHASE1_OUT / "splits.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["path"] = remap_path(r["path"])
    return rows


class FORGEDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
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
    m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, 1)
    return m.to(DEVICE)


def load_ckpt(epoch):
    m = build_model()
    m.load_state_dict(torch.load(CKPT_DIR / f"model_epoch{epoch}.pt", map_location=DEVICE))
    m.eval()
    return m


@torch.no_grad()
def probs_for(model, rows, batch_size=128):
    loader = DataLoader(FORGEDataset(rows), batch_size=batch_size, shuffle=False, num_workers=2)
    out = []
    for x, _ in loader:
        out.append(torch.sigmoid(model(x.to(DEVICE)).squeeze(1)))
    return torch.cat(out)


def eval_by_family(probs, rows):
    preds = (probs > 0.5).int().tolist()
    per_family = defaultdict(lambda: [0, 0])
    for r, p in zip(rows, preds):
        true = 1 if r["label"] == "fake" else 0
        per_family[r["family"]][0] += int(p == true)
        per_family[r["family"]][1] += 1
    return per_family


def weight_average(models_list):
    state_dicts = [m.state_dict() for m in models_list]
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state:
        if avg_state[key].is_floating_point():
            avg_state[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
    avg_model = build_model()
    avg_model.load_state_dict(avg_state)
    return avg_model


def main():
    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    heldout_rows = [r for r in rows if r["split"] == "heldout"]
    test_rows = [r for r in rows if r["split"] == "test"]
    real_test_rows = [r for r in test_rows if r["label"] == "real"]
    eval_rows = heldout_rows + real_test_rows

    bn_sample_rows = random.sample(train_rows, min(BN_RECAL_SAMPLE, len(train_rows)))
    bn_loader = DataLoader(FORGEDataset(bn_sample_rows), batch_size=32, shuffle=False, num_workers=2)

    models_by_epoch = {e: load_ckpt(e) for e in range(1, 9)}

    report = ["# FORGE phase 2k -- ensembling / weight-averaging check\n\n"]
    report.append("Baseline (single checkpoints, from the stability check): "
                   "StyleGAN3 ranged 0.005-0.305, SFHQ-T2I ranged 0.280-0.709.\n\n")
    report.append("| method | window | StyleGAN3 | SFHQ-T2I | real | in-dist test |\n")
    report.append("|---|---|---:|---:|---:|---:|\n")

    def row_for(name, window, probs):
        by_fam = eval_by_family(probs, eval_rows)
        sg3 = by_fam["real-vs-fake-faces-stylegan3/Fake faces"]
        sfhq = by_fam["sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models"]
        real = by_fam["real"]
        line = (f"| {name} | {window} | {sg3[0]/sg3[1]:.3f} | {sfhq[0]/sfhq[1]:.3f} | "
                f"{real[0]/real[1]:.3f} | - |\n")
        report.append(line)
        print(line.strip())

    windows = {
        "all 8 epochs": list(range(1, 9)),
        "last 4 epochs": list(range(5, 9)),
        "last 6 epochs": list(range(3, 9)),
        "epochs 2+4 (the two highest individually)": [2, 4],
    }

    for label, epochs in windows.items():
        # prediction averaging
        all_probs = torch.stack([probs_for(models_by_epoch[e], eval_rows) for e in epochs])
        avg_probs = all_probs.mean(dim=0)
        row_for("prediction-average", label, avg_probs)

        # weight averaging (needs BN recalibration)
        avg_model = weight_average([models_by_epoch[e] for e in epochs])
        avg_model.train()
        update_bn(bn_loader, avg_model, device=DEVICE)
        avg_model.eval()
        wa_probs = probs_for(avg_model, eval_rows)
        row_for("weight-average (SWA-style)", label, wa_probs)

    with open(HERE / "output" / "ENSEMBLE_REPORT.md", "w") as f:
        f.writelines(report)
    print("".join(report))


if __name__ == "__main__":
    main()
