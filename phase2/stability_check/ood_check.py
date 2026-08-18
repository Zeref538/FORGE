"""
FORGE phase 2m -- out-of-distribution (OOD) check: instead of trusting the
classifier's own confidence (already shown to be falsely confident 58% of
the time on StyleGAN3), measure how UNFAMILIAR an image looks to the model
directly, and flag it regardless of what the classifier says.

Method: k-nearest-neighbor distance in feature space (Sun et al. 2022,
"Out-of-Distribution Detection with Deep Nearest Neighbors" -- a simple,
well-established OOD baseline). Pull the 576-dim feature vector MobileNetV3
computes right before its final classification layer (via a forward hook on
avgpool), for a bank of training images. For any new image, find its
distance to the nearest handful of training images in that feature space --
close to several = "looks like something I trained on", far from all of
them = "I've never seen anything like this".

Threshold is picked from held-back IN-DISTRIBUTION validation images (not
StyleGAN3 -- that would be cheating, using the test answer to build the
test), at whatever distance keeps ~90% of ordinary images correctly marked
"familiar". Then checks what fraction of the two held-out families that
same threshold catches as "unfamiliar" -- a signal genuinely separate from
whether the classifier itself gets the label right.
"""
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

random.seed(0)

HERE = Path(__file__).parent
PHASE1_OUT = HERE.parent.parent / "phase1" / "output"
PHASE2_OUT = HERE.parent / "output"
OLD_PREFIX = "/kaggle/working/normalized/"
IMG_SIZE = 224
DEVICE = torch.device("cpu")
BANK_SIZE = 3000   # training images used as the "what does familiar look like" reference set
K = 5              # distance to the 5th-nearest training image
FAMILIAR_KEEP_RATE = 0.90  # threshold picked so 90% of ordinary val images stay "familiar"


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
        return self.tf(im), 0.0


def build_model():
    m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, 1)
    return m.to(DEVICE)


@torch.no_grad()
def extract_features(model, rows, batch_size=64):
    feats = []
    captured = {}

    def hook(module, inp, out):
        captured["feat"] = out

    handle = model.avgpool.register_forward_hook(hook)
    loader = DataLoader(FORGEDataset(rows), batch_size=batch_size, shuffle=False, num_workers=2)
    for x, _ in loader:
        model(x.to(DEVICE))
        feats.append(captured["feat"].flatten(1))
    handle.remove()
    return torch.cat(feats)


def knn_distance(query_feats, bank_feats, k=K):
    # cosine distance: 1 - cosine similarity, robust to feature-vector scale
    q = torch.nn.functional.normalize(query_feats, dim=1)
    b = torch.nn.functional.normalize(bank_feats, dim=1)
    sims = q @ b.T
    topk_sims, _ = sims.topk(k, dim=1)
    kth_sim = topk_sims[:, -1]
    return 1 - kth_sim


def main():
    model = build_model()
    model.load_state_dict(torch.load(PHASE2_OUT / "model_best.pt", map_location=DEVICE))
    model.eval()

    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    sg3_rows = [r for r in rows if r["family"] == "real-vs-fake-faces-stylegan3/Fake faces"]
    sfhq_rows = [r for r in rows if r["family"] == "sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models"]

    bank_rows = random.sample(train_rows, min(BANK_SIZE, len(train_rows)))
    val_sample = random.sample(val_rows, min(1000, len(val_rows)))
    test_sample = random.sample(test_rows, min(1000, len(test_rows)))

    print(f"extracting features: bank={len(bank_rows)}, val={len(val_sample)}, test={len(test_sample)}, "
          f"StyleGAN3={len(sg3_rows)}, SFHQ-T2I={len(sfhq_rows)}")
    bank_feats = extract_features(model, bank_rows)
    val_feats = extract_features(model, val_sample)
    test_feats = extract_features(model, test_sample)
    sg3_feats = extract_features(model, sg3_rows)
    sfhq_feats = extract_features(model, sfhq_rows)

    val_dist = knn_distance(val_feats, bank_feats)
    threshold = val_dist.quantile(FAMILIAR_KEEP_RATE).item()

    def flagged_rate(dist):
        return (dist > threshold).float().mean().item()

    test_dist = knn_distance(test_feats, bank_feats)
    sg3_dist = knn_distance(sg3_feats, bank_feats)
    sfhq_dist = knn_distance(sfhq_feats, bank_feats)

    lines = [
        "# FORGE phase 2m -- out-of-distribution (unfamiliar-image) check\n\n",
        f"Threshold picked from validation images so {1-FAMILIAR_KEEP_RATE:.0%} of ordinary images "
        f"get flagged \"unfamiliar\" (threshold={threshold:.4f}).\n\n",
        "| set | n | flagged unfamiliar |\n|---|---:|---:|\n",
        f"| in-distribution test (sanity check) | {len(test_sample)} | {flagged_rate(test_dist):.1%} |\n",
        f"| StyleGAN3 (held out) | {len(sg3_rows)} | {flagged_rate(sg3_dist):.1%} |\n",
        f"| SFHQ-T2I (held out) | {len(sfhq_rows)} | {flagged_rate(sfhq_dist):.1%} |\n",
    ]
    with open(HERE / "output" / "OOD_REPORT.md", "w") as f:
        f.writelines(lines)
    print("".join(lines))


if __name__ == "__main__":
    main()
