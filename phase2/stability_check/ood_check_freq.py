"""
FORGE phase 2n -- second attempt at "does this look unfamiliar", this time
in frequency space instead of the classifier's general-purpose features.

The first attempt (ood_check.py) measured familiarity using the model's
own internal picture of an image -- which turned out to mostly encode
CONTENT (is this a face, how's it framed), so a StyleGAN3 face looked just
as "familiar" as any other face. This attempt uses a different
representation entirely: the RADIAL POWER SPECTRUM (a standard technique
in GAN-forensics literature, e.g. Zhang et al. 2019 "Detecting and
Simulating Artifacts in GAN Fake Images") -- take the image's 2D frequency
spectrum (from an FFT, same as the earlier failed freq-channel ablation)
and collapse it into "how much energy sits at each distance from the
center frequency". Real photos have a smooth, predictable falloff; GAN
upsampling tends to leave odd bumps at specific frequencies that don't
depend on WHAT'S in the photo, only on HOW it was built. No neural network
needed for this one -- just image math, so it's fast.
"""
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

random.seed(0)

HERE = Path(__file__).parent
PHASE1_OUT = HERE.parent.parent / "phase1" / "output"
OLD_PREFIX = "/kaggle/working/normalized/"
BANK_SIZE = 3000
K = 5
FAMILIAR_KEEP_RATE = 0.90
N_BINS = 64  # radial frequency bins


def remap_path(p):
    return str(PHASE1_OUT / "normalized" / p[len(OLD_PREFIX):]) if p.startswith(OLD_PREFIX) else p


def load_rows():
    with open(PHASE1_OUT / "splits.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["path"] = remap_path(r["path"])
    return rows


def radial_profile(path, n_bins=N_BINS, size=224):
    im = Image.open(path).convert("L").resize((size, size))
    gray = np.asarray(im, dtype=np.float32) / 255.0
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = min(cy, cx)
    r_bin = np.clip((r * (n_bins - 1) / max_r).astype(int), 0, n_bins - 1)

    profile = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    np.add.at(profile, r_bin.ravel(), magnitude.ravel())
    np.add.at(counts, r_bin.ravel(), 1)
    profile = profile / np.maximum(counts, 1)
    return profile / (np.linalg.norm(profile) + 1e-8)  # unit-normalize, so distance is about shape not overall brightness


def extract_all(rows):
    return torch.tensor(np.stack([radial_profile(r["path"]) for r in rows]), dtype=torch.float32)


def knn_distance(query_feats, bank_feats, k=K):
    sims = query_feats @ bank_feats.T  # already unit-normalized -> this is cosine similarity
    topk_sims, _ = sims.topk(k, dim=1)
    return 1 - topk_sims[:, -1]


def main():
    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    sg3_rows = [r for r in rows if r["family"] == "real-vs-fake-faces-stylegan3/Fake faces"]
    sfhq_rows = [r for r in rows if r["family"] == "sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models"]

    bank_rows = random.sample(train_rows, min(BANK_SIZE, len(train_rows)))
    val_sample = random.sample(val_rows, min(1000, len(val_rows)))
    test_sample = random.sample(test_rows, min(1000, len(test_rows)))

    print(f"computing radial spectra: bank={len(bank_rows)}, val={len(val_sample)}, test={len(test_sample)}, "
          f"StyleGAN3={len(sg3_rows)}, SFHQ-T2I={len(sfhq_rows)}")
    bank_feats = extract_all(bank_rows)
    val_feats = extract_all(val_sample)
    test_feats = extract_all(test_sample)
    sg3_feats = extract_all(sg3_rows)
    sfhq_feats = extract_all(sfhq_rows)

    val_dist = knn_distance(val_feats, bank_feats)
    threshold = val_dist.quantile(FAMILIAR_KEEP_RATE).item()

    def flagged_rate(dist):
        return (dist > threshold).float().mean().item()

    test_dist = knn_distance(test_feats, bank_feats)
    sg3_dist = knn_distance(sg3_feats, bank_feats)
    sfhq_dist = knn_distance(sfhq_feats, bank_feats)

    lines = [
        "# FORGE phase 2n -- out-of-distribution check, radial frequency spectrum\n\n",
        f"Threshold picked from validation images so {1-FAMILIAR_KEEP_RATE:.0%} of ordinary images "
        f"get flagged \"unfamiliar\" (threshold={threshold:.4f}).\n\n",
        "| set | n | flagged unfamiliar |\n|---|---:|---:|\n",
        f"| in-distribution test (sanity check) | {len(test_sample)} | {flagged_rate(test_dist):.1%} |\n",
        f"| StyleGAN3 (held out) | {len(sg3_rows)} | {flagged_rate(sg3_dist):.1%} |\n",
        f"| SFHQ-T2I (held out) | {len(sfhq_rows)} | {flagged_rate(sfhq_dist):.1%} |\n",
    ]
    with open(HERE / "output" / "OOD_FREQ_REPORT.md", "w") as f:
        f.writelines(lines)
    print("".join(lines))


if __name__ == "__main__":
    main()
