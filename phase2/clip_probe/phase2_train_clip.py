"""
FORGE phase 2f — CLIP linear-probe ablation. Runs on Kaggle (GPU or CPU
fallback).

Ojha et al., CVPR 2023 ("Towards Universal Fake Image Detectors that
Generalize Across Generative Models") found that instead of training a CNN
end-to-end on fake/real images, freezing a big pretrained model called CLIP
(trained on huge internet image+caption pairs, never built for fake
detection) and just fitting a simple linear classifier on top of its
existing features generalizes far better to generators never seen in
training. Their result: +26% accuracy on unseen generators vs a from-scratch
CNN. The reasoning: a small CNN trained only on fake/real learns shortcuts
tied to what it was shown; CLIP's features come from broad general image
understanding, so they don't carry that narrow shortcut.

Feature extraction (the CLIP forward pass) is frozen — no gradients flow
through it — so features are extracted ONCE per image and cached, then the
linear probe trains on those cached vectors. This is fast even on CPU,
since only a single linear layer is actually being trained.

This is also a size test: CLIP ViT-B/32 is much bigger than the
MobileNetV3-Small baseline (2.5M params). The exported ONNX size at the end
of this script is the real number for deciding whether this fits FORGE's
"must run in the browser" constraint or needs a server-side fallback.
"""
import csv
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import open_clip

torch.manual_seed(0)

# nn.MultiheadAttention has a fused "fast path" (aten::_native_multi_head_attention)
# used automatically in eval mode for speed — it has no ONNX translation, which is
# why the first export attempt failed. Turning it off forces the plain matmul/softmax
# attention math, which IS exportable, at a small inference-speed cost we don't care
# about here.
if hasattr(torch.backends, "mha"):
    torch.backends.mha.set_fastpath_enabled(False)

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
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
BATCH_SIZE = 64
# Logistic regression on 512-dim cached features is convex — LBFGS (same optimizer
# calibrate_and_export.py uses for temperature scaling) converges properly instead
# of the previous 30 Adam steps, which barely moved the loss (0.3086 -> 0.2973).
LBFGS_STEPS = 5


def remap_path(p):
    return str(DATA_ROOT / p[len(OLD_PREFIX):]) if p.startswith(OLD_PREFIX) else p


def load_rows():
    with open(DATA_ROOT / "splits.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["path"] = remap_path(r["path"])
    return rows


class ImgDataset(Dataset):
    def __init__(self, rows, preprocess):
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        im = Image.open(r["path"]).convert("RGB")
        x = self.preprocess(im)
        y = 1.0 if r["label"] == "fake" else 0.0
        return x, y


@torch.no_grad()
def extract_features(clip_model, rows, preprocess, batch_size=BATCH_SIZE):
    loader = DataLoader(ImgDataset(rows, preprocess), batch_size=batch_size, shuffle=False, num_workers=2)
    feats, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        f = clip_model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


class CLIPClassifier(nn.Module):
    """Wraps frozen CLIP + trained linear head into one exportable module."""
    def __init__(self, clip_model, linear):
        super().__init__()
        self.clip_model = clip_model
        self.linear = linear

    def forward(self, x):
        f = self.clip_model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        return self.linear(f.float())


def main():
    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    heldout_rows = [r for r in rows if r["split"] == "heldout"]

    clip_model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    clip_model = clip_model.to(DEVICE).eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    t0 = time.time()
    train_feats, train_labels = extract_features(clip_model, train_rows, preprocess)
    val_feats, val_labels = extract_features(clip_model, val_rows, preprocess)
    test_feats, test_labels = extract_features(clip_model, test_rows, preprocess)
    heldout_feats, heldout_labels = extract_features(clip_model, heldout_rows, preprocess)
    extract_time = time.time() - t0

    feat_dim = train_feats.shape[1]
    linear = nn.Linear(feat_dim, 1).to(DEVICE)
    optimizer = torch.optim.LBFGS(linear.parameters(), lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    n_real = sum(1 for r in train_rows if r["label"] == "real")
    n_fake = sum(1 for r in train_rows if r["label"] == "fake")
    pos_weight = torch.tensor([n_real / n_fake], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_feats_d, train_labels_d = train_feats.to(DEVICE), train_labels.to(DEVICE)
    val_feats_d, val_labels_d = val_feats.to(DEVICE), val_labels.to(DEVICE)

    best_val_acc = 0.0
    log_lines = [
        f"device={DEVICE}, clip={CLIP_MODEL_NAME}/{CLIP_PRETRAINED}, feat_dim={feat_dim}, "
        f"feature_extraction_time={extract_time:.0f}s, train={len(train_rows)} "
        f"(real={n_real} fake={n_fake}, pos_weight={pos_weight.item():.3f}), val={len(val_rows)}, "
        f"test={len(test_rows)}, heldout={len(heldout_rows)}\n"
    ]
    def closure():
        optimizer.zero_grad()
        logits = linear(train_feats_d).squeeze(1)
        loss = loss_fn(logits, train_labels_d)
        loss.backward()
        return loss

    best_state = None
    for step in range(LBFGS_STEPS):
        loss = optimizer.step(closure)
        with torch.no_grad():
            train_acc = ((linear(train_feats_d).squeeze(1) > 0) == (train_labels_d > 0.5)).float().mean().item()
            val_logits = linear(val_feats_d).squeeze(1)
            val_acc = ((val_logits > 0) == (val_labels_d > 0.5)).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in linear.state_dict().items()}
        line = f"LBFGS step {step+1}/{LBFGS_STEPS}  loss={loss.item():.4f} train_acc={train_acc:.3f}  val_acc={val_acc:.3f}"
        print(line)
        log_lines.append(line + "\n")

    linear.load_state_dict(best_state)
    torch.save(linear.state_dict(), OUT / "clip_linear_probe.pt")

    def eval_by_family(feats, labels, rows_subset):
        with torch.no_grad():
            preds = (linear(feats.to(DEVICE)).squeeze(1) > 0).cpu().int().tolist()
        per_family = defaultdict(lambda: [0, 0])
        for r, p, y in zip(rows_subset, preds, labels.int().tolist()):
            per_family[r["family"]][0] += int(p == y)
            per_family[r["family"]][1] += 1
        return per_family

    test_by_family = eval_by_family(test_feats, test_labels, test_rows)

    real_mask = [i for i, r in enumerate(test_rows) if r["label"] == "real"]
    real_test_rows = [test_rows[i] for i in real_mask]
    real_test_feats = test_feats[real_mask]
    real_test_labels = test_labels[real_mask]
    heldout_rows_all = heldout_rows + real_test_rows
    heldout_feats_all = torch.cat([heldout_feats, real_test_feats])
    heldout_labels_all = torch.cat([heldout_labels, real_test_labels])
    heldout_by_family = eval_by_family(heldout_feats_all, heldout_labels_all, heldout_rows_all)

    report = ["# FORGE phase 2f — CLIP linear-probe ablation report\n\n", "".join(f"    {l}" for l in log_lines), "\n"]
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
    report.append("\nCompare against phase2_train.py's MobileNetV3-Small baseline (224px: in-dist 0.959, heldout 0.398, StyleGAN3 0.000, SFHQ-T2I 0.643).\n")
    report.append(f"\nCLIP preprocessing (must match if wired into the web app): {preprocess}\n")

    with open(OUT / "PHASE2_CLIP_REPORT.md", "w") as f:
        f.writelines(report)
    print("".join(report))

    # --- export size check: does this fit the "runs in the browser" budget? ---
    # set_fastpath_enabled(False) above should let the legacy exporter succeed now.
    # If it still doesn't, fall back to the newer dynamo-based exporter as a second
    # attempt in the same run, so one Kaggle session tests both fixes.
    full_model = CLIPClassifier(clip_model, linear).eval()
    dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
    onnx_path = OUT / "forge_clip_model.onnx"
    export_log = []
    try:
        torch.onnx.export(
            full_model, dummy, str(onnx_path),
            input_names=["image"], output_names=["logit"],
            dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
        size_mb = onnx_path.stat().st_size / 1e6
        export_log.append(f"Legacy exporter succeeded: {size_mb:.1f} MB (compare: MobileNetV3-Small baseline is ~6 MB)")
        print(f"ONNX export size: {size_mb:.1f} MB")
    except Exception as e1:
        export_log.append(f"Legacy exporter (opset 17, dynamo=False) failed: {e1}")
        print(f"Legacy ONNX export failed: {e1}; retrying with the dynamo exporter")
        try:
            torch.onnx.export(full_model, dummy, str(onnx_path), input_names=["image"], output_names=["logit"], dynamo=True)
            size_mb = onnx_path.stat().st_size / 1e6
            export_log.append(f"Dynamo exporter succeeded: {size_mb:.1f} MB (compare: MobileNetV3-Small baseline is ~6 MB)")
            print(f"Dynamo ONNX export size: {size_mb:.1f} MB")
        except Exception as e2:
            export_log.append(f"Dynamo exporter also failed: {e2}")
            print(f"Dynamo ONNX export failed: {e2}")

    with open(OUT / "PHASE2_CLIP_REPORT.md", "a") as f:
        f.write("\n## ONNX export\n\n" + "\n\n".join(export_log) + "\n")


if __name__ == "__main__":
    main()
