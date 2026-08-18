"""
FORGE phase 2b — calibration + ONNX export. Runs locally (torch + onnxruntime
installed on this machine, no Kaggle round-trip needed).

1. Loads the trained checkpoint (phase2/output/model_best.pt).
2. Fits a single temperature scalar on the val set (Guo et al. 2017) so the
   sigmoid output is an honest probability, not just a good ranking score.
3. Reports ECE before/after calibration on the in-distribution test set.
4. Picks the "uncertain" probability band from where calibrated accuracy is
   actually near chance in val, not an arbitrary [0.4, 0.6] guess.
5. Exports to ONNX and verifies PyTorch vs ONNX Runtime outputs match on a
   fixed batch, catching op-support gaps before the web app is built on it.
"""
import csv
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
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
IMG_SIZE = 224  # must match phase2_train.py's training resolution
DEVICE = torch.device("cpu")


def remap_path(p):
    return str(LOCAL_DATA_ROOT / p[len(OLD_PREFIX):]) if p.startswith(OLD_PREFIX) else p


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
    m.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    m.eval()
    return m


@torch.no_grad()
def collect_logits(model, rows, batch_size=128):
    loader = DataLoader(FORGEDataset(rows), batch_size=batch_size, shuffle=False, num_workers=2)
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x).squeeze(1))
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits, labels):
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=100)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return T.item()


def ece(probs, labels, n_bins=10):
    probs, labels = np.asarray(probs), np.asarray(labels)
    confidences = np.maximum(probs, 1 - probs)  # confidence in the predicted class
    predictions = (probs > 0.5).astype(float)
    correct = (predictions == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(probs)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        e += (mask.sum() / total) * abs(bin_acc - bin_conf)
    return e


def main():
    rows = load_rows()
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]

    model = build_model()

    val_logits, val_labels = collect_logits(model, val_rows)
    test_logits, test_labels = collect_logits(model, test_rows)

    T = fit_temperature(val_logits, val_labels)

    raw_probs_test = torch.sigmoid(test_logits).numpy()
    cal_probs_test = torch.sigmoid(test_logits / T).numpy()
    ece_before = ece(raw_probs_test, test_labels.numpy())
    ece_after = ece(cal_probs_test, test_labels.numpy())

    # uncertain band: widen a window around 0.5 (calibrated val probabilities)
    # until accuracy inside the window drops to within 5pts of chance (0.5) —
    # this is where the model's confidence is actually untrustworthy, not an
    # arbitrary guess
    cal_probs_val = torch.sigmoid(val_logits / T).numpy()
    val_labels_np = val_labels.numpy()
    band = None
    for w in np.arange(0.02, 0.35, 0.02):
        mask = np.abs(cal_probs_val - 0.5) <= w
        if mask.sum() < 20:
            continue
        acc_in_band = ((cal_probs_val[mask] > 0.5).astype(float) == val_labels_np[mask]).mean()
        if acc_in_band <= 0.55:
            band = w
        else:
            break
    if band is None:
        band = 0.05  # model is confident everywhere near 0.5 too — narrow fallback band

    mask_test = np.abs(cal_probs_test - 0.5) <= band
    coverage = mask_test.mean()
    acc_outside = ((cal_probs_test[~mask_test] > 0.5).astype(float) == test_labels.numpy()[~mask_test]).mean() if (~mask_test).sum() else float("nan")

    report = [
        "# FORGE phase 2b — calibration report\n\n",
        f"Temperature: {T:.3f}\n",
        f"ECE (test, before calibration): {ece_before:.4f}\n",
        f"ECE (test, after calibration): {ece_after:.4f}\n\n",
        f"Uncertain band: calibrated p(generated) in [{0.5-band:.2f}, {0.5+band:.2f}]\n",
        f"Coverage on test set: {coverage:.1%} of images fall in the uncertain band\n",
        f"Accuracy outside the band: {acc_outside:.3f}\n",
    ]
    with open(PHASE2_DIR / "output" / "CALIBRATION_REPORT.md", "w") as f:
        f.writelines(report)
    print("".join(report))

    # --- ONNX export + parity check ---
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    onnx_path = PHASE2_DIR / "output" / "forge_model.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["image"], output_names=["logit"],
        dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    fixed_batch, _ = next(iter(DataLoader(FORGEDataset(test_rows[:16]), batch_size=16)))
    with torch.no_grad():
        torch_out = model(fixed_batch).numpy()

    sess = ort.InferenceSession(str(onnx_path))
    onnx_out = sess.run(None, {"image": fixed_batch.numpy()})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    parity_ok = max_diff < 1e-4
    with open(PHASE2_DIR / "output" / "CALIBRATION_REPORT.md", "a") as f:
        f.write(f"\n## ONNX export parity\n\nmax |torch - onnx| on 16-image batch: {max_diff:.6f}\n")
        f.write(f"Parity check: {'PASS' if parity_ok else 'FAIL'}\n")
    print(f"ONNX parity max diff: {max_diff:.6f} ({'PASS' if parity_ok else 'FAIL'})")

    with open(PHASE2_DIR / "output" / "calibration.json", "w") as f:
        json.dump({"temperature": T, "uncertain_band": band}, f)


if __name__ == "__main__":
    main()
