"""
Calibrate the EfficientNet-B0 export and produce web/model/calibration.json.

Runs against the .onnx file rather than the PyTorch checkpoint on purpose:
the browser loads the ONNX, so the temperature is fitted on exactly the
numbers that will ship. Calibrating the checkpoint instead would leave a
small unverified gap between what was measured and what runs.

Reproduces the split from backbone_compare/phase2_backbones.py exactly
(same seed, same per-family shuffle) so the validation images used here are
the ones the model did not train on.
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PHASE2 = Path(__file__).parent
PHASE1_OUT = PHASE2.parent / "phase1" / "output"
ONNX_PATH = PHASE2 / "backbone_compare" / "output" / "forge_efficientnet_b0.onnx"
WEB_MODEL = PHASE2.parent / "web" / "model"
OLD_PREFIX = "/kaggle/working/normalized/"
IMG_SIZE = 224


def remap(p):
    return str(PHASE1_OUT / "normalized" / p[len(OLD_PREFIX):]) if p.startswith(OLD_PREFIX) else p


def load_and_resplit():
    """Byte-identical to phase2_backbones.load_and_resplit so the val split
    here is the same one the model was selected on."""
    random.seed(0)
    with open(PHASE1_OUT / "splits.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["path"] = remap(r["path"])
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


class Imgs(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.tf(Image.open(r["path"]).convert("RGB")), (1.0 if r["label"] == "fake" else 0.0)


def onnx_logits(sess, rows, batch=64):
    loader = DataLoader(Imgs(rows), batch_size=batch, shuffle=False, num_workers=2)
    out, ys = [], []
    for x, y in loader:
        logits = sess.run(None, {"image": x.numpy()})[0]
        out.append(torch.from_numpy(logits).squeeze(1))
        ys.append(y)
    return torch.cat(out), torch.cat(ys)


def fit_temperature(logits, labels):
    T = nn.Parameter(torch.ones(1))
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
    conf = np.maximum(probs, 1 - probs)
    correct = ((probs > 0.5).astype(float) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += (m.sum() / len(probs)) * abs(correct[m].mean() - conf[m].mean())
    return e


def main():
    rows = load_and_resplit()
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    print(f"val={len(val_rows)} test={len(test_rows)}")

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    val_logits, val_labels = onnx_logits(sess, val_rows)
    test_logits, test_labels = onnx_logits(sess, test_rows)

    T = fit_temperature(val_logits, val_labels)
    raw = torch.sigmoid(test_logits).numpy()
    cal = torch.sigmoid(test_logits / T).numpy()
    e_before, e_after = ece(raw, test_labels.numpy()), ece(cal, test_labels.numpy())

    # widen a window around 0.5 until accuracy inside it approaches chance --
    # that is where confidence is genuinely untrustworthy, rather than a guess
    cal_val = torch.sigmoid(val_logits / T).numpy()
    yv = val_labels.numpy()
    band = None
    for w in np.arange(0.02, 0.35, 0.02):
        m = np.abs(cal_val - 0.5) <= w
        if m.sum() < 20:
            continue
        if ((cal_val[m] > 0.5).astype(float) == yv[m]).mean() <= 0.55:
            band = float(w)
        else:
            break
    if band is None:
        band = 0.05

    m_test = np.abs(cal - 0.5) <= band
    coverage = float(m_test.mean())
    acc_out = float(((cal[~m_test] > 0.5).astype(float) == test_labels.numpy()[~m_test]).mean())

    report = [
        "# FORGE -- EfficientNet-B0 calibration\n\n",
        f"Fitted on the ONNX export ({ONNX_PATH.stat().st_size/1e6:.1f} MB), i.e. exactly what the browser runs.\n\n",
        f"Temperature: {T:.3f}\n",
        f"ECE before: {e_before:.4f}\n",
        f"ECE after:  {e_after:.4f}\n\n",
        f"Uncertain band: p in [{0.5-band:.2f}, {0.5+band:.2f}]\n",
        f"Coverage: {coverage:.1%} of test images land in the uncertain band\n",
        f"Accuracy outside the band: {acc_out:.3f}\n",
    ]
    (PHASE2 / "output").mkdir(exist_ok=True)
    (PHASE2 / "output" / "CALIBRATION_EFFICIENTNET.md").write_text("".join(report))

    WEB_MODEL.mkdir(parents=True, exist_ok=True)
    (WEB_MODEL / "calibration.json").write_text(
        json.dumps({"temperature": T, "uncertain_band": band})
    )
    print("".join(report))


if __name__ == "__main__":
    main()
