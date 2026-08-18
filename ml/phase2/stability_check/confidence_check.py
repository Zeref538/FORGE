"""
FORGE phase 2l -- does the model at least know it's unsure on StyleGAN3?

Ensembling washed out the one epoch that partially caught StyleGAN3 instead
of reinforcing it -- meaning that epoch's behavior wasn't a shared, robust
pattern nearby epochs also leaned toward, just a one-off. Raw accuracy on
this family looks like a wall for now.

Different question worth checking before writing raw accuracy off entirely:
even when the CURRENTLY SHIPPED model gets a StyleGAN3 image wrong, is it
at least LESS CONFIDENT about it than when it's wrong on an ordinary image?
If so, the calibration this project already built (temperature scaling +
an "uncertain" band) is doing real, useful work here even though the raw
accuracy number looks bad -- the honest "I don't know" state matters as
much as the raw accuracy for this project's whole thesis.
"""
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

HERE = Path(__file__).parent
PHASE1_OUT = HERE.parent.parent / "phase1" / "output"
PHASE2_OUT = HERE.parent / "output"
OLD_PREFIX = "/kaggle/working/normalized/"
IMG_SIZE = 224
DEVICE = torch.device("cpu")


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
        return self.tf(im), 1.0 if r["label"] == "fake" else 0.0


def build_model():
    m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, 1)
    return m.to(DEVICE)


def main():
    calib = json.loads((PHASE2_OUT / "calibration.json").read_text())
    T = calib["temperature"]
    band = calib["uncertain_band"]
    lower, upper = 0.5 - band, 0.5 + band

    model = build_model()
    model.load_state_dict(torch.load(PHASE2_OUT / "model_best.pt", map_location=DEVICE))
    model.eval()

    rows = load_rows()
    sg3_rows = [r for r in rows if r["family"] == "real-vs-fake-faces-stylegan3/Fake faces"]
    real_test_rows = [r for r in rows if r["split"] == "test" and r["label"] == "real"]

    def calibrated_probs(subset_rows):
        loader = DataLoader(FORGEDataset(subset_rows), batch_size=128, shuffle=False, num_workers=2)
        out = []
        with torch.no_grad():
            for x, _ in loader:
                logits = model(x.to(DEVICE)).squeeze(1)
                out.append(torch.sigmoid(logits / T))
        return torch.cat(out)

    sg3_probs = calibrated_probs(sg3_rows)
    real_probs = calibrated_probs(real_test_rows)

    def bucket_counts(probs, true_label):
        n = len(probs)
        uncertain = ((probs >= lower) & (probs <= upper)).sum().item()
        if true_label == "fake":
            correct = (probs > upper).sum().item()
            wrong_confident = (probs < lower).sum().item()
        else:
            correct = (probs < lower).sum().item()
            wrong_confident = (probs > upper).sum().item()
        return n, correct, uncertain, wrong_confident

    n, correct, uncertain, wrong_conf = bucket_counts(sg3_probs, "fake")
    lines = [
        "# FORGE phase 2l -- confidence check on StyleGAN3 (shipped model + calibration)\n\n",
        f"Deployed model: temperature={T:.3f}, uncertain band=[{lower:.2f}, {upper:.2f}]\n\n",
        f"## StyleGAN3 held-out ({n} images)\n\n",
        f"- Correctly called fake (confident, right): {correct} ({correct/n:.1%})\n",
        f"- Called uncertain (honest 'not sure'): {uncertain} ({uncertain/n:.1%})\n",
        f"- Confidently called real (confident, WRONG): {wrong_conf} ({wrong_conf/n:.1%})\n\n",
    ]

    n2, correct2, uncertain2, wrong_conf2 = bucket_counts(real_probs, "real")
    lines += [
        f"## Real photos, test set ({n2} images), for comparison\n\n",
        f"- Correctly called real (confident, right): {correct2} ({correct2/n2:.1%})\n",
        f"- Called uncertain: {uncertain2} ({uncertain2/n2:.1%})\n",
        f"- Confidently called fake (confident, WRONG): {wrong_conf2} ({wrong_conf2/n2:.1%})\n",
    ]

    with open(HERE / "output" / "CONFIDENCE_REPORT.md", "w") as f:
        f.writelines(lines)
    print("".join(lines))


if __name__ == "__main__":
    main()
