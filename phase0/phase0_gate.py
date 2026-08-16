"""
FORGE phase 0 gate — runs entirely on Kaggle CPU, no GPU.

Attach these datasets to the kernel before running:
  - cartografia/unbiased-tiny-genimage   (7 generator families + ImageNet real)
  - troykueh/real-vs-fake-faces-stylegan3 (extra GAN family)
  - selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models (newer T2I)

Writes /kaggle/working/AUDIT.md with: per-family counts, null-test accuracy,
leak-audit table. Everything is written incrementally so a 12h kill doesn't
lose the run (KAGGLE-PLATFORM-NOTES.md).
"""
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

random.seed(0)
INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
OUT.mkdir(exist_ok=True)
REPORT = OUT / "AUDIT.md"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Substring match against the top-level folder name (case-insensitive) —
# "Real faces", "Nature", "real" etc. all count as real.
REAL_HINTS = ("real", "nature", "imagenet")

# Datasets whose entire top-level folder tree is one generator family
# (subfolders are just that dataset's internal splits, not separate models).
SINGLE_FAMILY_DATASETS = {
    "sfhq-t2i-synthetic-faces-from-text-2-image-models": "sfhq-t2i",
}


def walk_families():
    """Bucket every image under /kaggle/input by (dataset, family label)."""
    buckets = defaultdict(list)
    for dataset_dir in sorted(INPUT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        single_family = SINGLE_FAMILY_DATASETS.get(dataset_dir.name)
        for root, dirs, files in os.walk(dataset_dir):
            imgs = [f for f in files if Path(f).suffix.lower() in IMG_EXT]
            if not imgs:
                continue
            rel = Path(root).relative_to(dataset_dir)
            folder_label = rel.parts[0] if rel.parts else dataset_dir.name
            if single_family and not any(h in folder_label.lower() for h in REAL_HINTS):
                label = single_family
            else:
                label = folder_label
            key = (dataset_dir.name, label)
            buckets[key].extend(str(Path(root) / f) for f in imgs)
    return buckets


def write_incremental(text):
    with open(REPORT, "a") as f:
        f.write(text)


def main():
    if REPORT.exists():
        REPORT.unlink()

    write_incremental("# FORGE phase 0 gate report\n\n")

    # --- 1. generator diversity audit -------------------------------
    buckets = walk_families()
    write_incremental("## 1. Generator diversity audit\n\n")
    write_incremental("| dataset | label | count |\n|---|---|---:|\n")
    fake_families = {}
    real_paths = []
    for (dataset, label), paths in sorted(buckets.items()):
        write_incremental(f"| {dataset} | {label} | {len(paths)} |\n")
        if any(h in label.lower() for h in REAL_HINTS):
            real_paths.extend(paths)
        else:
            fake_families[f"{dataset}/{label}"] = paths

    n_families = len(fake_families)
    write_incremental(
        f"\n**Distinct fake-generator families found: {n_families}** "
        f"(need >=4 to proceed). Real images pooled: {len(real_paths)}.\n\n"
    )
    gate1 = n_families >= 4 and len(real_paths) >= 1000
    write_incremental(f"Gate 1 (diversity): {'PASS' if gate1 else 'FAIL'}\n\n")

    if not gate1:
        write_incremental("STOP: insufficient generator diversity. Do not proceed to Phase 1.\n")
        print(REPORT.read_text())
        return

    # --- 2. null test: split real set in half, label arbitrarily -----
    write_incremental("## 2. Null test (real vs real)\n\n")
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    def cheap_features(path, n=200):
        try:
            im = Image.open(path).convert("RGB").resize((32, 32))
            return np.asarray(im).flatten() / 255.0
        except Exception:
            return None

    sample = random.sample(real_paths, min(2000, len(real_paths)))
    feats, labels = [], []
    for i, p in enumerate(sample):
        f = cheap_features(p)
        if f is None:
            continue
        feats.append(f)
        labels.append(0 if i % 2 == 0 else 1)  # arbitrary split, no real signal

    X = np.array(feats)
    y = np.array(labels)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    write_incremental(f"Real-vs-real accuracy: {acc:.3f} (target: ~0.50)\n\n")
    gate2 = abs(acc - 0.5) < 0.08
    write_incremental(f"Gate 2 (null test): {'PASS' if gate2 else 'FAIL'}\n\n")
    if not gate2:
        write_incremental(
            "STOP: pipeline leaks signal on a label with no real difference. "
            "Fix before trusting any later number.\n"
        )

    # --- 3. leak audit: resolution / aspect / format across classes --
    write_incremental("## 3. Leak audit (real vs pooled fake)\n\n")

    def stats_for(paths, n=500):
        sub = random.sample(paths, min(n, len(paths)))
        sizes, formats = [], []
        for p in sub:
            try:
                im = Image.open(p)
                sizes.append(im.size)
                formats.append(im.format)
            except Exception:
                continue
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        aspects = [w / h for w, h in sizes if h]
        fmt_counts = defaultdict(int)
        for f in formats:
            fmt_counts[f] += 1
        return {
            "n": len(sizes),
            "width_mean": np.mean(widths) if widths else 0,
            "height_mean": np.mean(heights) if heights else 0,
            "aspect_mean": np.mean(aspects) if aspects else 0,
            "formats": dict(fmt_counts),
        }

    all_fake_paths = [p for ps in fake_families.values() for p in ps]
    real_stats = stats_for(real_paths)
    fake_stats = stats_for(all_fake_paths)
    write_incremental("| | real | fake |\n|---|---|---|\n")
    write_incremental(f"| n sampled | {real_stats['n']} | {fake_stats['n']} |\n")
    write_incremental(f"| mean width | {real_stats['width_mean']:.0f} | {fake_stats['width_mean']:.0f} |\n")
    write_incremental(f"| mean height | {real_stats['height_mean']:.0f} | {fake_stats['height_mean']:.0f} |\n")
    write_incremental(f"| mean aspect | {real_stats['aspect_mean']:.3f} | {fake_stats['aspect_mean']:.3f} |\n")
    write_incremental(f"| formats | {real_stats['formats']} | {fake_stats['formats']} |\n\n")
    write_incremental(
        "Any large divergence above (formats especially) is a shortcut the model "
        "can learn instead of \"was it generated\" — fix in Phase 1 (e.g. re-encode "
        "everything to a common format/quality before training).\n\n"
    )

    # --- 4. degradation harness smoke test ----------------------------
    write_incremental("## 4. Degradation harness smoke test\n\n")
    from io import BytesIO

    def degrade(im, jpeg_quality=None, resize_to=None):
        if resize_to:
            im = im.resize(resize_to)
        if jpeg_quality:
            buf = BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
            buf.seek(0)
            im = Image.open(buf)
        return im

    probe = Image.open(sample[0]) if sample else None
    if probe:
        for q in (95, 75, 50, 25):
            d = degrade(probe, jpeg_quality=q)
            write_incremental(f"- JPEG q{q}: ok, size={d.size}\n")
        for scale in (0.5, 0.25):
            w, h = probe.size
            d = degrade(probe, resize_to=(int(w * scale), int(h * scale)))
            write_incremental(f"- resize x{scale}: ok, size={d.size}\n")
    write_incremental("\nHarness reusable as-is for Phase 2 evaluation.\n\n")

    # --- summary -------------------------------------------------------
    overall = gate1 and gate2
    write_incremental(f"## Verdict: {'GO' if overall else 'NO-GO'}\n")
    print(REPORT.read_text())


if __name__ == "__main__":
    main()
