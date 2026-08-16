"""
FORGE phase 1 — split assignment. Runs anywhere (pure stdlib + no image I/O),
operates on the manifest.csv produced by phase1_manifest.py.

Splits by generator family, not by image:
  - HELD_OUT families: never touched during training, reserved for the
    leave-one-generator-out number reported in Phase 2. Chosen up front so no
    training decision can leak into that number.
  - Everything else: ordinary 70/15/15 train/val/test split, done at the
    image level within each remaining family (real images are split the
    same way).

Usage: python splits.py manifest.csv splits.csv
"""
import csv
import random
import sys
from collections import defaultdict

random.seed(0)

# Held out entirely from training — never trained on, used only to measure
# the generalization gap. Picked for family diversity: one GAN, one recent
# T2I model, distinct from the GenImage diffusion families that dominate
# the rest of the pool.
HELD_OUT_FAMILIES = {
    "real-vs-fake-faces-stylegan3/Fake faces",
    # phase1_manifest.py's single-family datasets get family = "<slug>/<slug>";
    # a bare "sfhq-t2i" here silently never matched and the family leaked
    # into ordinary train/val/test instead of being held out.
    "sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models",
}


def main(manifest_path, out_path):
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_family = defaultdict(list)
    for r in rows:
        key = r["family"] if r["label"] == "fake" else f"real/{r['source_dataset']}"
        by_family[key].append(r)

    assigned = []
    for family, frows in by_family.items():
        is_fake_heldout = family in HELD_OUT_FAMILIES
        if is_fake_heldout:
            for r in frows:
                r["split"] = "heldout"
            assigned.extend(frows)
            continue

        random.shuffle(frows)
        n = len(frows)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        for i, r in enumerate(frows):
            if i < n_train:
                r["split"] = "train"
            elif i < n_train + n_val:
                r["split"] = "val"
            else:
                r["split"] = "test"
        assigned.extend(frows)

    fieldnames = list(rows[0].keys()) + ["split"] if "split" not in rows[0] else list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(assigned)

    counts = defaultdict(int)
    for r in assigned:
        counts[r["split"]] += 1
    print(f"wrote {out_path}: {dict(counts)}")
    print(f"held-out families (never trained on): {sorted(HELD_OUT_FAMILIES)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "manifest.csv",
         sys.argv[2] if len(sys.argv) > 2 else "splits.csv")
