"""
FORGE phase 1 — manifest + normalization, runs on Kaggle CPU.

Attach these datasets:
  - cartografia/unbiased-tiny-genimage
  - troykueh/real-vs-fake-faces-stylegan3
  - selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models
  - kshitizbhargava/deepfake-face-images (StyleGAN/StyleGAN2 fake faces —
    added so training sees a style-based GAN family; StyleGAN3 stays held
    out as the generalization test. GenImage's BigGAN alone didn't transfer:
    a classic conv-GAN's artifacts don't teach a style-based GAN's)

What phase0's leak audit found: real images skewed small + PNG, fake images
skewed large + JPEG — a model could learn "PNG+small=real" instead of "was
this generated". Fix: re-encode every image (real and fake alike) to the same
format/quality/size before anything is trained on it.

Also caps images per family. SFHQ-T2I alone has 124k images vs GenImage's
2500/family — left uncapped, the model would mostly learn "SFHQ" vs
"everything else", not "generated vs real". A fixed per-family cap keeps
family sizes comparable so leave-one-family-out means what it says.

Writes:
  /kaggle/working/normalized/<label>/<family>__<n>.jpg   (re-encoded images)
  /kaggle/working/manifest.csv                            (path,label,family,source_dataset,orig_w,orig_h,orig_format)
  /kaggle/working/MANIFEST_REPORT.md                       (post-normalization leak recheck)
"""
import csv
import os
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

random.seed(0)
INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
NORM_DIR = OUT / "normalized"
MANIFEST_PATH = OUT / "manifest.csv"
REPORT_PATH = OUT / "MANIFEST_REPORT.md"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
REAL_HINTS = ("real", "nature", "imagenet")
SPLIT_FOLDER_NAMES = {"train", "test", "valid", "validation", "val"}

# Match by dataset slug name, not path depth — Kaggle mounts attached
# datasets inconsistently between sessions: sometimes flat
# (/kaggle/input/<slug>/...), sometimes nested
# (/kaggle/input/datasets/<owner>/<slug>/...). Locating the slug anywhere in
# the path is robust to both; a fixed-depth walk broke on a nested-mount run.
DATASET_SLUGS = [
    "unbiased-tiny-genimage",
    "real-vs-fake-faces-stylegan3",
    "sfhq-t2i-synthetic-faces-from-text-2-image-models",
    "deepfake-face-images",
    # Added for subject-matter diversity: everything above is faces or
    # GenImage's mix, so the model had little non-face AI imagery to learn
    # from. These two bring animals/city/food/nature scenes and artwork.
    "ai-vs-real-images-dataset",
    "real-and-fake-ai-generated-art-images-dataset",
    # NOT added: 200k-real-vs-ai-visuals-by-mbilal -- its filenames
    # (00276TOPP4.jpg etc) are byte-for-byte the same as
    # 140k-real-and-fake-faces, i.e. a repackaged copy of a dataset already
    # considered here. Adding it would duplicate images across families.
]

# Datasets whose real/fake split is encoded in a WRAPPER folder rather than
# the leaf. ai-vs-real-images-dataset lays out Ai_generated_dataset/nature/
# -- and "nature" is in REAL_HINTS (it was added for a genuinely real photo
# source), so the ordinary leaf rule would have labeled AI-generated nature
# scenes as real photos and silently poisoned training. For these, scan the
# whole subpath and check fake indicators FIRST, since they are the more
# specific signal.
PATH_SCAN_DATASETS = {
    "ai-vs-real-images-dataset",
    "real-and-fake-ai-generated-art-images-dataset",
}
FAKE_HINTS = ("ai_generated", "ai-generated", "aigenerated", "fake", "ai_image")
SINGLE_FAMILY_DATASETS = {"sfhq-t2i-synthetic-faces-from-text-2-image-models"}

# These datasets' images sit one or more folders deeper than the others
# (e.g. .../deepfake-face-images/Final Dataset/Fake/*.jpg, or
# .../140k-real-and-fake-faces/real_vs_fake/real-vs-fake/train/fake/*.jpg),
# so the normal "next folder after the slug" rule would read the wrong
# wrapper folder as the label. Use the deepest folder name instead, and skip
# each dataset's Real/ folder entirely — we already have a real-image
# pipeline and don't want a second, differently-sourced real distribution
# competing with it. (140k-real-and-fake-faces also has its own train/valid
# /test split folders above "fake" -- those all collapse to the same leaf
# name "fake" and get pooled into one family, which is what we want; our own
# splits.py re-splits everything anyway.)
LEAF_LABEL_FAKE_ONLY_DATASETS = {"deepfake-face-images", "140k-real-and-fake-faces"}

PER_FAMILY_CAP = 2500          # matches GenImage's own per-family size
# Round 1 tried 140k-real-and-fake-faces capped at 10,000 (4x
# deepfake-face-images alone) -- StyleGAN3 held-out accuracy went DOWN
# (0.365 -> 0.210) even though every other number improved. Likely cause:
# with one source providing 4x the volume of the other, the model may have
# started keying on that source's specific fingerprint (its exact crop,
# compression, alignment) rather than general style-based-GAN traits, which
# doesn't transfer to StyleGAN3 -- a different fingerprint, same lineage.
# This round: cap both StyleGAN sources at the SAME size (2,500 each), so
# the model sees the style-based-GAN family from two distinct fingerprints
# in roughly equal amounts -- diversity instead of raw volume from one place.
FAMILY_CAP_OVERRIDES = {"140k-real-and-fake-faces": 2500}
NORM_SIZE = 512                # longest side, aspect preserved
NORM_QUALITY = 90              # single JPEG quality for every class


def find_dataset_and_label(path_parts):
    for i, part in enumerate(path_parts):
        for slug in DATASET_SLUGS:
            if slug == part or slug in part or part in slug:
                if slug in SINGLE_FAMILY_DATASETS:
                    next_part = path_parts[i + 1] if i + 1 < len(path_parts) else None
                    if next_part and any(h in next_part.lower() for h in REAL_HINTS):
                        return slug, next_part
                    return slug, slug
                if slug in PATH_SCAN_DATASETS:
                    sub = "/".join(path_parts[i + 1:]).lower()
                    if any(h in sub for h in FAKE_HINTS):
                        # NOT `slug` as the label: both of these slugs contain
                        # the substring "real" (ai-vs-REAL..., REAL-and-fake...),
                        # and the caller decides real-vs-fake by substring-
                        # matching REAL_HINTS against this label -- so returning
                        # the slug would flip every AI image to "real photo".
                        return slug, "aigen"       # one mixed-generator family
                    if any(h in sub for h in REAL_HINTS):
                        return slug, "real"
                    return None, None              # unlabelable -- skip rather than guess
                if slug in LEAF_LABEL_FAKE_ONLY_DATASETS:
                    leaf = path_parts[-1]
                    if any(h in leaf.lower() for h in REAL_HINTS):
                        return None, None
                    return slug, leaf
                # real-vs-fake-faces-stylegan3 ships a redundant pre-split
                # copy of itself (train/test/valid folders) alongside its
                # normal flat Real faces/Fake faces layout -- same underlying
                # images, organized twice. Ingesting both risks the exact
                # image existing under two different family names (one that
                # matches HELD_OUT_FAMILIES, one that silently doesn't and
                # leaks into training). Skip anything sitting under a split
                # wrapper folder for datasets that didn't opt into pooling
                # them (LEAF_LABEL_FAKE_ONLY_DATASETS opts in on purpose,
                # e.g. 140k-real-and-fake-faces, which has no other layout).
                between = path_parts[i + 1:-1]
                if any(p.lower() in SPLIT_FOLDER_NAMES for p in between):
                    return None, None
                # Use the deepest folder name (path_parts[-1]), not "whatever
                # comes right after the dataset name" (path_parts[i + 1]) --
                # a no-op for datasets that are already flat (e.g. GenImage,
                # where the deepest folder IS the one right after the slug).
                label = path_parts[-1]
                return slug, label
    return None, None


def walk_families():
    buckets = defaultdict(list)
    for root, dirs, files in os.walk(INPUT):
        imgs = [f for f in files if Path(f).suffix.lower() in IMG_EXT]
        if not imgs:
            continue
        rel_parts = Path(root).relative_to(INPUT).parts
        dataset, label = find_dataset_and_label(rel_parts)
        if dataset is None:
            continue
        key = (dataset, label)
        buckets[key].extend(str(Path(root) / f) for f in imgs)
    return buckets


def normalize_image(src_path, dst_path):
    im = Image.open(src_path).convert("RGB")
    w, h = im.size
    # always resize to NORM_SIZE (up or down) — capping only when larger
    # left small originals (many GenImage fakes are native 256px) at their
    # native size, so per-family resolution differences survived "normalization"
    scale = NORM_SIZE / max(w, h)
    im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst_path, format="JPEG", quality=NORM_QUALITY)
    return w, h


SPLIT_FOLDER_NAMES = {"train", "test", "valid", "validation", "val"}


def main():
    buckets = walk_families()
    # catch a dataset mounting with an unexpected extra layer (e.g. a
    # redundant train/test/valid split folder sitting where a real/fake
    # label was expected) here, in a few minutes on CPU -- not after a
    # 10-20 minute training run has already trained on the mislabeled data.
    bad = [f"{d}/{lbl}" for d, lbl in buckets if lbl.lower() in SPLIT_FOLDER_NAMES]
    assert not bad, (
        f"these bucket(s) look like raw dataset-split folders, not real/fake "
        f"labels -- a dataset mounted with an unexpected extra folder layer "
        f"this session: {bad}"
    )
    rows = []

    for (dataset, folder_label), paths in sorted(buckets.items()):
        is_real = any(h in folder_label.lower() for h in REAL_HINTS)
        label = "real" if is_real else "fake"
        family = "real" if is_real else f"{dataset}/{folder_label}"

        cap = FAMILY_CAP_OVERRIDES.get(dataset, PER_FAMILY_CAP)
        sample = paths if len(paths) <= cap else random.sample(paths, cap)
        for i, src in enumerate(sample):
            # filename must be unique per (dataset, folder_label) bucket, not
            # per collapsed `family` — both real sources map to family="real",
            # so keying on family alone made bucket 2 silently overwrite
            # bucket 1's files (2500 "real" images vanished this way before)
            safe_key = f"{dataset}-{folder_label}".replace("/", "-")
            dst = NORM_DIR / label / f"{safe_key}__{i}.jpg"
            try:
                w, h = normalize_image(src, dst)
            except Exception as e:
                continue
            fmt = Path(src).suffix.lstrip(".").upper().replace("JPG", "JPEG")
            rows.append({
                "path": str(dst),
                "label": label,
                "family": family,
                "source_dataset": dataset,
                "orig_width": w,
                "orig_height": h,
                "orig_format": fmt,
            })

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "family", "source_dataset", "orig_width", "orig_height", "orig_format"])
        writer.writeheader()
        writer.writerows(rows)

    # --- post-normalization recheck: confirm the leak is actually closed ---
    # (must run before zipping — it reads the normalized files by path)
    import numpy as np

    def stats_for(label):
        sub = [r for r in rows if r["label"] == label]
        sample = random.sample(sub, min(300, len(sub)))
        norm_sizes = [Image.open(r["path"]).size for r in sample]
        formats = set()
        for r in sample:
            formats.add(Image.open(r["path"]).format)
        widths = [s[0] for s in norm_sizes]
        heights = [s[1] for s in norm_sizes]
        return len(sub), np.mean(widths), np.mean(heights), formats

    n_real, w_real, h_real, fmt_real = stats_for("real")
    n_fake, w_fake, h_fake, fmt_fake = stats_for("fake")
    families = sorted(set(r["family"] for r in rows if r["label"] == "fake"))

    with open(REPORT_PATH, "w") as f:
        f.write("# FORGE phase 1 manifest report\n\n")
        f.write(f"Total images: {len(rows)} (real={n_real}, fake={n_fake})\n\n")
        f.write(f"Fake families ({len(families)}): {', '.join(families)}\n\n")
        f.write("## Post-normalization leak recheck\n\n")
        f.write("| | real | fake |\n|---|---|---|\n")
        f.write(f"| mean width | {w_real:.0f} | {w_fake:.0f} |\n")
        f.write(f"| mean height | {h_real:.0f} | {h_fake:.0f} |\n")
        f.write(f"| formats present | {fmt_real} | {fmt_fake} |\n\n")
        gate = fmt_real == fmt_fake == {"JPEG"} and abs(w_real - w_fake) < 20
        f.write(f"Gate (leak closed): {'PASS' if gate else 'FAIL — investigate before Phase 2'}\n")

    print(open(REPORT_PATH).read())

    # zip the normalized images so kernel-output download is one file
    # instead of ~25k individual HTTP round trips
    import shutil
    if NORM_DIR.exists():
        shutil.make_archive(str(OUT / "normalized"), "zip", root_dir=NORM_DIR)
        shutil.rmtree(NORM_DIR)


if __name__ == "__main__":
    main()
