"""Check that real/fake labeling is right for every dataset layout we ingest.

Run before pushing phase1_manifest.py:  python test_labeling.py

Exists because two separate near-misses came from guessing a label off a
folder name: (1) ai-vs-real-images-dataset stores AI-generated images under
a folder called "nature", which is in REAL_HINTS, and (2) two dataset slugs
themselves contain the substring "real", so labeling by slug flipped every
AI image to "real photo". Both would have trained on inverted labels
without erroring -- exactly the kind of bug that costs a full Kaggle run.
"""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("manifest", "phase1_manifest.py")
manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest)

CASES = [
    # ai-vs-real-images-dataset: "nature" leaf is a REAL_HINTS trap
    (("ai-vs-real-images-dataset", "Ai_generated_dataset", "nature"), "fake"),
    (("ai-vs-real-images-dataset", "Ai_generated_dataset", "animals"), "fake"),
    (("ai-vs-real-images-dataset", "Real_dataset", "nature"), "real"),
    # slug itself contains "real" -- must not leak into the label
    (("real-and-fake-ai-generated-art-images-dataset", "Data", "FAKE"), "fake"),
    (("real-and-fake-ai-generated-art-images-dataset", "Data", "REAL"), "real"),
    # the original flat layouts must keep working
    (("unbiased-tiny-genimage", "BigGAN"), "fake"),
    (("unbiased-tiny-genimage", "nature"), "real"),
    (("deepfake-face-images", "Final Dataset", "Fake"), "fake"),
    (("real-vs-fake-faces-stylegan3", "Fake faces"), "fake"),
    (("real-vs-fake-faces-stylegan3", "Real faces"), "real"),
]


def resolve(parts):
    dataset, label = manifest.find_dataset_and_label(parts)
    if dataset is None:
        return "skipped"
    return "real" if any(h in label.lower() for h in manifest.REAL_HINTS) else "fake"


def main():
    failures = 0
    for parts, expected in CASES:
        got = resolve(parts)
        if got != expected:
            failures += 1
            print(f"FAIL {'/'.join(parts)} -> {got} (expected {expected})")
    assert not failures, f"{failures} labeling case(s) wrong -- do NOT push this manifest"
    print(f"all {len(CASES)} labeling cases OK")


if __name__ == "__main__":
    main()
