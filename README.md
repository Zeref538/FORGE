# FORGE — Fake Or Real: Generated-image Examiner

Drop in an image, get a probability that it was made by an AI image
generator rather than a camera. The model runs **entirely in your browser** —
the image is never uploaded anywhere.

The interesting part of this project is not the accuracy number. It is the
evaluation: what happens when the detector meets a generator it was never
trained on, and what happened when a promising result turned out not to
reproduce.

---

## What it does

| | |
|---|---|
| **Input** | any JPG / PNG / WebP image |
| **Output** | probability the image is AI-generated, plus an explicit *uncertain* verdict when the evidence does not support a confident call |
| **Where it runs** | in the browser, via ONNX Runtime Web (~6 MB model) |
| **Cost to host** | nothing — static files, no server |

## How accurate it actually is

Trained on 11 generator families. Every number below is on a test slice the
model never trained on, reported per family and **never pooled into one
flattering figure**.

| generator | accuracy |
|---|---:|
| BigGAN | 100.0% |
| SFHQ-T2I | 98.9% |
| GLIDE | 98.9% |
| StyleGAN / StyleGAN2 | 97.9% |
| VQDM | 96.8% |
| Stable Diffusion 1.5 | 94.9% |
| Wukong | 93.9% |
| ADM | 92.8% |
| Midjourney | 92.0% |
| **real photographs** | **86.5%** |
| **StyleGAN3** | **70.9%** |

**Overall: 92.5%**

The weakest number is the one that matters most: **roughly 1 in 7 genuine
photos gets called AI-generated.** That is the wrong direction for this
error to run — wrongly accusing a real photo is worse than missing a fake.
The cause is a training set with about 5x more AI images than real ones.

It also degrades under recompression, measured rather than assumed:

| JPEG quality | accuracy |
|---|---:|
| original | 93.3% |
| q95 | 93.2% |
| q75 | 93.0% |
| q50 | 89.9% |
| q25 | 85.3% |

---

## The finding

The project's original setup deliberately **hid two generator families from
training entirely**, to answer a harder question than "is it accurate":
*what happens when someone uses a generator that did not exist when this was
trained?*

The answer was brutal. On a held-out GAN family, accuracy was **0.000** —
not "poor", zero. All 2,500 fake faces called real.

Six attempts to fix it, each measured:

| attempt | held-out GAN accuracy |
|---|---:|
| baseline | 0.000 |
| higher input resolution | 0.000 |
| frequency-domain (FFT) input channel | 0.000 |
| blur + JPEG augmentation (Wang et al. 2020) | 0.000 |
| CLIP frozen features (Ojha et al. 2023) | 0.003 |
| **adding a related GAN family to training** | **0.365** |

The sixth appeared to work. **It did not reproduce.**

Re-running it while logging accuracy after *every* training epoch instead of
trusting the single saved checkpoint showed the number swinging between
**0.005 and 0.305** on identical data. The reported 0.365 never appeared
again, and the epoch that checkpoint-selection would actually have picked
scored 0.034.

The cause is structural: checkpoints are selected on validation accuracy,
and the validation set only contains families the model trained on. The
selection process is **blind to the exact thing the project was measuring.**

Three follow-ups asked whether the tool could at least fail *honestly*
instead of accurately. None worked:

- **Averaging epoch checkpoints** erased the one good epoch instead of
  reinforcing it — so that epoch was luck, not a skill being learned.
- **The model's own calibrated confidence** is *confidently wrong* 58.2% of
  the time on the held-out family, and says "uncertain" on only 10.5%.
- **Novelty detection** (twice — on model features, and on radial frequency
  spectra) flagged under 0.5% of held-out images versus 9.7% of ordinary
  ones. Measured reason: the held-out images sit *closer* to the training
  data than ordinary test images do. They are aligned, evenly-lit face
  crops — more uniform than the training mix. "Does this look unusual?" is
  the wrong question, because a good fake does not look unusual.

**The honest conclusion:** generalizing to an unseen generator architecture
is unsolved here, and unsolved in the literature too — the paper this
compares against reports its CNN dropping to near coin-flip on unseen
architectures. What fixed it was not a training trick. It was showing the
model examples of that architecture. The shipped model does exactly that,
which is why StyleGAN3 above reads 70.9% instead of 0.000.

---

## Layout

```
phase0/            feasibility gate — runs before any GPU spend
phase1/            data pipeline: normalization, manifest, splits
  test_labeling.py regression test for real/fake labeling
phase2/            training, calibration, ONNX export, evaluation
  */output/*.md    result report from every experiment, kept as the record
  final_model/     the shipped model (trained on all families)
  stability_check/ the per-epoch measurement that caught the overclaim
web/               the browser app + deployed model
```

## Reproducing

Training runs on [Kaggle](https://www.kaggle.com) (free CPU/GPU). Each
directory holds a `kernel-metadata.json` naming its datasets.

```bash
# build the normalized dataset + manifest
cd phase1 && python -m kaggle kernels push

# always run before pushing a manifest change --
# catches label-inversion bugs in about a second
python test_labeling.py

# train the shipping model
cd ../phase2/final_model && python -m kaggle kernels push

# calibrate + export to ONNX, locally
cd .. && python calibrate_and_export.py

# serve the web app
cd ../web && python -m http.server 8000
```

## What this tool cannot tell you

- It is **not forensic evidence** and cannot be used to accuse anyone of
  anything.
- It will misjudge generators released after it was trained. That gap is
  measured above, not hidden.
- Paintings, game renders and heavily filtered photos are out of scope; a
  confident-sounding number on them is still a guess.
- A missing content credential means the platform stripped it, not that the
  image is synthetic.
