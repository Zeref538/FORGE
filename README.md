# FORGE — Fake Or Real: Generated-image Examiner

Drop in an image, get a probability that an AI generator made it rather than
a camera. **The model runs entirely in your browser** — the image is never
uploaded, there is no server, and hosting costs nothing.

The point of this project is not the accuracy number. It is the evaluation:
what happens when the detector meets a generator it was never trained on,
and what happened when a promising result turned out not to reproduce.

---

## Quick start

```bash
git clone https://github.com/Zeref538/FORGE.git
cd FORGE/web
python -m http.server 8000     # then open http://localhost:8000
```

No build step, no npm install. The site is plain HTML, CSS and JavaScript.

> Opening `index.html` directly by double-clicking will not work — browsers
> block `file://` pages from loading the model. Any local server fixes it.

## Repo layout

```
web/     the site — HTML, CSS, JS, and the deployed model
ml/      training and evaluation code (Kaggle kernels)
docs/    project brief, portfolio write-up, contributor handoffs
```

| If you want to… | Read |
|---|---|
| work on the site or deploy it | [`docs/FRONTEND_HANDOFF.md`](docs/FRONTEND_HANDOFF.md) |
| understand the research arc | [`docs/PORTFOLIO_CARD.md`](docs/PORTFOLIO_CARD.md) |
| see the original goals | [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md) |

---

## How accurate it is

EfficientNet-B0, trained on 13 generator families. Every figure below is a
test slice the model never trained on, reported per family, **never pooled
into one flattering number**.

| generator | accuracy |
|---|---:|
| BigGAN | 100.0% |
| GLIDE | 99.5% |
| SFHQ-T2I | 98.7% |
| VQDM | 97.3% |
| Wukong | 97.3% |
| StyleGAN / StyleGAN2 | 96.8% |
| Stable Diffusion 1.5 | 96.5% |
| ADM | 95.7% |
| **real photographs** | **94.6%** |
| Midjourney | 93.3% |
| AI artwork | 86.9% |
| **StyleGAN3** | **51.5%** |

**Overall: 92.7%**

Two of those deserve to be read carefully.

**Real photographs, 94.6%** means roughly **1 in 19 genuine photos is
wrongly called AI-generated.** That is the wrong direction for the error to
run — falsely accusing a real photo is worse than missing a fake.

**StyleGAN3, 51.5%** is a coin flip, on a family the model *did* train on.

## The finding

The original setup deliberately **hid two generator families from training**
to answer a harder question than "is it accurate": *what happens when
someone uses a generator that did not exist when this shipped?*

The answer was **0.000**. Not "poor" — zero. All 2,500 fakes called real.

Six attempts to fix it:

| attempt | accuracy on the unseen family |
|---|---:|
| baseline | 0.000 |
| higher input resolution | 0.000 |
| frequency-domain (FFT) input channel | 0.000 |
| blur + JPEG augmentation (Wang et al. 2020) | 0.000 |
| CLIP frozen features (Ojha et al. 2023) | 0.003 |
| adding a related GAN family to training | 0.365 |

The last one appeared to work, and **was published to the site.**

**It did not reproduce.** Re-running while logging accuracy after *every*
training epoch — instead of trusting the single saved checkpoint — showed
the number swinging between **0.005 and 0.305** on identical data. The 0.365
never appeared again, and the checkpoint the selection rule would actually
have kept scored 0.034.

The cause is structural: checkpoints are chosen by validation accuracy, and
the validation set only contains families the model trained on. The
selection process is blind to the exact thing being measured. The site was
corrected to report a range.

Three follow-ups asked whether the tool could at least *fail honestly*
instead. None worked:

- **Averaging checkpoints** erased the good epoch instead of reinforcing it,
  which confirms it was luck rather than a skill being learned.
- **The model's own calibrated confidence** is *confidently wrong* 58.2% of
  the time on the unseen family, and says "uncertain" on only 10.5%.
- **Novelty detection** (twice — on model features, then on radial frequency
  spectra) flagged under 0.5% of unseen-generator images, versus 9.7% of
  ordinary ones. The measured reason: those images sit *closer* to the
  training data than ordinary photos do. A well-made fake face does not look
  unusual — that is the point of it.

**Conclusion:** generalising to an unseen generator architecture is unsolved
here, and unsolved in the literature — the paper this compares against
reports its CNN dropping to near coin-flip on unseen architectures too. No
training trick fixed it. What worked was showing the model examples of that
architecture, which is why the shipped model trains on all 13 families.

That works for generators that exist today. It says nothing about the next
one.

## Model choice

Three backbones trained on identical data:

| backbone | params | accuracy | download | train time |
|---|---:|---:|---:|---:|
| MobileNetV3-Small | 1.5M | 90.1% | 6.1 MB | 30 min |
| **EfficientNet-B0** ← shipped | 4.0M | **92.7%** | 16.0 MB | 130 min |
| ResNet-50 | 23.5M | 90.2% | 94.0 MB | 273 min |

Bigger is not automatically better. ResNet-50 costs 6× the download for no
accuracy gain — its training accuracy was still climbing when the run
ended, so at 3 epochs it is undertrained rather than outclassed. Either way
94 MB is too large to send to a browser.

## Data

Public Kaggle datasets only — nothing scraped, no personal photos. About
36,000 images across 13 generator families. Sources are listed in
[`docs/FRONTEND_HANDOFF.md`](docs/FRONTEND_HANDOFF.md#8-where-the-data-came-from).

Every image is re-encoded to identical size and JPEG quality before
training. That fixed a real shortcut: real photos were mostly small PNGs and
fakes mostly large JPEGs, so a model could score well by recognising the
file format instead of the image.

## Reproducing

Training runs on Kaggle. Each directory under `ml/` holds a
`kernel-metadata.json` naming its datasets.

```bash
cd ml/phase1 && python -m kaggle kernels push     # build the dataset
python test_labeling.py                            # ALWAYS run before pushing
cd ../phase2/final_model && python -m kaggle kernels push
cd .. && python calibrate_efficientnet.py          # calibrate + write calibration.json
```

`test_labeling.py` takes about a second and exists because three separate
bugs silently inverted real/fake labels — one dataset stores AI images in a
folder named `nature`, which was in the real-photo hint list. None of them
crashed; all of them would have trained on wrong labels.

There is also a GitHub Action (`.github/workflows/kaggle-run.yml`) that runs
a kernel, waits, and commits the report back — so training does not need a
machine left switched on.

## What this tool cannot tell you

- It is **not forensic evidence** and cannot be used to accuse anyone of
  anything.
- It will misjudge generators released after it was trained. That gap is
  measured above, not hidden.
- Paintings, game renders and heavily filtered photos are out of scope; a
  confident-sounding number on them is still a guess.
- A missing content credential means the platform stripped it, not that the
  image is synthetic.

## Contributors

- [@Zeref538](https://github.com/Zeref538) — ML, data pipeline, evaluation
- [@Tinenen-cs](https://github.com/Tinenen-cs) — frontend
