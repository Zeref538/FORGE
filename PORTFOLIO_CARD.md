# FORGE — portfolio card (draft)

> Draft only — matches the pattern from Munti/APAW per HANDOFF.md. Refine before
> publishing; this captures the finding while it's fresh.

## One-line pitch

A camera-vs-AI image detector that measures its own blind spots instead of
hiding them — found a total blind spot, tried six fixes, and caught itself
overclaiming the one that seemed to work before it shipped.

## The finding (the part that makes this a study, not a demo)

Trained a MobileNetV3-Small classifier on 9 generator families (7 diffusion
models from the GenImage benchmark, plus StyleGAN3 and SFHQ-T2I held out
entirely). In-distribution test accuracy: **95.9%**. Looks great.

Then it was tested against StyleGAN3 — a GAN, architecturally different from
every diffusion model it trained on. Accuracy: **0.000**. Not "worse." Zero.
Every single one of 2,500 fake faces called "real."

Six follow-up attempts, each measured. Five failed. One worked:

| attempt | in-dist acc | StyleGAN3 held-out | held-out gap |
|---|---:|---:|---:|
| baseline (224px, RGB) | 0.959 | 0.000 | 0.561 |
| higher resolution (384px) | 0.952 | 0.000 | 0.582 |
| + frequency-domain channel (FFT) | 0.949 | 0.000 | 0.616 |
| + blur/JPEG augmentation (Wang et al. 2020) | 0.933 | 0.000 | 0.624 |
| CLIP ViT-B/32, undertrained probe | 0.859 | 0.053 | 0.545 |
| CLIP ViT-B/32, properly trained probe | 0.967 | 0.003 | 0.676 (worst) |
| **+ real GAN training data (StyleGAN/2 faces)** | **0.933** | **0.365** | **0.487 (best)** |

Resolution, a frequency channel, and training-time blur/JPEG augmentation
all left StyleGAN3 at exactly 0.000 — and each made the model worse
somewhere else along the way. That consistency ruled out noise or bad luck:
the model had never been shown what a GAN artifact looks like, so no amount
of resolution or clever features taught it something it was never shown.

CLIP (Ojha et al., CVPR 2023 — freeze a model trained on generic web
images, train only a small linear layer on top of its features) got two
honest tries. First attempt used an optimizer (Adam, 30 full-batch steps)
that never properly converged — and that run happened to nudge StyleGAN3
off zero, 0.000 → 0.053. Second attempt fixed the optimizer (LBFGS, which
actually solves this convex problem instead of guessing at a learning
rate) and fixed a real ONNX export bug along the way (PyTorch's fused
attention op has no ONNX translation — disabling it with
`torch.backends.mha.set_fastpath_enabled(False)` fixed the export cleanly).
Properly trained, the probe got *better* than the baseline on images it was
trained on (0.967) and StyleGAN3 fell straight back to 0.003 — worse
generalization gap than anything else tried. The earlier "improvement"
wasn't CLIP's features helping; it was an undertrained classifier
accidentally guessing more evenly, which stopped once it was allowed to
actually fit the data. And even with the export bug fixed, the file came
out to 351.8 MB — 58x the MobileNet baseline's ~6 MB — which rules it out
for an in-browser, no-upload tool on its own, independent of accuracy.

All five attempts above changed how the model looked at its existing
training images — bigger resolution, a frequency channel, augmentation,
CLIP's features. None of them worked, and the pattern was consistent
enough to point at the real cause: the training set had exactly one GAN
family in it (BigGAN, a classic convolutional GAN), and StyleGAN3 is a
completely different GAN lineage — style-based, adaptive instance
normalization, noise injected per layer. Seeing one kind of GAN didn't
teach the model what a different kind of GAN looks like, no matter how
the input was reshaped.

The sixth attempt changed the training data instead: added ~2,500 fake
faces from a StyleGAN/StyleGAN2 dataset (same architectural lineage as
StyleGAN3, just earlier generations of it) as a new training-only family.
StyleGAN3 itself stayed completely held out — the model never saw it, only
a relative of it. First measurement: StyleGAN3 held-out accuracy went from
0.000 to 0.365. First genuine movement out of six tries, and by a wide
margin.

**That number didn't hold up, and finding out why is the more important
result.** A single accuracy number comes from whichever training round
("epoch" — one full pass over the training images) happened to score best
on a validation set. The problem: that validation set only contains
generator families the model actually trained on. It has no way to see
StyleGAN3 at all, so "best checkpoint" and "best at the one thing we
actually care about" are not the same thing — the selection process is
structurally blind to the metric the whole project is about. To check
whether 0.365 was real, the exact same recipe was retrained with StyleGAN3
accuracy logged after *every* epoch instead of just the one that got kept:

| epoch | StyleGAN3 held-out |
|---:|---:|
| 1 | 0.005 |
| 2 | 0.305 |
| 3 | 0.013 |
| 4 | 0.074 |
| 5 | 0.016 |
| 6 | 0.063 |
| 7 | 0.024 |
| 8 | 0.034 |

0.365 never reproduced. The closest any epoch got was 0.305, and the
epoch validation would have actually picked (highest val accuracy) scored
0.034 — near the original 0.000 baseline, not the shipped number. The
honest range observed across a clean, verified run: **roughly 0.5% to
31%**, not a stable 37%. SFHQ-T2I (the other held-out family) swung just as
hard, 0.280 to 0.709, in the same run. Both numbers on the live site were
corrected to reflect this range rather than the single lucky draw.

Worth naming directly: while chasing this down, a separate real bug
surfaced and got fixed before it could corrupt further results — one
source dataset mounted with a redundant, differently-organized copy of
itself, and the manifest pipeline's folder-parsing rule silently mislabeled
real photos from it as "fake" and let them leak into training instead of
staying held out. A cheap assertion now runs at data-build time (seconds)
to catch that class of bug before it can waste a 10-20 minute training run
on corrupted data again.

The real finding, reframed: adding a related GAN family to training does
give the model *some* ability to catch a fake type it never trained on —
up to roughly 30% on a good run, versus a flat 0% with no GAN exposure at
all. But that ability doesn't reproduce reliably with the current training
setup, because nothing in the process ever checks or optimizes for it.
"Train on more generators" is still the right instinct — but validating a
fix on a held-out family requires measuring across multiple epochs (or
seeds), not trusting whichever single number a validation-blind selection
rule happens to keep.

## If accuracy can't be fixed, can the tool at least fail honestly?

Three follow-ups, all run locally against the shipped model, asking whether
the safety nets catch what the classifier misses. All three say no, each
for a different and specific reason:

**Averaging several epochs together** (both prediction-averaging and
SWA-style weight-averaging, over four different epoch windows). If epoch
2's 0.305 came from a real skill the model was building, neighbouring
epochs would share some of it and averaging would preserve it. Averaging
all 8 dropped StyleGAN3 to 0.019-0.041 — *worse* than most individual
epochs. The good epoch wasn't a skill being learned; training briefly
wandered somewhere lucky and wandered back out.

**The model's own confidence.** Calibration (temperature scaling + an
"uncertain" band) is already built into this project, so the natural hope
is that wrong answers at least come with a hedge. They don't: on 2,500
StyleGAN3 fakes the shipped model is *confidently wrong* on **58.2%** —
it states "real" with no hedge at all — and says "uncertain" on only 10.5%.
On ordinary real photos it's confidently wrong just 8.8% of the time.
Temperature scaling teaches honest confidence about the kinds of images
in training; it has no mechanism to notice an image unlike any of them.

**Novelty detection (out-of-distribution flagging), twice.** The idea:
skip the classifier entirely, measure how *unfamiliar* an image looks
(k-nearest-neighbour distance in feature space, Sun et al. 2022) and force
"uncertain" when it's far from anything in training. Attempt one used the
classifier's own penultimate features: flagged **0.0%** of StyleGAN3, versus
9.7% of ordinary in-distribution images. Attempt two swapped in radial
power spectra (a GAN-forensics standard — collapse the FFT into energy-per-
frequency-ring, which keys on *how* an image was built rather than what's
in it): **0.2%**.

The measured reason both failed is the most interesting result in this
section. StyleGAN3's distance-to-training-data is *lower* than ordinary
test images — median 0.000316 vs 0.000418, max 0.001179 vs 0.015419. The
held-out families are not outliers; they are 2,500 aligned, centered,
evenly-lit face crops, i.e. **far more uniform than the training mix**
(which spans landscapes, objects, and scenes from GenImage). "Does this
look unusual?" is structurally the wrong question — a well-made fake face
is not unusual-looking, that's the point of it. The tell lives in a signal
neither general-purpose features nor radial spectra capture.

## Why this is the throughline with Refusal Calibration / APAW

Same discipline: report the number that makes the tool look worst, not the
one that makes it look finished. A confident wrong answer is worse than "I
don't know" — here that shows up as leave-one-generator-out evaluation
instead of a flattering pooled accuracy score.

## Shape of the build

- MobileNetV3-Small, ONNX-exported, runs entirely in the browser via ONNX
  Runtime Web — nothing uploaded, verified end-to-end in a live test.
- Temperature-scaled calibration + an evidence-based "uncertain" band, not
  forced real/fake.
- Trained on 10 generator families (8 diffusion/GenImage + StyleGAN/2 +
  BigGAN), StyleGAN3 and SFHQ-T2I held out entirely for the generalization
  numbers above.
- Degradation curve measured (not assumed): accuracy holds from 0.933
  (original) to 0.853 at heavy JPEG recompression (q25) — a real decline,
  reported as-is rather than smoothed over.
- ₱0 cost: Kaggle for data/training, static hosting for the site.

## Open items before this is publish-ready

- Six ablations complete (resolution, frequency channel, augmentation,
  CLIP x2, real GAN data), plus the stability check that turned the sixth
  result from a single number into an honest range. See table above.
- The shipped model's held-out numbers are currently reported as ranges
  (StyleGAN3 ~0.5-31%, SFHQ-T2I ~28-71%), not single points, because
  single-epoch measurement was shown to be unreliable. A better fix than
  reporting a range: change model selection to track held-out accuracy
  during training (not just validation accuracy) and pick a checkpoint on
  that basis, or report an average across several seeds. Not yet done.
- Two more data-recipe experiments (a larger single-source StyleGAN dose,
  and two StyleGAN sources at equal size) were run and produced results
  worth noting only as an aside: neither beat the original small single
  dose, but both ran on data later found to be affected by the mounting
  bug described above, before the fix was verified clean. Not reliable
  enough to report as findings; would need rerunning on verified-clean data
  to mean anything.
- Saliency heatmap: not implemented (panel currently shows the resized
  input image, honestly labeled as not a heatmap).
- Screenshots to `Portfolio/source-assets/FORGE/`, README.md for the indexer.
