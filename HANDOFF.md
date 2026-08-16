# FORGE — session handoff

**FORGE** — **F**ake **O**r **R**eal: **G**enerated-image **E**xaminer.

> Context-only handoff, in the pattern of `LiitLLM/HANDOFF.md`. The owner writes
> the PRD and PLAN himself in this folder. Nothing here is committed to the
> portfolio repo.

---

## What it is

A website where you drop in an image and it tells you whether a camera made it or
a generative model did — with a calibrated probability, an honest *uncertain*
band, and a plain statement of what it cannot tell you.

The detector is the easy half. **The project is the half that everyone skips:**
proving the detector still works on a generator it was never trained on, and on
images that have been through the compression grinder of the real internet.

## Why this one is portfolio-worthy (and why most versions aren't)

There are hundreds of "AI image detector" repos. Almost all of them report a
single number — accuracy on a held-out split of the same dataset they trained on
— and that number is close to meaningless, because train and test contain images
from the same generators. The model learns *"this is what Stable Diffusion looks
like"*, not *"this is what synthetic looks like"*. Point it at a generator that
came out last month and it collapses.

So the deliverable has the same shape as Refusal Calibration and APAW:

1. **Leave-one-generator-out evaluation.** Train on generators A, B, C; test on D,
   never seen. Report per-generator, never pooled. The pooled number will be
   flattering and it is the one to distrust.
2. **A robustness axis, not just an accuracy axis.** Real images arrive
   screenshotted, re-compressed, resized, re-uploaded. Most detection cues die
   under JPEG q75. Report accuracy at each degradation level — a detector that
   only works on pristine PNGs does not work.
3. **Calibration and abstention.** Reuse the Refusal Calibration discipline
   directly: a confident wrong answer is worse than "I don't know". Report ECE,
   and give the UI a real third state instead of forcing real/fake.
4. **Publish the limits**, permanently, the way APAW does. Not forensic evidence,
   cannot be used to accuse anyone, will be wrong on generators released after
   training.

That combination is a study rather than a demo, and it's directly continuous with
three cards already on the site — which is itself the story: *honest evaluation is
the throughline; the domain keeps changing.*

## The honest hard parts — read before planning

- **Generalisation is genuinely unsolved.** Detectors that score 99% in their own
  paper drop to 50–70% on unseen generators. Do not plan around beating that. Plan
  around *measuring* it. "My detector loses 25 points out of distribution, here is
  the curve" is a stronger artifact than a fake 99%.
- **Post-processing destroys the signal.** Frequency-domain and upsampling
  artifacts are the most reliable cues and the most fragile. Recompression isn't
  an edge case, it's the normal path an image takes to reach you.
- **Dataset shortcuts are everywhere.** If the real images are COCO photos and the
  fakes are 512×512 square renders, the model learns *aspect ratio*. Any
  systematic difference that isn't "was it generated" is a leak — resolution,
  colour profile, JPEG quantisation tables, even file ordering. Expect to find at
  least one and to have to fix it.
- **Metadata is not the answer, but it's part of it.** C2PA / Content Credentials
  are the industry direction and are cheap to read. A valid credential is strong
  evidence; **absence proves nothing** — every social platform strips it. Show it
  as a separate signal, never folded into the model's probability.
- **Reviewers will test it adversarially.** Someone will upload a painting, a game
  screenshot, a heavily-filtered selfie. Decide now what the page says for
  out-of-scope input instead of letting it emit a confident number.

## Shape of the build

**Model.** A small ImageNet-pretrained backbone fine-tuned as a binary classifier
is the sane baseline — free-tier friendly, a couple of hours on a T4. A
frequency-domain input branch is worth adding as an *ablation arm*, measured, not
as an assumed improvement.

**The interesting engineering decision: run inference in the browser.** Export to
ONNX, run with ONNX Runtime Web. That buys three things at once — zero hosting
cost, no upload size limits, and a real privacy claim (*the image never leaves
your device*), which is the correct property for a tool people will feed personal
photos into. It also fits the existing stack; `ONNX` is already in the portfolio's
skill-icon map.

**Data.** Real: Open Images, COCO, RAISE, Flickr-sourced sets. Synthetic: the
public generated-image datasets on Hugging Face and Kaggle, deliberately spanning
several generator families so leave-one-out is even possible. **Getting the fake
side to span 4+ generator families is the constraint that decides whether this
project works.** Check it before anything else.

**Cost.** ₱0 is achievable — Kaggle T4 for training, HF/Kaggle for data, GitHub
Pages or Vercel for a static site with in-browser inference. No keys, no inference
server.

## Suggested phase 0 — the gate

Before any GPU time, prove on CPU in an afternoon:

- at least 4 distinct generator families are actually obtainable, with counts;
- **the null test** — split the real set in half, label it arbitrarily, train
  real-vs-real. Accuracy must land at ~50%. If it doesn't, the pipeline leaks and
  every later number is fiction;
- **a leak audit** — compare resolution, aspect ratio, format and JPEG quality
  distributions across the two classes, and report what differs;
- **a degradation harness** (resize → JPEG at several qualities → screenshot
  simulation) that the final evaluation reuses unchanged.

If generator diversity isn't there, stop and say so. That's a legitimate outcome
and it's cheap now; it's expensive after 20 GPU-hours.

## What the site should show

Not just a percentage. Three panels:

- **The verdict**, with *uncertain* as a first-class outcome, not an error state.
- **What the model is going off** — a saliency heatmap, labelled honestly as a
  heatmap and not as proof — plus the C2PA credential if one survived.
- **How much to trust it** — accuracy on generators it has seen, accuracy on the
  one it hasn't, and the degradation curve. The number that makes the tool look
  worst is the number that makes the project look best.

## Open decisions for the owner

1. Scope — images only, or a stretch goal of a "how was this likely made" breakdown?
2. Pipeline slot — competes with Kalis / YODA-mini / Tipid / Carson. It is a bigger
   project than it looks, because most of the work is data, not modelling.
3. Is in-browser inference a hard requirement (it caps model size) or a nice-to-have?
4. Does it get its own portfolio chip, or go under `ML & Forecasting` with ACRA —
   currently the only other CV project there?

## Portfolio integration (much later)

Write a `PORTFOLIO_CARD.md` in this folder the way Munti and APAW did, plus a
`README.md` here so zeref-bot indexes it — `scripts/build-index.mjs` reads
`source-assets/<Project>/README.md` **only**, other filenames are ignored.
Screenshots go to `Portfolio/source-assets/FORGE/`, converted to
`public/projects/*.jpg` at ~1000px q82. Use LF line endings or the indexer's
paragraph split still works but check the chunk count — CRLF was a real bug here.