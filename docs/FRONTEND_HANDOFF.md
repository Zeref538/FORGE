# Frontend handoff — FORGE

**For:** Christine Tamayo ([@Tinenen-cs](https://github.com/Tinenen-cs))
**From:** John Andrei Martinez ([@Zeref538](https://github.com/Zeref538))
**Your area:** everything in `web/`

Welcome. This doc is meant to be the only thing you need to read before you
can run the site, deploy it, and change it safely. Skim the first two
sections to get going; the rest is reference for when you need it.

---

## 1. What FORGE is, in one paragraph

You drop in a photo. It tells you whether a camera took it or an AI image
generator made it, as a probability. The unusual part: **the AI model runs
inside the visitor's own browser.** Nothing is uploaded, there is no server,
and there is no backend to maintain. That is a deliberate design choice — it
gives a real privacy guarantee and makes hosting free.

The second unusual part is the tone. This project reports the numbers that
make it look bad, on purpose. There is a whole section on the site about
what it gets wrong. Please keep that spirit if you redesign — details in
§7.

---

## 2. Run it on your machine (2 minutes)

You need Python installed. That is only to serve files — there is no Python
in the actual site.

```bash
git clone https://github.com/Zeref538/FORGE.git
cd FORGE/web
python -m http.server 8000
```

Open <http://localhost:8000>. Drop in any photo.

**Why a server instead of just opening index.html?** If you double-click the
file, the browser opens it as `file://`, and browser security rules block
`file://` pages from loading the model file. You get a silent failure and a
console error. Any local server fixes it — `python -m http.server` is just
the one with no install step. VS Code's Live Server extension works too.

**First load takes a few seconds.** The model is a 16 MB download. After
that the browser caches it and it is instant.

---

## 3. Deploying (Cloudflare Pages — recommended)

There is no build step. The site is plain files, so deploying is just
pointing a host at the `web/` folder.

### One-time setup

1. Go to <https://dash.cloudflare.com> and make a free account.
2. **Workers & Pages** → **Create** → **Pages** tab → **Connect to Git**.
3. Authorize GitHub and pick `Zeref538/FORGE`.
4. Build settings — this is the only screen where the values matter:

   | Field | Value |
   |---|---|
   | Framework preset | **None** |
   | Build command | **leave completely empty** |
   | Build output directory | **`web`** |

5. **Save and Deploy.**

You get a URL like `forge.pages.dev`. **From then on, every push to `main`
redeploys automatically** — you never touch the dashboard again.

The build output directory is the setting people get wrong. It tells
Cloudflare "treat this folder as the website root", so `web/index.html`
becomes the homepage. Point it at the repo root instead and visitors get a
directory listing of Python files.

### Why Cloudflare over the alternatives

Every visitor downloads a 16 MB model, so bandwidth is the thing to watch.

| Host | Free bandwidth | Verdict |
|---|---|---|
| **Cloudflare Pages** | unlimited | best fit |
| GitHub Pages | 100 GB/mo soft limit | fine (~6,000 visits), simplest if you already live in GitHub |
| Netlify | 100 GB/mo hard limit | works, but it will cut off |
| Vercel | 100 GB/mo | same |

If you prefer GitHub Pages: Settings → Pages → Deploy from branch → `main`
→ folder `/web`. Note that GitHub Pages **ignores the `_headers` file**, so
you lose the caching rules described in §6.

---

## 4. Tech stack (deliberately boring)

| Layer | What we use | Why |
|---|---|---|
| Markup | plain HTML | one file, no build |
| Styling | plain CSS with custom properties (variables) | dark mode via `prefers-color-scheme` |
| Logic | vanilla JavaScript | no framework |
| ML runtime | [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) 1.19 from a CDN | runs the model in-browser |
| Model format | ONNX (`.onnx`) | a portable format any runtime can load |
| Hosting | static files | no server, no database |
| Cost | ₱0 | free tiers all the way down |

**No npm, no build step, no `node_modules`.** You edit `index.html` and
refresh. That is the whole loop.

You are very welcome to introduce a framework (React, Vue, Astro, whatever
you are comfortable with) if you would rather work that way. Two things to
keep if you do:

- Keep the ONNX model loading client-side. It is the project's whole point.
- Keep the built output landing in a folder the host serves, and update the
  Cloudflare build settings to match.

---

## 5. The files you own

```
web/
├── index.html    the entire site — markup, CSS and JS in one file
├── _headers      caching rules for Cloudflare (see §6)
└── model/
    ├── forge_model.onnx    the trained model, 16 MB — do not edit
    └── calibration.json    two numbers that tune the verdict — do not edit
```

Everything else in the repo is machine-learning code (`ml/`) and docs
(`docs/`). You never need to touch those.

### Inside index.html

Roughly in order:

| Lines (approx) | What |
|---|---|
| `<style>` block | design tokens (colours), then layout |
| `nav`, `header.hero` | header and the drop zone |
| `section.results` | the three result panels |
| `section.trust` | accuracy charts — **numbers here are real, see §7** |
| `section.limits` | "what this cannot tell you" |
| `<script>` at the bottom | model loading and inference |

### The JavaScript bits not to break

```js
const LOWER = 0.5 - calib.uncertain_band;   // below this  -> "Likely Camera"
const UPPER = 0.5 + calib.uncertain_band;   // above this  -> "Likely Generated"
                                            // between     -> "Uncertain"
```

Three things the script must keep doing, or predictions silently go wrong:

1. **Resize the image to exactly 224×224.** The model was trained at that
   size and will produce nonsense at any other.
2. **Normalize using the ImageNet mean/std values** already in the code.
   These are not arbitrary — they must match training exactly.
3. **Apply the temperature** from `calibration.json` before the sigmoid.
   Skipping it makes the model overconfident.

Restyle freely. Just leave that arithmetic alone.

---

## 6. The caching rule (please read before changing the model)

`web/_headers` tells Cloudflare to cache the model for a year, so repeat
visitors do not re-download 16 MB.

That creates one trap. **If the model is ever retrained, the new file must
have a new name** (`forge_model_v2.onnx`, and update the `fetch` path in
`index.html`). If you overwrite `forge_model.onnx` in place, anyone who
visited before keeps the old model for up to a year and you will not be able
to reproduce their bug.

If you are only changing HTML or CSS, none of this applies — the page itself
is set to always re-check.

---

## 7. About the numbers on the site — please keep them honest

This is the one hard request in this handoff.

The accuracy figures in the trust section are **real measurements**, not
marketing. They come from evaluation runs whose reports live in
`ml/phase2/*/output/*.md`. Current shipped model:

| | |
|---|---|
| Overall | **92.7%** |
| Real photos | 94.6% — so about **1 in 19 real photos is wrongly called AI** |
| Worst family (StyleGAN3) | 51.5% — barely a coin flip |
| A generator it never trained on | **0%** |

Two of those look bad. They are on the site deliberately, because a detector
that hides its failure rate is worse than useless — someone could use a
confident wrong answer to accuse a real person of faking a photo.

So: **redesign the presentation however you like, but do not delete the bad
numbers, round them up, or bury them below the good ones.** If a number
changes because the model is retrained, John updates it; if you spot a
number on the site that no longer matches the reports in `ml/`, that is a
bug worth flagging.

The "Limits" section near the bottom exists for the same reason. Keep it
visible without scrolling past three screens of marketing.

---

## 8. Where the data came from

All training images are public datasets from [Kaggle](https://www.kaggle.com).
Nothing was scraped, and no personal photos were used.

| Source | What it provides |
|---|---|
| `cartografia/unbiased-tiny-genimage` | 7 diffusion-model families (Stable Diffusion, Midjourney, ADM, VQDM, GLIDE, Wukong) + BigGAN, plus real photos |
| `troykueh/real-vs-fake-faces-stylegan3` | StyleGAN3 faces + real faces |
| `selfishgene/sfhq-t2i-...` | synthetic faces from text-to-image models |
| `kshitizbhargava/deepfake-face-images` | StyleGAN / StyleGAN2 faces |
| `rhythmghai/ai-vs-real-images-dataset` | AI animals, city, food, nature scenes |
| `doctorstrange420/real-and-fake-ai-generated-art-...` | AI artwork |

13 generator families total, about 36,000 images after processing.

Every image is re-encoded to the same size and JPEG quality before training.
That sounds like a detail but it fixed a real bug: originally the real
photos were mostly small PNGs and the fakes were mostly large JPEGs, so the
model could score well by learning *"PNG means real"* — recognising the file
format rather than the image. Normalising removed that shortcut.

---

## 9. Training history, short version

Full detail is in `docs/PORTFOLIO_CARD.md`. The gist, because it explains
why the site is worded the way it is:

1. First model got **95.9%** and looked great.
2. Tested against a generator family deliberately hidden from training:
   **0.0%.** Every single one of 2,500 fakes called real.
3. Six fixes tried — bigger images, frequency analysis, training-time
   augmentation, CLIP features (twice), more data. Five did nothing.
4. The sixth appeared to work (0% → 36.5%), and it was **published to the
   site**.
5. Re-measuring properly showed that number bounced between 0.5% and 31%
   depending on which training checkpoint you happened to save. **It was
   luck, not a fix.** The site was corrected to show a range.
6. The shipped model sidesteps the problem by training on all 13 families.
   That works for generators that exist today and says nothing about the
   next one.

Model went from MobileNetV3-Small (6 MB, 90.1%) to **EfficientNet-B0
(16 MB, 92.7%)** after a three-way comparison. ResNet-50 was also tested:
94 MB for no accuracy gain, so it was rejected — too big to send to a
browser anyway.

---

## 10. Good first tasks

Pick whatever appeals; none of these need ML knowledge.

**Small**
- Mobile layout — the three result panels stack awkwardly under ~820px.
- Loading state — the first visit downloads 16 MB with no progress
  indicator. A progress bar would help a lot.
- The favicon and social preview image (Open Graph tags) are missing.

**Medium**
- Accessibility pass: keyboard navigation for the drop zone, focus styles,
  screen-reader labels, colour contrast in dark mode.
- The JPEG-degradation chart is a hand-drawn SVG with hardcoded points.
  Worth generating from the numbers instead so it cannot drift.
- Paste-from-clipboard and drag-from-another-tab support.

**Larger**
- The middle panel currently shows the resized input image and is honestly
  labelled *"not a saliency map"*. A real saliency map (highlighting which
  parts of the image drove the verdict) is unimplemented. Doing it properly
  needs ML work, so coordinate with John — but the UI for it is yours.

---

## 11. Working together

- **Branch, do not push to `main` directly.** `git checkout -b your-branch`,
  push, open a pull request.
- Anything inside `web/` is yours to decide. Anything in `ml/` please raise
  with John first.
- The GitHub Action in `.github/workflows/` runs training on a schedule and
  commits result files back. If you see commits appear that you did not
  make, that is the robot, not a conflict.
- Test in a browser before opening a PR — drop in a real photo and an AI
  image and confirm both still give sensible verdicts. Broken model wiring
  looks exactly like working model wiring until you actually try it.

Questions to John. Nothing in `web/` is precious except the three arithmetic
steps in §5 and the honesty of the numbers in §7.
