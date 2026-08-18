# FORGE phase 0 gate report

## 1. Generator diversity audit

| dataset | label | count |
|---|---|---:|
| real-vs-fake-faces-stylegan3 | Fake faces | 10000 |
| real-vs-fake-faces-stylegan3 | Real faces | 10000 |
| sfhq-t2i-synthetic-faces-from-text-2-image-models | sfhq-t2i | 124763 |
| unbiased-tiny-genimage | ADM | 2500 |
| unbiased-tiny-genimage | BigGAN | 2500 |
| unbiased-tiny-genimage | Midjourney | 2500 |
| unbiased-tiny-genimage | Nature | 5828 |
| unbiased-tiny-genimage | VQDM | 2500 |
| unbiased-tiny-genimage | glide | 2500 |
| unbiased-tiny-genimage | stable_diffusion_v_1_5 | 2500 |
| unbiased-tiny-genimage | wukong | 2500 |

**Distinct fake-generator families found: 9** (need >=4 to proceed). Real images pooled: 15828.

Gate 1 (diversity): PASS

## 2. Null test (real vs real)

Real-vs-real accuracy: 0.492 (target: ~0.50)

Gate 2 (null test): PASS

## 3. Leak audit (real vs pooled fake)

| | real | fake |
|---|---|---|
| n sampled | 500 | 500 |
| mean width | 345 | 906 |
| mean height | 343 | 906 |
| mean aspect | 1.004 | 1.000 |
| formats | {'JPEG': 188, 'PNG': 312} | {'JPEG': 466, 'PNG': 34} |

Any large divergence above (formats especially) is a shortcut the model can learn instead of "was it generated" â€” fix in Phase 1 (e.g. re-encode everything to a common format/quality before training).

## 4. Degradation harness smoke test

- JPEG q95: ok, size=(486, 500)
- JPEG q75: ok, size=(486, 500)
- JPEG q50: ok, size=(486, 500)
- JPEG q25: ok, size=(486, 500)
- resize x0.5: ok, size=(243, 250)
- resize x0.25: ok, size=(121, 125)

Harness reusable as-is for Phase 2 evaluation.

## Verdict: GO
