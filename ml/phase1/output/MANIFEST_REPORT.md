# FORGE phase 1 manifest report

Total images: 35995 (real=8245, fake=27750)

Fake families (12): ai-vs-real-images-dataset/aigen, deepfake-face-images/Fake, real-and-fake-ai-generated-art-images-dataset/aigen, real-vs-fake-faces-stylegan3/Fake faces, sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models, unbiased-tiny-genimage/ADM, unbiased-tiny-genimage/BigGAN, unbiased-tiny-genimage/Midjourney, unbiased-tiny-genimage/VQDM, unbiased-tiny-genimage/glide, unbiased-tiny-genimage/stable_diffusion_v_1_5, unbiased-tiny-genimage/wukong

## Post-normalization leak recheck

| | real | fake |
|---|---|---|
| mean width | 501 | 512 |
| mean height | 502 | 512 |
| formats present | {'JPEG'} | {'JPEG'} |

Gate (leak closed): PASS
