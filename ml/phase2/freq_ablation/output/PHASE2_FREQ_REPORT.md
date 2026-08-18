# FORGE phase 2d â€” frequency-domain ablation report

    device=cpu, in_channels=4 (RGB+freq), img_size=224, train=15750 (real=3500 fake=12250, pos_weight=0.286), val=3375, test=3375, heldout=5000
    epoch 1/3  train_loss=0.1095 train_acc=0.900  val_loss=0.0780 val_acc=0.932  (396s)
    epoch 2/3  train_loss=0.0632 train_acc=0.944  val_loss=0.0716 val_acc=0.934  (393s)
    epoch 3/3  train_loss=0.0461 train_acc=0.961  val_loss=0.0657 val_acc=0.938  (396s)

## Test accuracy (families seen in training)

| family | n | accuracy |
|---|---:|---:|
| real | 750 | 0.956 |
| unbiased-tiny-genimage/ADM | 375 | 0.917 |
| unbiased-tiny-genimage/BigGAN | 375 | 0.997 |
| unbiased-tiny-genimage/Midjourney | 375 | 0.920 |
| unbiased-tiny-genimage/VQDM | 375 | 0.923 |
| unbiased-tiny-genimage/glide | 375 | 1.000 |
| unbiased-tiny-genimage/stable_diffusion_v_1_5 | 375 | 0.947 |
| unbiased-tiny-genimage/wukong | 375 | 0.928 |

## Held-out accuracy (generators never trained on)

| family | n | accuracy |
|---|---:|---:|
| real | 750 | 0.956 |
| real-vs-fake-faces-stylegan3/Fake faces | 2500 | 0.000 |
| sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models | 2500 | 0.481 |

**In-distribution test accuracy: 0.949**
**Held-out (unseen generator) accuracy: 0.334**
**Generalization gap: 0.616**

Compare against phase2_train.py's RGB-only baseline (224px: in-dist 0.959, heldout 0.398, StyleGAN3 0.000, SFHQ-T2I 0.643).
