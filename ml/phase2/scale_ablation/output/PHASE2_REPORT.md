# FORGE phase 2h -- data-scale ablation report

    device=cpu, train=24500 (real=3500 fake=21000, pos_weight=0.167), val=5250, test=5250, heldout=5000
    epoch 1/3  train_loss=0.0911 train_acc=0.852  val_loss=0.0481 val_acc=0.919  (600s)
    epoch 2/3  train_loss=0.0501 train_acc=0.927  val_loss=0.0510 val_acc=0.943  (587s)
    epoch 3/3  train_loss=0.0407 train_acc=0.941  val_loss=0.0496 val_acc=0.920  (588s)

## Test accuracy (families seen in training)

| family | n | accuracy |
|---|---:|---:|
| 140k-real-and-fake-faces/fake | 1500 | 0.997 |
| deepfake-face-images/Fake | 375 | 0.989 |
| real | 750 | 0.915 |
| unbiased-tiny-genimage/ADM | 375 | 0.835 |
| unbiased-tiny-genimage/BigGAN | 375 | 1.000 |
| unbiased-tiny-genimage/Midjourney | 375 | 0.864 |
| unbiased-tiny-genimage/VQDM | 375 | 0.824 |
| unbiased-tiny-genimage/glide | 375 | 0.997 |
| unbiased-tiny-genimage/stable_diffusion_v_1_5 | 375 | 0.936 |
| unbiased-tiny-genimage/wukong | 375 | 0.931 |

## Held-out accuracy (generators never trained on)

| family | n | accuracy |
|---|---:|---:|
| real | 750 | 0.915 |
| real-vs-fake-faces-stylegan3/Fake faces | 2500 | 0.210 |
| sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models | 2500 | 0.644 |

**In-distribution test accuracy: 0.942**
**Held-out (unseen generator) accuracy: 0.491**
**Generalization gap: 0.452**

Compare against the smaller-StyleGAN-dose run (in-dist 0.933, heldout 0.447, StyleGAN3 0.365, SFHQ-T2I 0.398).
