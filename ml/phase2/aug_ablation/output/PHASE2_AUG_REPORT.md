# FORGE phase 2e â€” augmentation ablation report

    device=cpu, aug=blur+jpeg@0.5each, img_size=224, train=15750 (real=3500 fake=12250, pos_weight=0.286), val=3375, test=3375, heldout=5000
    epoch 1/3  train_loss=0.1504 train_acc=0.856  val_loss=0.1124 val_acc=0.866  (373s)
    epoch 2/3  train_loss=0.1085 train_acc=0.899  val_loss=0.0951 val_acc=0.891  (370s)
    epoch 3/3  train_loss=0.0915 train_acc=0.918  val_loss=0.0922 val_acc=0.918  (360s)

## Test accuracy (families seen in training)

| family | n | accuracy |
|---|---:|---:|
| real | 750 | 0.932 |
| unbiased-tiny-genimage/ADM | 375 | 0.875 |
| unbiased-tiny-genimage/BigGAN | 375 | 0.957 |
| unbiased-tiny-genimage/Midjourney | 375 | 0.907 |
| unbiased-tiny-genimage/VQDM | 375 | 0.939 |
| unbiased-tiny-genimage/glide | 375 | 0.960 |
| unbiased-tiny-genimage/stable_diffusion_v_1_5 | 375 | 0.955 |
| unbiased-tiny-genimage/wukong | 375 | 0.941 |

## Held-out accuracy (generators never trained on)

| family | n | accuracy |
|---|---:|---:|
| real | 750 | 0.932 |
| real-vs-fake-faces-stylegan3/Fake faces | 2500 | 0.000 |
| sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models | 2500 | 0.430 |

**In-distribution test accuracy: 0.933**
**Held-out (unseen generator) accuracy: 0.309**
**Generalization gap: 0.624**

Compare against phase2_train.py's no-augmentation baseline (224px: in-dist 0.959, heldout 0.398, StyleGAN3 0.000, SFHQ-T2I 0.643).
