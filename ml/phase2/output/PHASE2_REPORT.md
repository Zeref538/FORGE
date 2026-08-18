# FORGE phase 2 baseline report

    device=cpu, train=17500 (real=3500 fake=14000, pos_weight=0.250), val=3750, test=3750, heldout=5000
    epoch 1/3  train_loss=0.1264 train_acc=0.855  val_loss=0.0875 val_acc=0.894  (412s)
    epoch 2/3  train_loss=0.0755 train_acc=0.923  val_loss=0.0899 val_acc=0.936  (403s)
    epoch 3/3  train_loss=0.0586 train_acc=0.941  val_loss=0.0746 val_acc=0.918  (406s)

## Test accuracy (families seen in training)

| family | n | accuracy |
|---|---:|---:|
| deepfake-face-images/Fake | 375 | 0.987 |
| real | 750 | 0.881 |
| unbiased-tiny-genimage/ADM | 375 | 0.920 |
| unbiased-tiny-genimage/BigGAN | 375 | 0.997 |
| unbiased-tiny-genimage/Midjourney | 375 | 0.893 |
| unbiased-tiny-genimage/VQDM | 375 | 0.949 |
| unbiased-tiny-genimage/glide | 375 | 0.979 |
| unbiased-tiny-genimage/stable_diffusion_v_1_5 | 375 | 0.920 |
| unbiased-tiny-genimage/wukong | 375 | 0.925 |

## Held-out accuracy (generators never trained on)

| family | n | accuracy |
|---|---:|---:|
| real | 750 | 0.881 |
| real-vs-fake-faces-stylegan3/Fake faces | 2500 | 0.365 |
| sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models | 2500 | 0.398 |

**In-distribution test accuracy: 0.933**
**Held-out (unseen generator) accuracy: 0.447**
**Generalization gap: 0.487**
