# FORGE phase 2i -- source-diversity ablation report

    device=cpu, train=19250 (real=3500 fake=15750, pos_weight=0.222), val=4125, test=4125, heldout=5000
    epoch 1/3  train_loss=0.1176 train_acc=0.846  val_loss=0.0659 val_acc=0.920  (454s)
    epoch 2/3  train_loss=0.0694 train_acc=0.921  val_loss=0.0510 val_acc=0.941  (445s)
    epoch 3/3  train_loss=0.0536 train_acc=0.941  val_loss=0.0543 val_acc=0.925  (454s)

## Test accuracy (families seen in training)

| family | n | accuracy |
|---|---:|---:|
| 140k-real-and-fake-faces/fake | 375 | 0.928 |
| deepfake-face-images/Fake | 375 | 0.925 |
| real | 750 | 0.936 |
| unbiased-tiny-genimage/ADM | 375 | 0.901 |
| unbiased-tiny-genimage/BigGAN | 375 | 1.000 |
| unbiased-tiny-genimage/Midjourney | 375 | 0.893 |
| unbiased-tiny-genimage/VQDM | 375 | 0.917 |
| unbiased-tiny-genimage/glide | 375 | 0.992 |
| unbiased-tiny-genimage/stable_diffusion_v_1_5 | 375 | 0.939 |
| unbiased-tiny-genimage/wukong | 375 | 0.936 |

## Held-out accuracy (generators never trained on)

| family | n | accuracy |
|---|---:|---:|
| real | 750 | 0.936 |
| real-vs-fake-faces-stylegan3/Fake faces | 2500 | 0.024 |
| sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models | 2500 | 0.393 |

**In-distribution test accuracy: 0.937**
**Held-out (unseen generator) accuracy: 0.303**
**Generalization gap: 0.634**

Compare: no-GAN baseline (StyleGAN3 0.000), small single-source dose (0.365), big single-source dose 10k (0.210).
