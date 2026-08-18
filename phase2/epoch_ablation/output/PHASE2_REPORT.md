# FORGE phase 2g -- longer-training ablation report

    device=cpu, epochs=8, train=17500 (real=3500 fake=14000, pos_weight=0.250), val=3750, test=3750, heldout=5000
    epoch 1/8  train_loss=0.1264 train_acc=0.855  val_loss=0.0875 val_acc=0.894  (432s)
    epoch 2/8  train_loss=0.0755 train_acc=0.923  val_loss=0.0899 val_acc=0.936  (419s)
    epoch 3/8  train_loss=0.0586 train_acc=0.941  val_loss=0.0746 val_acc=0.918  (418s)
    epoch 4/8  train_loss=0.0496 train_acc=0.949  val_loss=0.0731 val_acc=0.940  (416s)
    epoch 5/8  train_loss=0.0421 train_acc=0.956  val_loss=0.0728 val_acc=0.944  (419s)
    epoch 6/8  train_loss=0.0364 train_acc=0.962  val_loss=0.0682 val_acc=0.935  (442s)
    epoch 7/8  train_loss=0.0335 train_acc=0.966  val_loss=0.0727 val_acc=0.946  (421s)
    epoch 8/8  train_loss=0.0298 train_acc=0.971  val_loss=0.0862 val_acc=0.954  (420s)

## Test accuracy (families seen in training)

| family | n | accuracy |
|---|---:|---:|
| deepfake-face-images/Fake | 375 | 0.981 |
| real | 750 | 0.923 |
| unbiased-tiny-genimage/ADM | 375 | 0.941 |
| unbiased-tiny-genimage/BigGAN | 375 | 0.997 |
| unbiased-tiny-genimage/Midjourney | 375 | 0.936 |
| unbiased-tiny-genimage/VQDM | 375 | 0.968 |
| unbiased-tiny-genimage/glide | 375 | 0.995 |
| unbiased-tiny-genimage/stable_diffusion_v_1_5 | 375 | 0.976 |
| unbiased-tiny-genimage/wukong | 375 | 0.965 |

## Held-out accuracy (generators never trained on)

| family | n | accuracy |
|---|---:|---:|
| real | 750 | 0.923 |
| real-vs-fake-faces-stylegan3/Fake faces | 2500 | 0.026 |
| sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models | 2500 | 0.571 |

**In-distribution test accuracy: 0.961**
**Held-out (unseen generator) accuracy: 0.380**
**Generalization gap: 0.581**

Compare against phase2_train.py's 3-epoch run (in-dist 0.933, heldout 0.447, StyleGAN3 0.365, SFHQ-T2I 0.398).
