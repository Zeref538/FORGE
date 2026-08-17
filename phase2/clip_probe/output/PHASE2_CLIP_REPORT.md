# FORGE phase 2f — CLIP linear-probe ablation report

    device=cpu, clip=ViT-B-32/openai, feat_dim=512, feature_extraction_time=1690s, train=15750 (real=3500 fake=12250, pos_weight=0.286), val=3375, test=3375, heldout=5000
    LBFGS step 1/5  loss=0.3086 train_acc=0.981  val_acc=0.960
    LBFGS step 2/5  loss=0.0224 train_acc=0.982  val_acc=0.960
    LBFGS step 3/5  loss=0.0223 train_acc=0.982  val_acc=0.960
    LBFGS step 4/5  loss=0.0223 train_acc=0.982  val_acc=0.960
    LBFGS step 5/5  loss=0.0223 train_acc=0.982  val_acc=0.960

## Test accuracy (families seen in training)

| family | n | accuracy |
|---|---:|---:|
| real | 750 | 0.952 |
| unbiased-tiny-genimage/ADM | 375 | 0.981 |
| unbiased-tiny-genimage/BigGAN | 375 | 1.000 |
| unbiased-tiny-genimage/Midjourney | 375 | 0.933 |
| unbiased-tiny-genimage/VQDM | 375 | 0.984 |
| unbiased-tiny-genimage/glide | 375 | 0.989 |
| unbiased-tiny-genimage/stable_diffusion_v_1_5 | 375 | 0.973 |
| unbiased-tiny-genimage/wukong | 375 | 0.941 |

## Held-out accuracy (generators never trained on)

| family | n | accuracy |
|---|---:|---:|
| real | 750 | 0.952 |
| real-vs-fake-faces-stylegan3/Fake faces | 2500 | 0.003 |
| sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models | 2500 | 0.382 |

**In-distribution test accuracy: 0.967**
**Held-out (unseen generator) accuracy: 0.291**
**Generalization gap: 0.676**

Compare against phase2_train.py's MobileNetV3-Small baseline (224px: in-dist 0.959, heldout 0.398, StyleGAN3 0.000, SFHQ-T2I 0.643).

CLIP preprocessing (must match if wired into the web app): Compose(
    Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
    CenterCrop(size=(224, 224))
    MaybeConvertMode()
    MaybeToTensor()
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
)

## ONNX export

Legacy exporter succeeded: 351.8 MB (compare: MobileNetV3-Small baseline is ~6 MB)
