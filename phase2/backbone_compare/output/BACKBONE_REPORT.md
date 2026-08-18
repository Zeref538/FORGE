# FORGE -- backbone size comparison

    device=cpu, epochs=3, train=25196 (real=5771 fake=19425), val=5398, test=5401

    [mobilenet_v3_small] epoch 1/3 train_acc=0.804 val_acc=0.852 (605s)
    [mobilenet_v3_small] epoch 2/3 train_acc=0.869 val_acc=0.895 (589s)
    [mobilenet_v3_small] epoch 3/3 train_acc=0.889 val_acc=0.897 (580s)
    [efficientnet_b0] epoch 1/3 train_acc=0.831 val_acc=0.893 (2569s)
    [efficientnet_b0] epoch 2/3 train_acc=0.908 val_acc=0.914 (2626s)
    [efficientnet_b0] epoch 3/3 train_acc=0.932 val_acc=0.926 (2603s)
    [resnet50] epoch 1/3 train_acc=0.781 val_acc=0.857 (5537s)
    [resnet50] epoch 2/3 train_acc=0.827 val_acc=0.902 (5437s)
    [resnet50] epoch 3/3 train_acc=0.857 val_acc=0.888 (5385s)

## Summary -- accuracy vs download size

| backbone | params | overall test acc | ONNX size | train time |
|---|---:|---:|---:|---:|
| mobilenet_v3_small | 1.5M | 0.901 | 6.1 MB | 30 min |
| efficientnet_b0 | 4.0M | 0.927 | 16.0 MB | 130 min |
| resnet50 | 23.5M | 0.902 | 94.0 MB | 273 min |

(ONNX size is what a browser must download before the first prediction. ~6MB is instant; ~100MB is not.)

### mobilenet_v3_small

| family | n | accuracy |
|---|---:|---:|
| ai-vs-real-images-dataset/aigen | 38 | 0.868 |
| deepfake-face-images/Fake | 375 | 0.987 |
| real | 1238 | 0.858 |
| real-and-fake-ai-generated-art-images-dataset/aigen | 375 | 0.859 |
| real-vs-fake-faces-stylegan3/Fake faces | 375 | 0.795 |
| sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models | 375 | 0.992 |
| unbiased-tiny-genimage/ADM | 375 | 0.808 |
| unbiased-tiny-genimage/BigGAN | 375 | 0.997 |
| unbiased-tiny-genimage/Midjourney | 375 | 0.843 |
| unbiased-tiny-genimage/VQDM | 375 | 0.885 |
| unbiased-tiny-genimage/glide | 375 | 1.000 |
| unbiased-tiny-genimage/stable_diffusion_v_1_5 | 375 | 0.947 |
| unbiased-tiny-genimage/wukong | 375 | 0.949 |

### efficientnet_b0

| family | n | accuracy |
|---|---:|---:|
| ai-vs-real-images-dataset/aigen | 38 | 0.974 |
| deepfake-face-images/Fake | 375 | 0.968 |
| real | 1238 | 0.946 |
| real-and-fake-ai-generated-art-images-dataset/aigen | 375 | 0.869 |
| real-vs-fake-faces-stylegan3/Fake faces | 375 | 0.515 |
| sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models | 375 | 0.987 |
| unbiased-tiny-genimage/ADM | 375 | 0.957 |
| unbiased-tiny-genimage/BigGAN | 375 | 1.000 |
| unbiased-tiny-genimage/Midjourney | 375 | 0.933 |
| unbiased-tiny-genimage/VQDM | 375 | 0.973 |
| unbiased-tiny-genimage/glide | 375 | 0.995 |
| unbiased-tiny-genimage/stable_diffusion_v_1_5 | 375 | 0.965 |
| unbiased-tiny-genimage/wukong | 375 | 0.973 |

### resnet50

| family | n | accuracy |
|---|---:|---:|
| ai-vs-real-images-dataset/aigen | 38 | 0.816 |
| deepfake-face-images/Fake | 375 | 0.989 |
| real | 1238 | 0.800 |
| real-and-fake-ai-generated-art-images-dataset/aigen | 375 | 0.944 |
| real-vs-fake-faces-stylegan3/Fake faces | 375 | 0.704 |
| sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models | 375 | 0.989 |
| unbiased-tiny-genimage/ADM | 375 | 0.949 |
| unbiased-tiny-genimage/BigGAN | 375 | 1.000 |
| unbiased-tiny-genimage/Midjourney | 375 | 0.888 |
| unbiased-tiny-genimage/VQDM | 375 | 0.923 |
| unbiased-tiny-genimage/glide | 375 | 0.997 |
| unbiased-tiny-genimage/stable_diffusion_v_1_5 | 375 | 0.960 |
| unbiased-tiny-genimage/wukong | 375 | 0.931 |
