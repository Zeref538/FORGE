# FORGE -- final shipping model (trained on all generator families)

    device=cpu, epochs=5, train=21000 (real=3500 fake=17500), val=4500, test=4500, fake families trained on=10
    epoch 1/5  train_acc=0.804  val_acc=0.895  (567s)
    epoch 2/5  train_acc=0.869  val_acc=0.878  (556s)
    epoch 3/5  train_acc=0.892  val_acc=0.901  (561s)
    epoch 4/5  train_acc=0.911  val_acc=0.894  (561s)
    epoch 5/5  train_acc=0.921  val_acc=0.923  (561s)

## Test accuracy per family (test slices never trained on)

| family | n | accuracy |
|---|---:|---:|
| deepfake-face-images/Fake | 375 | 0.979 |
| real | 750 | 0.865 |
| real-vs-fake-faces-stylegan3/Fake faces | 375 | 0.709 |
| sfhq-t2i-synthetic-faces-from-text-2-image-models/sfhq-t2i-synthetic-faces-from-text-2-image-models | 375 | 0.989 |
| unbiased-tiny-genimage/ADM | 375 | 0.928 |
| unbiased-tiny-genimage/BigGAN | 375 | 1.000 |
| unbiased-tiny-genimage/Midjourney | 375 | 0.920 |
| unbiased-tiny-genimage/VQDM | 375 | 0.968 |
| unbiased-tiny-genimage/glide | 375 | 0.989 |
| unbiased-tiny-genimage/stable_diffusion_v_1_5 | 375 | 0.949 |
| unbiased-tiny-genimage/wukong | 375 | 0.939 |

**Overall test accuracy: 0.925**

Note: unlike earlier phase-2 runs, no generator family was withheld from training. The leave-one-generator-out numbers reported separately still stand as the honest answer to 'what about a generator released after this was trained'.
