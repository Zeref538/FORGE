# FORGE phase 2j -- epoch-by-epoch stability check

    device=cpu, epochs=8, train=17500 (real=3500 fake=14000)
    epoch 1/8  train_acc=0.855 val_acc=0.894  StyleGAN3=0.005 SFHQ-T2I=0.280 real=0.951  (477s)
    epoch 2/8  train_acc=0.922 val_acc=0.932  StyleGAN3=0.305 SFHQ-T2I=0.709 real=0.885  (461s)
    epoch 3/8  train_acc=0.937 val_acc=0.934  StyleGAN3=0.013 SFHQ-T2I=0.480 real=0.951  (455s)
    epoch 4/8  train_acc=0.948 val_acc=0.944  StyleGAN3=0.074 SFHQ-T2I=0.656 real=0.935  (459s)
    epoch 5/8  train_acc=0.956 val_acc=0.951  StyleGAN3=0.016 SFHQ-T2I=0.517 real=0.927  (454s)
    epoch 6/8  train_acc=0.961 val_acc=0.946  StyleGAN3=0.063 SFHQ-T2I=0.626 real=0.936  (451s)
    epoch 7/8  train_acc=0.966 val_acc=0.951  StyleGAN3=0.024 SFHQ-T2I=0.442 real=0.923  (456s)
    epoch 8/8  train_acc=0.971 val_acc=0.957  StyleGAN3=0.034 SFHQ-T2I=0.680 real=0.912  (453s)

## Held-out accuracy by epoch (not just the best-val checkpoint)

| epoch | val_acc | StyleGAN3 | SFHQ-T2I | real |
|---:|---:|---:|---:|---:|
| 1 | 0.894 | 0.005 | 0.280 | 0.951 |
| 2 | 0.932 | 0.305 | 0.709 | 0.885 |
| 3 | 0.934 | 0.013 | 0.480 | 0.951 |
| 4 | 0.944 | 0.074 | 0.656 | 0.935 |
| 5 | 0.951 | 0.016 | 0.517 | 0.927 |
| 6 | 0.946 | 0.063 | 0.626 | 0.936 |
| 7 | 0.951 | 0.024 | 0.442 | 0.923 |
| 8 | 0.957 | 0.034 | 0.680 | 0.912 |

**StyleGAN3 range across epochs: 0.005 - 0.305**
**StyleGAN3 at the epoch validation would have picked (highest val_acc): 0.034**
