# FORGE phase 2k -- ensembling / weight-averaging check

Baseline (single checkpoints, from the stability check): StyleGAN3 ranged 0.005-0.305, SFHQ-T2I ranged 0.280-0.709.

| method | window | StyleGAN3 | SFHQ-T2I | real | in-dist test |
|---|---|---:|---:|---:|---:|
| prediction-average | all 8 epochs | 0.019 | 0.555 | 0.940 | - |
| weight-average (SWA-style) | all 8 epochs | 0.041 | 0.476 | 0.944 | - |
| prediction-average | last 4 epochs | 0.021 | 0.560 | 0.932 | - |
| weight-average (SWA-style) | last 4 epochs | 0.037 | 0.518 | 0.940 | - |
| prediction-average | last 6 epochs | 0.019 | 0.565 | 0.937 | - |
| weight-average (SWA-style) | last 6 epochs | 0.030 | 0.508 | 0.943 | - |
| prediction-average | epochs 2+4 (the two highest individually) | 0.154 | 0.684 | 0.932 | - |
| weight-average (SWA-style) | epochs 2+4 (the two highest individually) | 0.129 | 0.552 | 0.936 | - |
