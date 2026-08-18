# FORGE phase 2l -- confidence check on StyleGAN3 (shipped model + calibration)

Deployed model: temperature=0.905, uncertain band=[0.38, 0.62]

## StyleGAN3 held-out (2500 images)

- Correctly called fake (confident, right): 784 (31.4%)
- Called uncertain (honest 'not sure'): 262 (10.5%)
- Confidently called real (confident, WRONG): 1454 (58.2%)

## Real photos, test set (750 images), for comparison

- Correctly called real (confident, right): 631 (84.1%)
- Called uncertain: 53 (7.1%)
- Confidently called fake (confident, WRONG): 66 (8.8%)
