# FORGE phase 2b — calibration report

Temperature: 0.905
ECE (test, before calibration): 0.0089
ECE (test, after calibration): 0.0034

Uncertain band: calibrated p(generated) in [0.38, 0.62]
Coverage on test set: 4.3% of images fall in the uncertain band
Accuracy outside the band: 0.950

## ONNX export parity

max |torch - onnx| on 16-image batch: 0.000069
Parity check: PASS
