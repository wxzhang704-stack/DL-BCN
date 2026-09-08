# DL-BCN core U-Net

This repository contains the core shallow U-Net architecture used in the
Deep Learning-Based Boundary-Constrained Multi-Domain Nesting (DL-BCN)
framework for high-resolution air-quality refinement.

## Contents

- `DL_BCN_UNet_core.py`: two-level U-Net and the real/pseudo group-mean MSE
  training loss used in the reported experiments.

## Model input and output

The model accepts a square multichannel predictor field and returns one
continuous refined concentration field. The spatial size and number of input
channels can be specified when calling `build_unet`.

```python
from DL_BCN_UNet_core import build_unet, group_mean_mse

model = build_unet(input_size=128, input_channels=8)
model.compile(optimizer="adam", loss=group_mean_mse(0.25))
```

For the supplied loss, target channel 0 contains the concentration target,
channel 1 the real-station training mask, and channel 2 the pseudo-station
training weight. A pseudo relative weight of 0.25 corresponds to normalized
real and pseudo group weights of 0.8 and 0.2, respectively.

## Environment used for verification

- Python 3.10
- TensorFlow 2.18.1
- Keras 3.6.0

