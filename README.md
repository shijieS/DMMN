# An End-To-End Detector Matcher
This is an end-to-end network which combines detector and matcher into one single network. Our target is to design an **End-to-End network** for detection and tracking.

## Protocol
- bbox: the format is *(left, top, right, bottom)*
- $N_{tr}$: the track number.
- $N_{ba}$: the batch number.
- $N_{ti}$: the selected frame number
- $W, H$: the input network image size (W, H).
- $W_{re}, H_{re}$: the real input image size.
- $F_t$: the $t^{th}$ frame
- $N_{fn}$: the input frame number

## Network

The framework of our net is as following:

![framework](./images/framework.png)














## Framework


## Requirement
|name           |version
|:---:          |:---:
|python         |3.6
|cuda           |8.0

Besides, install all the python package by following command

```shell
cd <project path>
pip install -r requiement.txt
```

## Preparation
- Download this code
- Download the [pre-trained base net model](https://drive.google.com/open?id=1CYb-RBZpz3UTbQRM4oIRipZrWrq10iIQ)

## Train

## Test

## Citation

## Copyright
