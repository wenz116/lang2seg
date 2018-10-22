# lang2seg

## Training
1. Train the baseline segmentation model:
```
./experiments/scripts/train_baseline.sh GPUID refcoco unc
```
The Mask-RCNN model is in `pyutils/mask-faster-rcnn/lib/nets/resnet_v1.py` and `pyutils/mask-faster-rcnn/lib/nets/network.py`.

2. Train the whole model with caption loss:
```
./experiments/scripts/train_cycle.sh GPUID refcoco unc att2in2 CAPTION_LOSS_WEIGHT
```
The whole model is in `pyutils/mask-faster-rcnn/lib/nets/resnet_v1_cycle.py` and `pyutils/mask-faster-rcnn/lib/nets/network_cycle.py`.

Losses are calculated in [_add_losses()](https://github.com/wenz116/lang2seg/blob/master/pyutils/mask-faster-rcnn/lib/nets/network_cycle.py#L376).

## Evaluation
```
./experiments/scripts/eval_baseline.sh GPUID refcoco unc
```
