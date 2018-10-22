# lang2seg
The Mask-RCNN model is in `pyutils/mask-faster-rcnn/lib/nets/resnet_v1.py` and `pyutils/mask-faster-rcnn/lib/nets/network.py`.

Losses are calculated in [_add_losses()](https://github.com/wenz116/lang2seg/blob/master/pyutils/mask-faster-rcnn/lib/nets/network.py#L373).

## Training
Train the baseline segmentation model:
```
./experiments/scripts/train_baseline.sh GPUID refcoco unc
```
Train the whole network with caption loss:
```
./experiments/scripts/train_cycle.sh GPUID refcoco unc att2in2 CAPTION_LOSS_WEIGHT
```
## Evaluation
```
./experiments/scripts/eval_baseline.sh GPUID refcoco unc
```
