# lang2seg

## Prerequisites

* Python 2.7
* Pytorch 0.2 or 0.3
* CUDA 8.0
* Mask R-CNN: Follow the instructions of the [mask-faster-rcnn](https://github.com/lichengunc/mask-faster-rcnn) repo, preparing everything needed for `pyutils/mask-faster-rcnn`.
* REFER API and data: Use the download links of [REFER](https://github.com/lichengunc/refer) and go to the foloder running `make`. Follow `data/README.md` to prepare images and refcoco/refcoco+/refcocog annotations.
* COCO training set should be downloaded in `pyutils/mask-faster-rcnn/data/coco/images/train2014`.

## Preprocessing
```
python tools/prepro.py --dataset <DATASET> --splitBy <SPLITBY>
```
`<DATASET> <SPLITBY>` pairs contain: refcoco unc/refcoco+ unc/refcocog umd/refcocog google

## Training
1. Train the baseline segmentation model with only 1 dynamic filter:
```
./experiments/scripts/train_baseline.sh <GPUID> <DATASET> <SPLITBY> <OUTPUT_POSTFIX>
```
Output model will be saved at `<DATASET>_<SPLITBY>/output_<OUTPUT_POSTFIX>`.

The Mask R-CNN model is in `pyutils/mask-faster-rcnn/lib/nets/resnet_v1.py` and `pyutils/mask-faster-rcnn/lib/nets/network.py`.

2. Train the model with spatial dynamic filters:
```
./experiments/scripts/train_spatial.sh <GPUID> <DATASET> <SPLITBY> <OUTPUT_POSTFIX>
```
The Mask R-CNN model is in `pyutils/mask-faster-rcnn/lib/nets/resnet_v1_7f.py` and `pyutils/mask-faster-rcnn/lib/nets/network_7f.py`.

3. Train the model with spatial dynamic filters and caption loss:
```
./experiments/scripts/train_cycle.sh <GPUID> <DATASET> <SPLITBY> <OUTPUT_POSTFIX> att2in2 <CAPTION_LOSS_WEIGHT>
```
The Mask R-CNN model is in `pyutils/mask-faster-rcnn/lib/nets/resnet_v1_cycle_res5_2.py` and `pyutils/mask-faster-rcnn/lib/nets/network_cycle_res5_2.py`.

4. Train the model with spatial dynamic filters and response loss:
```
./experiments/scripts/train_response.sh <GPUID> <DATASET> <SPLITBY> <OUTPUT_POSTFIX>
```
The Mask R-CNN model is in `pyutils/mask-faster-rcnn/lib/nets/resnet_v1_7f_response.py` and `pyutils/mask-faster-rcnn/lib/nets/network_7f_response.py`.


## Evaluation
1. Evaluate the baseline segmentation model:
```
./experiments/scripts/eval_baseline.sh <GPUID> <DATASET> <SPLITBY> <OUTPUT_POSTFIX> <MODEL_ITER>
```
Evaluate the model at `<DATASET>_<SPLITBY>/output_<OUTPUT_POSTFIX>`, of trained iteration `<MODEL_ITER>`.

Detection and segmentation results will be saved at `experiments/det_results.txt` and `experiments/mask_results.txt` respectively.

2. Evaluate the model with spatial dynamic filters (and caption loss):
```
./experiments/scripts/eval_spatial.sh <GPUID> <DATASET> <SPLITBY> <OUTPUT_POSTFIX> <MODEL_ITER>
```

3. Evaluate the model with spatial dynamic filters and response loss:
```
./experiments/scripts/eval_response.sh <GPUID> <DATASET> <SPLITBY> <OUTPUT_POSTFIX> <MODEL_ITER>
```
