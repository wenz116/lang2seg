# lang2seg

## Prerequisites
* Python 2.7
* Pytorch 0.2 or 0.3
* CUDA 8.0
* Mask R-CNN: Follow the instructions of the [mask-faster-rcnn](https://github.com/lichengunc/mask-faster-rcnn) repo, preparing everything needed for `pyutils/mask-faster-rcnn`.
* REFER API and data: Use the download links of [REFER](https://github.com/lichengunc/refer) and go to the foloder running `make`. Follow `data/README.md` to prepare images and refcoco/refcoco+/refcocog annotations.
* COCO training set should be downloaded in `pyutils/mask-faster-rcnn/data/coco/images/train2014`.

## Preprocessing
The processed data is uploaded in `cache/prepro/`.

## Training

* `<DATASET> <SPLITBY>` pairs contain: refcoco unc/refcoco+ unc/refcocog umd/refcocog google

* Output model will be saved at `<DATASET>_<SPLITBY>/output_<OUTPUT_POSTFIX>`. If there are trained models in this directory, the model of the latest iteratioin will be loaded.

* The iteration when learning rate decay is specified as `STEPSIZE` in `train_*.sh`.

1. Train the baseline segmentation model with only 1 dynamic filter:
```
./experiments/scripts/train_baseline.sh <GPUID> <DATASET> <SPLITBY> <OUTPUT_POSTFIX>
```
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
The pretrained Mask R-CNN model should be placed at `<DATASET>_<SPLITBY>/output_<OUTPUT_POSTFIX>`. If there are multiple models in the directory, the model of the latest iteration will be loaded.

The pretrained caption model should be placed at `<DATASET>_<SPLITBY>/caption_log_res5_2/`, named as `model-best.pth` and `infos-best.pkl`.

The Mask R-CNN model is in `pyutils/mask-faster-rcnn/lib/nets/resnet_v1_cycle_res5_2.py` and `pyutils/mask-faster-rcnn/lib/nets/network_cycle_res5_2.py`.

4. Train the model with spatial dynamic filters and response loss:
```
./experiments/scripts/train_response.sh <GPUID> <DATASET> <SPLITBY> <OUTPUT_POSTFIX>
```
The Mask R-CNN model is in `pyutils/mask-faster-rcnn/lib/nets/resnet_v1_7f_response.py` and `pyutils/mask-faster-rcnn/lib/nets/network_7f_response.py`.

5. Train the model with spatial dynamic filters, response loss and caption loss:
```
./experiments/scripts/train_cycle_response.sh <GPUID> <DATASET> <SPLITBY> <OUTPUT_POSTFIX> att2in2 <CAPTION_LOSS_WEIGHT>
```
The pretrained Mask R-CNN model should be placed at `<DATASET>_<SPLITBY>/output_<OUTPUT_POSTFIX>`. If there are multiple models in the directory, the model of the latest iteration will be loaded.

The pretrained caption model should be placed at `<DATASET>_<SPLITBY>/caption_log_response/`, named as `model-best.pth` and `infos-best.pkl`.

The Mask R-CNN model is in `pyutils/mask-faster-rcnn/lib/nets/resnet_v1_cycle_response.py` and `pyutils/mask-faster-rcnn/lib/nets/network_cycle_response.py`.

6. Train the model with spatial dynamic filters and response loss for VGG16 and Faster R-CNN:

Download the pre-trained Faster R-CNN model [here](https://drive.google.com/drive/folders/0B7fNdx_jAqhtU1laZEhlM09fazA) (coco_900k-1190k.tar), and put the .pth and .pkl files in `pyutils/mask-faster-rcnn/output/vgg16/coco_2014_train+coco_2014_valminusminival/`
```
./experiments/scripts/train_vgg.sh <GPUID> <DATASET> <SPLITBY> <OUTPUT_POSTFIX>
```
The Faster R-CNN model is in `pyutils/mask-faster-rcnn/lib/nets/vgg16.py` and `pyutils/mask-faster-rcnn/lib/nets/network_vgg.py`.

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

3. Evaluate the model with spatial dynamic filters and response loss (and caption loss):
```
./experiments/scripts/eval_response.sh <GPUID> <DATASET> <SPLITBY> <OUTPUT_POSTFIX> <MODEL_ITER>
```

4. Evaluate the model with spatial dynamic filters and response loss for VGG16 and Faster R-CNN:
```
./experiments/scripts/eval_vgg.sh <GPUID> <DATASET> <SPLITBY> <OUTPUT_POSTFIX> <MODEL_ITER>
```
