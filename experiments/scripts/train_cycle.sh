

GPU_ID=$1
DATASET=$2
SPLITBY=$3
OUTPUT_POSTFIX=$4
CAPTION_MODEL=$5
CAPTION_LOSS_WEIGHT=$6

IMDB="coco_minus_refer"
ITERS=1250000
TAG="notime"
NET="res101"
ID="mrcn_cmr_with_st"

STEPSIZE="[360000]" ####
ANCHORS="[4,8,16,32]"
RATIOS="[0.5,1,2]"

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/train_cycle.py \
    --imdb_name ${IMDB} \
    --net_name ${NET} \
    --iters ${ITERS} \
    --tag ${TAG} \
    --dataset ${DATASET} \
    --splitBy ${SPLITBY} \
    --output_postfix ${OUTPUT_POSTFIX} \
    --caption_model ${CAPTION_MODEL} \
    --start_from caption_log_res5_2_240k \
    --cap_loss_weight ${CAPTION_LOSS_WEIGHT} \
    --max_iters 800000 \
    --with_st 1 \
    --id ${ID} \
    --cfg experiments/cfgs/${NET}.yml \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
    TRAIN.STEPSIZE ${STEPSIZE} # ${EXTRA_ARGS}
