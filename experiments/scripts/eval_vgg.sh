

GPU_ID=$1
DATASET=$2
SPLITBY=$3
OUTPUT_POSTFIX=$4
MODEL_ITER=$5

# IMDB="coco_minus_refer"
# ITERS=1150000
# TAG="notime"
NET="vgg16"
ID="mrcn_cmr_with_st"
# ID="mrcn_dets_cmr_with_st"

ANCHORS="[4,8,16,32]"
RATIOS="[0.5,1,2]"

case ${DATASET} in
    refcoco)
        for SPLIT in val testA testB
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_vgg.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --output_postfix ${OUTPUT_POSTFIX} \
                --model_iter ${MODEL_ITER} \
                --split ${SPLIT} \
                --id ${ID} \
                --cfg experiments/cfgs/${NET}.yml \
                --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS}
        done
    ;;
    refcoco+)
        for SPLIT in val testA testB
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_vgg.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --output_postfix ${OUTPUT_POSTFIX} \
                --model_iter ${MODEL_ITER} \
                --split ${SPLIT} \
                --id ${ID} \
                --cfg experiments/cfgs/${NET}.yml \
                --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS}
        done
    ;;
    refcocog)
        for SPLIT in val test
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_vgg.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --output_postfix ${OUTPUT_POSTFIX} \
                --model_iter ${MODEL_ITER} \
                --split ${SPLIT} \
                --id ${ID} \
                --cfg experiments/cfgs/${NET}.yml \
                --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS}
        done
    ;;
esac