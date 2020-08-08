#!/usr/bin/env bash
# Created by Chun-Ming Su
SLACK_TOKEN="YOUR_SLACK_TOKEN"
SLACK_ID="YOUR_SLACK_ID"
SEND_TO_SLACK=true
DELETE_OLD_CKPT=false
SEED=0
DETERMINISTIC="--deterministic"
WORK_DIR="runs"

# Internal Field Separator
IFS="|"
ARG_COUNT=4
CONFIGS=(
# > "dataset    | model | loss | epochs "
# e.g.
    "TINY_IMAGENET | mobilenetv2 | ce | 50  |"
)


for CONFIG in "${CONFIGS[@]}" ;
do
    # > remove white spaces and split configs
    CFG="${CONFIG// /}"
    IFS_COUNT=$(tr -dc '|' <<< ${CFG} | wc -c)
    echo "IFS_COUNT: ${IFS_COUNT}"
    if [[ "${IFS_COUNT}" -ne "${ARG_COUNT}" ]]; then
        echo "> Invalid arguments = ${CFG}"
        continue
    fi

    # > parse configs
    set -- "$CFG"
    declare -a CFG=($*)
    DATASET=${CFG[0]}
    MODEL=${CFG[1]}
    LOSS=${CFG[2]}
    EPOCHS=${CFG[3]}
    echo "> DATASET= ${DATASET}"
    echo "> MODEL= ${MODEL}"
    echo "> LOSS = ${LOSS}"
    echo "> EPOCHS = ${EPOCHS}"
    echo ""

    # > run training script
    sleep 5
    echo "> Start training ..."
    CUDA_VISIBLE_DEVICES='0,1'
    python main.py --dataset ${DATASET} --model-name ${MODEL} --loss ${LOSS} --epochs ${EPOCHS}
#    python -m torch.distributed.launch --nproc_per_node=1 --nnode=2 --node_rank=0 --master_port=29500 \
#    python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 \
#    main.py --dataset ${DATASET} --model-name ${MODEL} --loss ${LOSS} --epochs ${EPOCHS}

    sleep 5
    if ${SEND_TO_SLACK} ; then
        slack-cli -t ${SLACK_TOKEN} -d ${SLACK_ID} "> ${MODEL}-${LOSS}-${EPOCHS} training is DONE"
        import -window root -delay 1000 screenshot.png
        sleep 5
        slack-cli -t ${SLACK_TOKEN} -d ${SLACK_ID} -f screenshot.png
    fi
    echo ""
    sleep 5

    echo ""
    echo "=============================== [end of training/testing] ======================================"
    echo ""
done
