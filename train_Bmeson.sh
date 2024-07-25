#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_Bmeson`
DATADIR=${DATADIR_Bmeson}
[[ -z $DATADIR ]] && DATADIR='./datasets/Bmeson'

# set a comment via `COMMENT`
suffix=${COMMENT}

# PN, PFN, PCNN, ParT
model=$1
extraopts=""
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParticleTransformer.py --use-amp --optimizer-option weight_decay 0.01"
    lr="1e-3"
elif [[ "$model" == "ParT-FineTune" ]]; then
    modelopts="networks/example_ParticleTransformer_finetune.py --use-amp --optimizer-option weight_decay 0.01"
    lr="1e-4"
    extraopts="--optimizer-option lr_mult (\"fc.*\",50) --lr-scheduler none"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    lr="1e-2"
elif [[ "$model" == "PN-FineTune" ]]; then
    modelopts="networks/example_ParticleNet_finetune.py"
    lr="1e-3"
    extraopts="--optimizer-option lr_mult (\"fc_out.*\",50) --lr-scheduler none"
elif [[ "$model" == "PFN" ]]; then
    modelopts="networks/example_PFN.py"
    lr="2e-2"
    extraopts="--batch-size 4096"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="networks/example_PCNN.py"
    lr="2e-2"
    extraopts="--batch-size 4096"
else
    echo "Invalid model $model!"
    exit 1
fi

# "kin", "kinpid", "kinpidplus"
FEATURE_TYPE=$2
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="kinpid"

if [[ "${FEATURE_TYPE}" == "kin" ]]; then
    pretrain_type="kin"
elif [[ "${FEATURE_TYPE}" =~ ^(kinpid|kinpidplus)$ ]]; then
    pretrain_type="kinpid"
else
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

if [[ "$model" == "ParT-FineTune" ]]; then
    modelopts+=" --load-model-weights models/ParT_${pretrain_type}.pt"
fi
if [[ "$model" == "PN-FineTune" ]]; then
    modelopts+=" --load-model-weights models/ParticleNet_${pretrain_type}.pt"
fi

weaver \
    --data-train "${DATADIR}/train_file.parquet" \
    --data-test "${DATADIR}/test_file.parquet" \
    --data-config data/Bmeson/bm_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/Bmeson/${model}/{auto}${suffix}/net \
    --num-workers 1 --fetch-step 1 --in-memory --train-val-split 0.8889 \
    --batch-size 128 --samples-per-epoch 1600000 --samples-per-epoch-val 200000 --num-epochs 1 --gpus 0 \
    --start-lr $lr --optimizer ranger --log logs/Bmeson_${model}_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard Bmeson_${FEATURE_TYPE}_${model}${suffix} \
    --regression-mode \
    # open the regression-mode
    ${extraopts} "${@:3}"