#!/bin/sh
export PYTHONPATH=./
PYTHON=python



dataset_dir=$1
test_area=5

dataset=s3dis
exp_name=PointNL_results


exp_dir=exp/${dataset}/${exp_name}/area${test_area}
model_log=${exp_dir}/log
mkdir -p ${model_log}
cp tool/train.sh tool/train.py ${exp_dir}

now=$(date +"%Y%m%d_%H%M%S")

$PYTHON tool/train.py \
    --train_gpu 0\
    --epochs 100 \
    --save_path=${exp_dir} \
    --train_batch_size 16 \
    --train_workers 16 \
    --minpoints 1024 \
    --save_freq 5 \
    --test_area ${test_area} \
    --train_full_folder ${dataset_dir} \
    --arch=PointNL 2>&1 | tee ${model_log}/train_area${test_area}-$now.log
