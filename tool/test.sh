#!/bin/sh
export PYTHONPATH=./

PYTHON=python
dataset=s3dis
model=PointNL
test_area=5

dataset_dir=$1

exp_name=test/
model_path=./pretrained_model/PointNL_model.pth

exp_dir=exp/${dataset}/${exp_name}/area${test_area}
model_log=${exp_dir}/log
save_folder=${exp_dir}/

mkdir -p ${model_log}
mkdir -p ${save_folder}
now=$(date +"%Y%m%d_%H%M%S")


cp tool/test.sh tool/test.py ${exp_dir}
$PYTHON tool/test.py 2>&1 --model_path ${model_path} \
    --train_gpu 1 \
    --save_folder ${save_folder} \
    --arch ${model} \
    --test_area ${test_area} \
    --stride_rate 0.5 \
    --train_full_folder  ${dataset_dir} | tee ${model_log}/test_area${test_area}_${epoch}-$now.log
