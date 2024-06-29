#!/bin/bash
#
#PBS -N t1
#PBS -l select=1:ncpus=40:mem=125gb:ngpus=1:gpu_model=a100
#PBS -l walltime=36:00:00
#PBS -o out1.txt
#PBS -j oe

module load anaconda3/2022.05-gcc/9.5.0 cuda/11.1.1-gcc/9.5.0 
cd ../../scratch1/xiwenc/SwimUnet/
cudnn/8.0.5.39-11.1-gcc/9.5.0-cu11_1

source activate milenv



python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --max_epochs 150 --output_dir 11_1  --gpu_id 0 --img_size 224 --base_lr 0.05 --batch_size 32 --seed 1 --lambda_x 0.010 


python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --output_dir 11_1 --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
