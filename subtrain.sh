#!/usr/bin/env sh

# =====================================================
# Description: subtrain.sh
#
# =====================================================
#
# Created by YongBai on 2020/8/19 11:08 AM.

echo "python train.py">run_train_16_1.sh
qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e16-1 -l vf=40g,p=1 run_train_16_1.sh