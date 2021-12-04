#!/bin/bash
TASK=prior_lvis
logfile=$(date +%Y%m%d)_$(date +%H%M%S)_$TASK.log
CUDA_VISIBLE_DEVICES=6 python -u train_lvis.py \
--data=/data/lyz/datasets/coco/ --path_dest=./outputs_lvis_tresnetm_singlegpu \
--model-name=tresnet_m --model-path=tresnet_m_miil_21k.pth \
--gamma_pos=0 \
--gamma_neg=0 \
--gamma_unann=1 \
--lr=6e-4 \
--partial_loss_mode=selective \
--likelihood_topk=10 \
--prior_threshold=0.5 \
2>&1 |tee logs/$logfile
#--prior_path=./outputs/priors/prior_fpc_1000.csv
