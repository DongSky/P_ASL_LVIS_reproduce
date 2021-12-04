#!/bin/bash
TASK=prior_lvis
#logfile=$(date +%Y%m%d)_$(date +%H%M%S)_$TASK.log
logfile=feedback_to_author.log
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 34567 --nproc_per_node=2 main_dist_lvis.py \
--data=/data/lyz/datasets/coco/ --path_dest=./outputs_lvis_tresnet_m_feedback \
--world-size 1 --rank 0 --dist-url=env:// \
--model-name=tresnet_m --model-path=tresnet_m_miil_21k.pth \
--gamma_pos=0 \
--gamma_neg=0 \
--gamma_unann=1 \
--lr=6e-4 \
--partial_loss_mode=selective \
--likelihood_topk=5 \
--prior_threshold=0.5 \
2>&1 |tee logs/$logfile
#--prior_path=./outputs/priors/prior_fpc_1000.csv
