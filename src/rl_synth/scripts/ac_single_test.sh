#!/bin/bash
# generated at Sun Dec 12 14:05:46 PST 2021
# Job name:
#SBATCH --job-name=batch1000_lr0.05_eb50_NN32x2_ntu10_ngstu10_discount1.0
#
# Account:
#SBATCH --account=fc_bdmesh
#
# Partition:
#SBATCH --partition=savio3
#
# Quality of Service:
#SBATCH --qos=savio_normal
# Num Cores per Task
#SBATCH --cpus-per-task=8
#
# Wall clock limit:
#SBATCH --time=10:00:00
#
## Command(s) to run:
source /global/home/users/mkang/.bashrc
source activate rl
python ../rl_synth/scripts/run_ac.py --exp_name  batch1000_lr0.05_eb50_NN32x2_ntu1_ngstu1_discount1.0 --env_name synthesis-v0 --batch_size 1000 --n_iter 5 --n_layers 2 --size 32 --scalar_log_freq 1 --video_log_freq -1 --num_target_updates 01 --num_grad_steps_per_target_update 10 --eval_batch_size 50
