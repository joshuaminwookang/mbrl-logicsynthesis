launch_mbrl_train_job(){
    BATCH_INITIAL=$1
    AGENT_TRAIN_STEPS=$2
    TRAINING_BATCH=$3
    NN_SIZE=$4
    NN_LAYERS=$5
    LEARNING_RATE=$6
    ENSEMBLE=$7
    HORIZON=$8
    slurm_script_name="mbrl_train_batchinit$1_natspi$2_tb$3_model$4x$5_lr$6_ensemble$7_horizon$8"
    echo $slurm_script_name
    cat > "${slurm_script_name}.sh" <<EOT
#!/bin/bash
# generated at $(date)
# Job name:
#SBATCH --job-name=${slurm_script_name}
#
# Account:
#SBATCH --account=fc_bdmesh
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# Quality of Service:
#SBATCH --qos=savio_normal
# Num Cores per Task
#SBATCH --cpus-per-task=8
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=10:00:00
#
## Command(s) to run:
source /global/home/users/${USER}/.bashrc
source activate rl
python ../rl_synth/scripts/run.py --exp_name  ${slurm_script_name} --env_name synthesis-v0 \
        --add_sl_noise --n_iter 2 --mpc_horizon ${HORIZON} --ensemble_size ${ENSEMBLE} \
        --eval_batch_size 50 --save_params \
        --batch_size_initial 1000 --num_agent_train_steps_per_iter ${AGENT_TRAIN_STEPS}\
        --train_batch_size ${TRAINING_BATCH} --batch_size 500 \
        --n_layers ${NN_LAYERS} --size ${NN_SIZE} --scalar_log_freq 1 --video_log_freq -1 \
        --mpc_action_sampling_strategy 'random'  --learning_rate ${LEARNING_RATE} 
EOT
    sbatch "${slurm_script_name}.sh"
}

launch_mbrl_run_job(){
    BATCH=$1
    AGENT_TRAIN_STEPS=$2
    TRAINING_BATCH=$3
    NN_SIZE=$4
    NN_LAYERS=$5
    LEARNING_RATE=$6
    ENSEMBLE=$7
    HORIZON=$8
    SEED=$9
    slurm_script_name="mbrl_run_tb$3_model$4x$5_ensemble$7_horizon$8_seed$9"
    echo $slurm_script_name
    cat > "${slurm_script_name}.sh" <<EOT
#!/bin/bash
# generated at $(date)
# Job name:
#SBATCH --job-name=${slurm_script_name}
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
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=20:00:00
#
## Command(s) to run:
source /global/home/users/${USER}/.bashrc
source activate rl
python ../rl_synth/scripts/run.py --exp_name  ${slurm_script_name} --env_name synthesis-v0 \
        --add_sl_noise --n_iter 5 --mpc_horizon ${HORIZON} --ensemble_size ${ENSEMBLE} \
        --eval_batch_size 50 --save_params \
        --batch_size_initial 1000 --num_agent_train_steps_per_iter ${AGENT_TRAIN_STEPS}\
        --train_batch_size ${TRAINING_BATCH} --batch_size ${BATCH} \
        --n_layers ${NN_LAYERS} --size ${NN_SIZE} --scalar_log_freq 1 --video_log_freq -1 \
        --mpc_action_sampling_strategy 'random'  --learning_rate ${LEARNING_RATE} --seed ${SEED}
EOT
    sbatch "${slurm_script_name}.sh"
}

if ! [ -d "runs" ]; then
  mkdir -p "runs"
fi
pushd runs

# declare -a BATCH_SIZE_ARR=("1000" "2000" )
# declare -a TB_ARR=("200" "500")
# declare -a NATSPI_ARR=("500" "1000")
# declare -a NN_SIZE_ARR=("128" "256")
# declare -a NN_DEPTH=("2" "3")
# declare -a HORIZON_ARR=("3" "5")
# declare -a ENSEMBLE_ARR=("3" "5")
# # declare -a LR_ARR=("0.1" "0.01")

# for tb in "${TB_ARR[@]}" 
# do 
#     for natspi in "${NATSPI_ARR[@]}" 
#     do
#         for nnsize in "${NN_SIZE_ARR[@]}"
#         do 
#             for nnlayers in "${NN_DEPTH[@]}"
#             do
#                 for horizon in "${HORIZON_ARR[@]}"
#                 do
#                     for ensemble in "${ENSEMBLE_ARR[@]}"
#                     do
#                         launch_mbrl_train_job 1000 $natspi $tb $nnsize $nnlayers 0.01 $ensemble $horizon
#                     done
#                 done
#             done
#         done
#     done
# done

declare -a TB_ARR=("200" "500")
# declare -a NN_SIZE_ARR=("128" "256")
declare -a NN_DEPTH=("2" "3")
# declare -a NATSPI_ARR=("500" "1000")
declare -a HORIZON_ARR=("3" "5")
# declare -a ENSEMBLE_ARR=("3" "5")
declare -a SEED_ARR=("1" "2" "3")

for tb in "${TB_ARR[@]}" 
do 
    for nnlayers in "${NN_DEPTH[@]}"
    do
        for horizon in "${HORIZON_ARR[@]}"
        do
            for seed in "${SEED_ARR[@]}"
            do
                launch_mbrl_run_job 1000 500 $tb 128 $nnlayers 0.01 5 $horizon $seed
            done
        done
    done
done
