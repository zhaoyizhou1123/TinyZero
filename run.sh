#!/bin/bash
#SBATCH --job-name=countdown-qwen2.5-3b
#SBATCH -o ./slurm/%x/job_%A.out # STDOUT
# SBATCH -p HGPU
#SBATCH --gres=gpu:H100:4    # Request N GPUs per machine
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task 2
#SBATCH --chdir /home/zhaoyiz/projects/reasoning/TinyZero

export N_GPUS=4
# export CUDA_VISIBLE_DEVICES=2,3
local_dir=.local_reason
echo "Local directory: $local_dir"
apptainer exec --nv -B $HOME/$local_dir:$HOME/.local --env-file  $HOME/containers/env.txt $HOME/containers/pytorch_23.11-py3.sif bash -c "
ray stop --force && ray start --head --include-dashboard=True
export BASE_MODEL='Qwen/Qwen2.5-3B'
export DATA_DIR='/home/zhaoyiz/datasets/countdown'
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS
export TRITON_LIBCUDA_PATH="/usr/lib64"
bash ./scripts/train_tiny_zero_h100.sh
"