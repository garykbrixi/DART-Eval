#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user="kevin.zhu@arcinstitute.org"
#SBATCH --job-name=DART_lls_zeroshot_t1
#SBATCH --partition=gpu_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1                # Total number of tasks (e.g., 4 for 4 GPUs)
#SBATCH --gres=gpu:1              # Number of GPUs per node
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --mem=320G                 # Memory per node
#SBATCH --time=32:00:00           # Maximum job time
#SBATCH --output=logs/%x-%j.out   # Output file (job-name_job-id.out)

# source /opt/conda/etc/profile.d/conda.sh
# conda activate evo2

export DNAGEN_DIR=/home/kevqyzhu/dna-gen
export DART_WORK_DIR=/large_storage/hsulab/kzhu/DART_work_dir

cd $DNAGEN_DIR
SLURM_JOB_NUM_NODES=1 CUDA_VISIBLE_DEVICES=0 python -m eval.DART-Eval.src.dnalm_bench.task_1_paired_control.zero_shot.encode_ccre.evo
