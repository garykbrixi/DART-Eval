#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user="kevin.zhu@arcinstitute.org"
#SBATCH --job-name=DART_lls_zeroshot_t2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1                # Total number of tasks (e.g., 4 for 4 GPUs)
#SBATCH --gres=gpu:1              # Number of GPUs per node
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --mem=320G                 # Memory per node
#SBATCH --time=16:00:00           # Maximum job time
#SBATCH --output=logs/%x-%j.out   # Output file (job-name_job-id.out)

# source /opt/conda/etc/profile.d/conda.sh
# conda activate evo2


export DNAGEN_DIR=/home/kevqyzhu/dna-gen
export DART_WORK_DIR=/large_storage/hsulab/kzhu/DART_work_dir
export MODEL_SPECIFIC_NAME=evo2_cascade_1p5_7b_458k

cd $DNAGEN_DIR
# CUDA_VISIBLE_DEVICES=0 python -m eval.DART-Eval.src.dnalm_bench.task_2_5_single.experiments.task_2_transcription_factor_binding.likelihoods.evo
SLURM_JOB_NUM_NODES=1 CUDA_VISIBLE_DEVICES=0 python -m eval.DART-Eval.src.dnalm_bench.task_2_5_single.experiments.task_2_transcription_factor_binding.likelihoods.evo
python -m eval.DART-Eval.src.dnalm_bench.task_2_5_single.experiments.task_2_transcription_factor_binding.footprint_eval_likelihoods --input_seqs $DART_WORK_DIR/task_2_footprinting/processed_data/footprint_dataset_350_v1.txt --likelihoods $DART_WORK_DIR/task_2_footprinting/outputs/likelihoods/$MODEL_SPECIFIC_NAME.tsv --output_file $DART_WORK_DIR/task_2_footprinting/outputs/evals/likelihoods/$MODEL_SPECIFIC_NAME.tsv
