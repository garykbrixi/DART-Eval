#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user="kevin.zhu@arcinstitute.org"
#SBATCH --job-name=DART_zeroshot_embeddings_t5
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
export MODEL=evo

cd $DNAGEN_DIR
# CUDA_VISIBLE_DEVICES=0 python -m eval.DART-Eval.src.dnalm_bench.task_2_5_single.experiments.task_2_transcription_factor_binding.likelihoods.evo
SLURM_JOB_NUM_NODES=1 CUDA_VISIBLE_DEVICES=0 python -m eval.DART-Eval.src.dnalm_bench.task_2_5_single.experiments.task_5_variant_effect_prediction.zero_shot_embeddings.$MODEL $DART_WORK_DIR/task_5_variant_effect_prediction/input_data/Afr.CaQTLS.tsv Afr.CaQTLS $DART_WORK_DIR/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta
SLURM_JOB_NUM_NODES=1 CUDA_VISIBLE_DEVICES=0 python -m eval.DART-Eval.src.dnalm_bench.task_2_5_single.experiments.task_5_variant_effect_prediction.zero_shot_embeddings.$MODEL $DART_WORK_DIR/task_5_variant_effect_prediction/input_data/yoruban.dsqtls.benchmarking.tsv yoruban.dsqtls.benchmarking $DART_WORK_DIR/refs/male.hg19.fa

