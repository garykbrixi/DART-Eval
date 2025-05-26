#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user="kevin.zhu@arcinstitute.org"
#SBATCH --job-name=DART_embedding_extraction_t3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1                # Total number of tasks (e.g., 4 for 4 GPUs)
#SBATCH --gres=gpu:1              # Number of GPUs per node
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --mem=320G                 # Memory per node
#SBATCH --time=12:00:00           # Maximum job time
#SBATCH --output=logs/%x-%j.out   # Output file (job-name_job-id.out)

export DNAGEN_DIR=/home/kevqyzhu/dna-gen
export DART_WORK_DIR=/large_storage/hsulab/kzhu/DART_work_dir
# export MODEL_SPECIFIC_NAME=evo2_cascade_1p5_7b_458k_35.norm
export MODEL_SPECIFIC_NAME=evo2_nv_7b_500k_sequential.26.mlp

cd $DNAGEN_DIR
SLURM_JOB_NUM_NODES=1 CUDA_VISIBLE_DEVICES=0 python -m eval.DART-Eval.src.dnalm_bench.task_2_5_single.experiments.task_3_peak_classification.extract_embeddings.evo
python -m eval.DART-Eval.src.dnalm_bench.task_2_5_single.experiments.task_3_peak_classification.cluster.run_clustering_subset $DART_WORK_DIR/task_3_peak_classification/embeddings/$MODEL_SPECIFIC_NAME.h5 $DART_WORK_DIR/task_3_peak_classification/processed_inputs/peaks_by_cell_label_unique_dataloader_format.tsv $DART_WORK_DIR/task_3_peak_classification/processed_inputs/indices_of_new_peaks_in_old_file.tsv $DART_WORK_DIR/task_3_peak_classification/clustering/$MODEL_SPECIFIC_NAME

