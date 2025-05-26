#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user="kevin.zhu@arcinstitute.org"
#SBATCH --job-name=DART_embedding_eval_t2
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1                # Total number of tasks (e.g., 4 for 4 GPUs)
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --mem=640G                 # Memory per node
#SBATCH --time=12:00:00           # Maximum job time
#SBATCH --output=logs/%x-%j.out   # Output file (job-name_job-id.out)

# source /opt/conda/etc/profile.d/conda.sh
# conda activate evo2

export DNAGEN_DIR=/home/kevqyzhu/dna-gen
export DART_WORK_DIR=/large_storage/hsulab/kzhu/DART_work_dir
export MODEL_SPECIFIC_NAME=evo2_cascade_1p5_7b_458k_35.norm

cd $DNAGEN_DIR
python -m eval.DART-Eval.src.dnalm_bench.task_2_5_single.experiments.task_2_transcription_factor_binding.footprint_eval_embeddings --input_seqs $DART_WORK_DIR/task_2_footprinting/processed_data/footprint_dataset_350_v1.txt --embeddings $DART_WORK_DIR/task_2_footprinting/outputs/embeddings/$MODEL_SPECIFIC_NAME.h5 --output_file $DART_WORK_DIR/task_2_footprinting/outputs/evals/embeddings/$MODEL_SPECIFIC_NAME.tsv

