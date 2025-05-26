#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user="kevin.zhu@arcinstitute.org"
#SBATCH --job-name=DART_embedding_extraction_t1
#SBATCH --partition=gpu_batch
#SBATCH --nodes=1
#SBATCH --ntasks=4                # Total number of tasks (4 for 4 GPUs)
#SBATCH --gres=gpu:4              # Number of GPUs per node
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --mem=320G                # Memory per node
#SBATCH --time=02:00:00           # Maximum job time
#SBATCH --output=logs/%x-%j.out   # Output file (job-name_job-id.out)

# source /opt/conda/etc/profile.d/conda.sh
# conda activate evo2

# Set up environment
export DNAGEN_DIR=/home/kevqyzhu/dna-gen
export DART_WORK_DIR=/large_storage/hsulab/kzhu/DART_work_dir

cd $DNAGEN_DIR

# Run the script on 4 GPUs in parallel with CUDA_VISIBLE_DEVICES
for gpu_id in {0..3}; do
    CUDA_VISIBLE_DEVICES=$gpu_id python -m eval.DART-Eval.src.dnalm_bench.task_1_paired_control.supervised.encode_ccre.extract_embeddings.evo_multi_gpu &
done
wait
