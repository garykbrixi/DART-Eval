import os
import torch
from torch.utils.data import Subset
from ...embeddings import Evo2EmbeddingExtractor
from ....components import PairedControlDataset

work_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    # Model and layer info
    model_name = "evo2_cascade_1p5_7b_458k"
    layer_name = "35.norm"
    genome_fa = os.path.join(work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    elements_tsv = os.path.join(work_dir, "task_1_ccre/processed_inputs/ENCFF420VPZ_processed.tsv")
    chroms = None  # Can be adjusted for specific chromosomes
    batch_size = 96
    num_workers = 4
    seed = 0

    # Output directory and file setup
    out_dir = os.path.join(work_dir, "task_1_ccre/embeddings")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}_part_{os.environ['CUDA_VISIBLE_DEVICES']}.h5")

    # Load the dataset
    dataset = PairedControlDataset(genome_fa, elements_tsv, chroms, seed)

    # Split dataset into 4 parts
    dataset_length = len(dataset)
    part_size = dataset_length // 4
    start_idx = int(os.environ['CUDA_VISIBLE_DEVICES']) * part_size
    end_idx = (int(os.environ['CUDA_VISIBLE_DEVICES']) + 1) * part_size if int(os.environ['CUDA_VISIBLE_DEVICES']) < 3 else dataset_length
    dataset_part = Subset(dataset, range(start_idx, end_idx))

    # Set device to the current GPU
    # device = torch.device(f"cuda:{os.environ['CUDA_VISIBLE_DEVICES']}")
    device = "cuda:0"

    # Embedding extractor
    extractor = Evo2EmbeddingExtractor(model_name, layer_name, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset_part, out_path, progress_bar=True)
