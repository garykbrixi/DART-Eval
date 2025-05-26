import os
import sys

from ....embeddings import Evo2EmbeddingExtractor
from ....components import SimpleSequence

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    # model_name = "evo2_cascade_1p5_7b_458k"
    # layer_name = "35.norm"
    model_name = "evo2_nv_7b_500k"
    layer_name = "sequential.26.mlp"
    genome_fa = os.path.join(root_output_dir,"refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    elements_tsv = os.path.join(root_output_dir,"task_3_peak_classification/processed_inputs/peaks_by_cell_label_unique_dataloader_format.tsv")
    chroms = None
    batch_size = 64
    num_workers = 0
    seed = 0
    device = "cuda:0"

    out_dir = os.path.join(root_output_dir,"task_3_peak_classification/embeddings/")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(root_output_dir,f"task_3_peak_classification/embeddings/{model_name}_{layer_name}.h5")

    dataset = SimpleSequence(genome_fa, elements_tsv, chroms, seed)
    extractor = Evo2EmbeddingExtractor(model_name, layer_name, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)
