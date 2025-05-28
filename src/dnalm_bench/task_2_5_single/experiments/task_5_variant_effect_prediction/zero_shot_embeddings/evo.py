import os
import sys
import numpy as np
import polars as pl

from ....evaluators import EvoVariantEmbeddingEvaluator
from ....components import VariantDataset

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    dataset = sys.argv[1]
    model_name = "evo2_7b_base"
    layer_name = "blocks.28.mlp.l3"
    # model_name = "evo2_nv_7b_500k"
    # layer_name = "sequential.26.mlp"
    batch_size = 32
    num_workers = 0
    seed = 0
    device = "cuda:0"
    chroms=None

    variants_bed = sys.argv[1]
    output_prefix = sys.argv[2]
    genome_fa = sys.argv[3]
    cell_line = "GM12878"

    # out_dir = os.path.join(root_output_dir, f"task_5_variant_effect_prediction/outputs/zero_shot/embeddings/{model_name}")
    out_dir = os.path.join(root_output_dir, f"task_5_variant_effect_prediction/outputs/zero_shot/embeddings/{model_name}_{layer_name}")
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, output_prefix + ".tsv")

    allele1_embeddings_path = os.path.join(out_dir, f"{output_prefix}_allele1_embeddings.npy")
    allele2_embeddings_path = os.path.join(out_dir, f"{output_prefix}_allele2_embeddings.npy")

    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
    evaluator = EvoVariantEmbeddingEvaluator(model_name, layer_name, batch_size, num_workers, device)
    score_df, allele1_embeddings, allele2_embeddings = evaluator.evaluate(dataset, out_path, progress_bar=True)

    df = dataset.elements_df
    scored_df = pl.concat([df, score_df], how="horizontal")
    print(out_path)
    scored_df.write_csv(out_path, separator="\t")

    # Save embeddings
    np.save(allele1_embeddings_path, allele1_embeddings)
    np.save(allele2_embeddings_path, allele2_embeddings)

# if __name__ == "__main__":
#     model_name = "evo2_7b_base"
#     layer_name = "blocks.28.mlp.l3"
#     genome_fa = os.path.join(root_output_dir, f"refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
#     cell_line = sys.argv[1] #cell line name
#     category = sys.argv[2] #peaks, nonpeaks, or idr
#     if category == "idr":
#         elements_tsv = os.path.join(root_output_dir, f"task_4_chromatin_activity/processed_data/cell_line_idr_peaks/{cell_line}.bed")
#     else:
#         elements_tsv = os.path.join(root_output_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_{category}.bed")
#     # chroms = ["chr22"]
#     chroms = None
#     batch_size = 32
#     num_workers = 0
#     seed = 0
#     device = "cuda"

#     out_dir = os.path.join(root_output_dir, f"task_4_chromatin_activity/embeddings/{model_name}_{layer_name}/")
#     os.makedirs(out_dir, exist_ok=True)
#     out_path = os.path.join(out_dir, f"{cell_line}_{category}.h5")

#     dataset = SimpleSequence(genome_fa, elements_tsv, chroms, seed)
#     extractor = Evo2EmbeddingExtractor(model_name, layer_name, batch_size, num_workers, device)
#     extractor.extract_embeddings(dataset, out_path, progress_bar=True)
