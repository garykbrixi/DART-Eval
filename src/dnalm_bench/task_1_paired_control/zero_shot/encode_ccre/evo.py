import os

from ..evaluators import PairedControlDataset, EvoEvaluator

os.environ["TOKENIZERS_PARALLELISM"] = "false"

work_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    # model_name = "evo2_cascade_1p5_7b_458k"
    # model_name = "evo"
    model_name = "evo2_nv_7b_500k"
    genome_fa = os.path.join(work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    elements_tsv = os.path.join(work_dir, f"task_1_ccre/processed_inputs/ENCFF420VPZ_processed.tsv")

    out_dir = os.path.join(work_dir, f"task_1_ccre/zero_shot_outputs/likelihoods/{model_name}")

    chroms = [
        "chr5",
        "chr10",
        "chr14",
        "chr18",
        "chr20",
        "chr22"
    ]

    # batch_size = 64
    batch_size = 64
    num_workers = 4
    seed = 0
    device = "cuda:0"

    dataset = PairedControlDataset(genome_fa, elements_tsv, chroms, seed)
    evaluator = EvoEvaluator(model_name, dataset, batch_size, num_workers, device)
    metrics = evaluator.evaluate(out_dir, progress_bar=True)

    for k, v in metrics.items():
        print(f"{k}: {v}")
