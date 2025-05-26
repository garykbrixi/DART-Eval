import os
import sys

from ....evaluators import EvoEvaluator
from ....components import FootprintingDataset

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    # model_name = "evo2_cascade_1p5_7b_458k"
    # model_name = "evo"
    model_name = "evo2_nv_7b_500k"
    seq_table = os.path.join(root_output_dir, f"task_2_footprinting/processed_data/footprint_dataset_350_v1.txt")
    batch_size = 256
    num_workers = 0
    seed = 0
    device = "cuda:0"

    out_dir = os.path.join(root_output_dir,"task_2_footprinting/outputs/likelihoods/")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(root_output_dir,f"task_2_footprinting/outputs/likelihoods/{model_name}.tsv")


    dataset = FootprintingDataset(seq_table, seed)
    evaluator = EvoEvaluator(model_name, batch_size, num_workers, device)
    evaluator.evaluate(dataset, out_path, progress_bar=True)
