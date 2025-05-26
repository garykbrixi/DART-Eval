import os
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForCausalLM, BertConfig
from scipy.stats import wilcoxon
from tqdm import tqdm
import h5py

from ..components import PairedControlDataset
from ...utils import onehot_to_chars
from ...embeddings import HFEmbeddingExtractor, SequenceBaselineEmbeddingExtractor, EmbeddingExtractor
from ......models import load_model


class PairedControlEmbeddingExtractor:
    _idx_mode = "variable"

    @staticmethod
    def _offsets_to_indices(offsets, seqs):
        gather_idx = np.zeros((seqs.shape[0], seqs.shape[1]), dtype=np.uint32)
        for i, offset in enumerate(offsets):
            for j, (start, end) in enumerate(offset):
                gather_idx[i,start:end] = j
        
        return gather_idx

    def extract_embeddings(self, dataset, out_path, progress_bar=False):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        with h5py.File(out_path + ".tmp", "w") as out_f:
            seq_grp = out_f.create_group("seq")
            ctrl_grp = out_f.create_group("ctrl")

            start = 0
            for seqs, ctrls, idx_orig in tqdm(dataloader, disable=(not progress_bar)):
                end = start + len(seqs)

                seq_tokens, seq_offsets = self.tokenize(seqs)
                ctrl_tokens, ctrl_offsets = self.tokenize(ctrls)

                seq_token_emb = self.model_fwd(seq_tokens)
                ctrl_token_emb = self.model_fwd(ctrl_tokens)

                if self._idx_mode == "variable":
                    seq_indices = self._offsets_to_indices(seq_offsets, seqs)
                    seq_indices_dset = seq_grp.require_dataset("idx_var", (len(dataset), seq_indices.shape[1]), dtype=np.uint32)
                    seq_indices_dset[start:end] = seq_indices

                    ctrl_indices = self._offsets_to_indices(ctrl_offsets, ctrls)
                    ctrl_indices_dset = ctrl_grp.require_dataset("idx_var", (len(dataset), ctrl_indices.shape[1]), dtype=np.uint32)
                    ctrl_indices_dset[start:end] = ctrl_indices

                elif (start == 0) and (self._idx_mode == "fixed"):
                    seq_indices = self._offsets_to_indices(seq_offsets, seqs)
                    seq_indices_dset = seq_grp.create_dataset("idx_fix", data=seq_indices, dtype=np.uint32)
                    ctrl_indices = self._offsets_to_indices(ctrl_offsets, ctrls)
                    ctrl_indices_dset = ctrl_grp.create_dataset("idx_fix", data=ctrl_indices, dtype=np.uint32)

                seq_grp.create_dataset(f"emb_{start}_{end}", data=seq_token_emb.numpy(force=True))
                ctrl_grp.create_dataset(f"emb_{start}_{end}", data=ctrl_token_emb.numpy(force=True))

                start = end

        os.rename(out_path + ".tmp", out_path)


class SequenceBaselinePairedControlEmbeddingExtractor(SequenceBaselineEmbeddingExtractor, PairedControlEmbeddingExtractor):
    _idx_mode = "fixed"

    @staticmethod
    def _offsets_to_indices(offsets, seqs):
        slice_idx = [0, seqs.shape[1]]
        
        return np.array(slice_idx)


class DNABERT2EmbeddingExtractor(HFEmbeddingExtractor, PairedControlEmbeddingExtractor):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"zhihan1996/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        config = BertConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, config=config, trust_remote_code=True)
        # model = AutoModelForMaskedLM.from_config(config)
        super().__init__(tokenizer, model, batch_size, num_workers, device)


class GenaLMEmbeddingExtractor(HFEmbeddingExtractor, PairedControlEmbeddingExtractor):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"AIRI-Institute/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)


class HyenaDNAEmbeddingExtractor(HFEmbeddingExtractor, PairedControlEmbeddingExtractor):
    _idx_mode = "fixed"

    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"LongSafari/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
        model =  AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    def tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]

        return tokens, None

    @staticmethod
    def _offsets_to_indices(offsets, seqs):
        slice_idx = [0, seqs.shape[1]]
        
        return np.array(slice_idx)


class MistralDNAEmbeddingExtractor(HFEmbeddingExtractor, PairedControlEmbeddingExtractor):
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"RaphaelMourad/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model =  AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)


class NucleotideTransformerEmbeddingExtractor(HFEmbeddingExtractor, PairedControlEmbeddingExtractor):
    _idx_mode = "fixed"

    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"InstaDeepAI/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model =  AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    def tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]

        return tokens, None

    @staticmethod
    def _offsets_to_indices(offsets, seqs):
        seq_len = seqs.shape[1]
        inds = np.zeros(seq_len, dtype=np.int32)
        # seq_len_contig = (seq_len // 6) * 6
        for i in range(seq_len // 6):
            inds[i*6:(i+1)*6] = i + 1
        inds[(i+1)*6:] = np.arange(i+2, i+(seq_len%6)+2)

        return inds
    
class CaduceusEmbeddingExtractor(HFEmbeddingExtractor, PairedControlEmbeddingExtractor):
    _idx_mode = "fixed"
    
    def __init__(self, model_name, batch_size, num_workers, device):
        model_name = f"kuleshov-group/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    def tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]

        return tokens, None
    
    @staticmethod
    def _offsets_to_indices(offsets, seqs):
        slice_idx = [0, seqs.shape[1]]
        
        return np.array(slice_idx)
    

class Evo2EmbeddingExtractor(EmbeddingExtractor, PairedControlEmbeddingExtractor):
    _idx_mode = "fixed"

    def __init__(self, model_name, layer_name, batch_size, num_workers, device):
        evo_model = load_model(
            model_name,
            device=device,
            )
        model, tokenizer = evo_model.model, evo_model.tokenizer
        self.tokenizer = tokenizer
        self.model = model
        self.model.to(device)
        model.eval()

        self.layer_name = layer_name

        super().__init__(batch_size, num_workers, device)

    
    def tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer.tokenize_batch(seqs_str)
        tokens = torch.tensor(encoded, dtype=torch.int64)

        return tokens, None


    def model_fwd(self, tokens):
        # Check if tokens is a list and convert to tensor
        if isinstance(tokens, list):
            tokens = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(seq) for seq in tokens], 
                batch_first=True
            )

        tokens = tokens.to(device=self.device)

        # List to hold the extracted embeddings
        embedding_output = []

        def get_embedding_output(module, input, output):
            embedding_output.append(output)

        # Register the forward hook for the specified layer
        layer_to_hook = self.model.get_submodule(self.layer_name)

        with layer_to_hook.register_forward_hook(get_embedding_output) as handle:
            # Pass through the model without computing gradients
            with torch.no_grad():
                self.model(tokens)

        # Check if embeddings were captured
        if embedding_output:
            embedding_output = embedding_output[0]

            if not isinstance(embedding_output, torch.Tensor):
                embedding_output = embedding_output[0]
                
            # Remove nearest_16 padding
            return embedding_output[:, :tokens.shape[1], :].float().cpu()

    @staticmethod
    def _offsets_to_indices(offsets, seqs):
        slice_idx = [0, seqs.shape[1]]
        
        return np.array(slice_idx)