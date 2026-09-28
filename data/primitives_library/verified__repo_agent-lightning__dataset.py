from collections.abc import Sequence
from typing import Any

import torch
from datasets import Dataset as HuggingFaceDataset
from verl.utils.dataset.rl_dataset import RLHFDataset

__all__ = ["LoadedDataset"]


class LoadedDataset(RLHFDataset):
    """Dataset wrapper for pre-loaded in-memory data sequences."""

    def __init__(self, dataset: Sequence[Any]):
        records = [dataset[index] for index in range(len(dataset))]
        self.dataframe = HuggingFaceDataset.from_list(records)
        self.filter_overlong_prompts = False
        self.serialize_dataset = True
        self.original_data_files = None

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        row_dict["index"] = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["fake_ids"] = torch.ones(1, dtype=torch.int)
        return row_dict

    def _read_files_and_tokenize(self):
        pass