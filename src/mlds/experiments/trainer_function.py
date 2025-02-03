from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from transformers.tokenization_utils_base import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import *
import torch


def torch_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


@dataclass
class DataCollatorWithPaddingForT5:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_column_name: Optional[str] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        final_batch = {}
        for k in ["source_", "target_"]:
            in_feature = {
                "input_ids": [feature[k+"ids"] for feature in features],
                "attention_mask": [feature[k+"mask"] for feature in features],
            }
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                in_feature,
                padding=self.padding,
                # truncation=True,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            final_batch[k+"ids"] = batch["input_ids"]
            final_batch[k+"mask"] = batch["attention_mask"]

        return final_batch


@dataclass
class DataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_column_name: Optional[str] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # print(features[0].keys())
        input_features = [{k: v for k, v in feature.items() if k not in self.label_column_name} for feature in features]
        # print(self.label_column_name)
        # print(input_features[0].keys())
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        col = "seq_labels"
        batch[col] = torch.tensor([f[col] for f in features], dtype=torch.long)
        return batch


@dataclass
class DataCollatorForTokenClassification(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    label_column_name: Optional[str] = None

    def torch_call(self, features):
        import torch

        label_name = "token_labels" # "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)

        batch["seq_labels"] = torch.tensor([f["seq_labels"] for f in features], dtype=torch.long)
        return batch
