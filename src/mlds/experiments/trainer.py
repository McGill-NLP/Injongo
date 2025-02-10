from datasets import Dataset, DatasetDict
import numpy as np
import pandas as pd
from mlds.data_loader import SlotDataManager, LANGUAGES


def get_datasets(lan):
    slot_manager = SlotDataManager(data_folder="data/output")

    if lan == "all":
        results = {"train": [], "dev": [], "test": []}
        for lan in sorted(LANGUAGES + ["eng"]):
            merged = slot_manager.load_data(lan, split="split")
            for split in ["train", "test", "dev"]:
                results[split].append(merged[split])
        for split in ["train", "test", "dev"]:
            results[split] = pd.concat(results[split], ignore_index=True)

        # Check for consistency in the data
        for split in ["train", "test", "dev"]:
            for column in results[split].columns:
                if results[split][column].apply(lambda x: isinstance(x, list)).any():
                    results[split][column] = results[split][column].apply(lambda x: x if isinstance(x, list) else [x])
        
        return DatasetDict({
            "train": Dataset.from_pandas(results["train"]),
            "validation": Dataset.from_pandas(results["dev"]),
            "test": Dataset.from_pandas(results["test"]),
        })

    if lan == "eng+clinc":
        results = {"train": [], "dev": [], "test": []}
        for lan in ["eng", "clinc"]:
            merged = slot_manager.load_data(lan, split="split")
            for split in ["train", "test", "dev"]:
                tmp_df = merged[split][['intent', 'text', 'domain', 'spans', 'logical_form', 'xtreme-up', 'source_language']]
                results[split].append(tmp_df)

        for split in ["train", "test", "dev"]:
            results[split] = pd.concat(results[split])

        print(results["train"].shape, results["dev"].shape, results["test"].shape)

        return DatasetDict({
            "train": Dataset.from_pandas(results["train"]),
            "validation": Dataset.from_pandas(results["dev"]),
            "test": Dataset.from_pandas(results["test"]),
        })

    if lan == "clinc+extend":
        results = {"train": [], "dev": [], "test": []}
        for lan in ["clinc", "clinc+extend"]:
            merged = slot_manager.load_data(lan, split="split")
            for split in ["train", "test", "dev"]:
                results[split].append(merged[split])

        for split in ["train", "test", "dev"]:
            results[split] = pd.concat(results[split], ignore_index=True)

        print(results["train"].shape, results["dev"].shape, results["test"].shape)

        return DatasetDict({
            "train": Dataset.from_pandas(results["train"]),
            "validation": Dataset.from_pandas(results["dev"]),
            "test": Dataset.from_pandas(results["test"]),
        })

    if lan == "eng+40shots":
        results = slot_manager.load_data("eng", split="split")
        print(results["train"].shape, results["dev"].shape, results["test"].shape)
        examples = slot_manager.get_fewshot_examples("eng", 40)
        print(results["train"].head(), examples.head())
        results["train"] = pd.concat([results["train"], examples], ignore_index=True)
        print(results["train"].shape, results["dev"].shape, results["test"].shape)

        return DatasetDict({
            "train": Dataset.from_pandas(results["train"]),
            "validation": Dataset.from_pandas(results["dev"]),
            "test": Dataset.from_pandas(results["test"]),
        })

    if lan == "eng+23shots":
        results = slot_manager.load_data("eng", split="split")
        print(results["train"].shape, results["dev"].shape, results["test"].shape)
        examples = slot_manager.get_fewshot_examples("eng", 23)
        print(results["train"].head(), examples.head())
        results["train"] = pd.concat([results["train"], examples], ignore_index=True)
        print(results["train"].shape, results["dev"].shape, results["test"].shape)

        return DatasetDict({
            "train": Dataset.from_pandas(results["train"]),
            "validation": Dataset.from_pandas(results["dev"]),
            "test": Dataset.from_pandas(results["test"]),
        })

    # if lan == "eng+5shots":
    #     results = slot_manager.load_data("eng", split="split")
    #     examples = slot_manager.get_fewshot_examples("eng", 5)
    #     results["train"] = pd.concat([results["train"], examples], ignore_index=True)
    #     print(results["train"].shape, results["dev"].shape, results["test"].shape)

    #     return DatasetDict({
    #         "train": Dataset.from_pandas(results["train"]),
    #         "validation": Dataset.from_pandas(results["dev"]),
    #         "test": Dataset.from_pandas(results["test"]),
    #     })

    # merged = slot_manager.load_data(lan, split="full")
    merged = slot_manager.load_data(lan, split="split")
    for split in ["train", "test", "dev"]:
        merged[split] = Dataset.from_pandas(merged[split])

    print(merged["train"].shape, merged["dev"].shape, merged["test"].shape)
    return DatasetDict({
        "train": merged["train"],
        "validation": merged["dev"],
        "test": merged["test"],
    })


# Normal NER Tranier
from transformers import Trainer, Seq2SeqTrainer
import evaluate

seq_metric = evaluate.load("accuracy")
token_metric = evaluate.load("seqeval")

class BaseTrainer(Trainer):
    pass

class Seq2SeqTrainer(Seq2SeqTrainer):
    pass
