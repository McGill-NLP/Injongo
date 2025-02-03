import json
import os
from pathlib import Path
from typing import Dict, List
import random
from jinja2 import Template

from mlds.data_loader import (
    LANGUAGES,
    LANGUAGES_MAPPER,
    SlotDataManager,
)

from mlds.experiments.prompt import PROMPTS

TOKENC_SYSTEM_PROMPT = """You are an expert in {language} language processing, specialized in named entity recognition. Extract slots from user text according to the specified entity types."""

SEQC_SYSTEM_PROMPT = """You are an expert in {language} language processing, specialized in intent classification. Classify user text into one of the predefined intent categories."""

def generate_sft_data(slot_data_manager: SlotDataManager, language: str, split: str) -> Dict[str, List[Dict]]:
    """Generate SFT data for both tasks with specified split."""
    if language == "all":
        data = slot_data_manager.load_all_data(split=split)
    else:
        data = slot_data_manager.load_data(language, split=split)

    tokenc_samples = []
    seqc_samples = []
    
    seqc_template = Template(PROMPTS["intent-detection-five-round"])
    tokenc_template = Template(PROMPTS["slot-filling-five-round"])

    for _, row in data.iterrows():
        # Token classification samples
        tokenc_samples.append({
            "instruction": tokenc_template.render(
                text=row["text"].replace("\n", " ").replace('""', '"').strip(),
                round=random.randint(1, 5),
                shot_count=0,
            ).replace("\n# Format Example:\nSentence: John went to Paris and paid 100 dollars at an Awater restaurant.\nOutput: PERSONAL_NAME John $$ CITY_OR_PROVINCE Paris $$ MONEY 100 $$ RESTAURANT_NAME Awater\n", ""),
            "input": "",
            "output": row['xtreme-up'],
            "system": ""
            # "system": TOKENC_SYSTEM_PROMPT.format(language=LANGUAGES_MAPPER[row["source_language"]])
        })

        # Sequence classification samples
        seqc_samples.append({
            "instruction": seqc_template.render(
                text=row["text"].replace("\n", " ").replace('""', '"').strip(),
                round=random.randint(1, 5),
                shot_count=0,
            ).replace("\n# Format Example:\nSentence: Can you tell me the weather forecast for today?\nOutput: weather\n\n\n", ""),
            "input": "",
            "output": row['intent'],
            "system": ""
            # "system": SEQC_SYSTEM_PROMPT.format(language=LANGUAGES_MAPPER[row["source_language"]])
        })

    print(language, split, len(tokenc_samples), len(seqc_samples))
    return {
        "tokenc": tokenc_samples,
        "seqc": seqc_samples
    }

def save_sft_data(output_dir: Path, language: str, task: str, split: str, data: List[Dict]):
    """Save SFT data with organized folder structure."""
    # Create task/language directory
    task_dir = output_dir / task / language
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save split-specific data
    with open(task_dir / f"{split}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    # Initialize data manager
    slot_data_manager = SlotDataManager()
    
    # Create output directory
    output_dir = Path("data/sft-dataset")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate dataset_info.json
    dataset_info = {}
    splits = ["train", "dev", "test"]
    
    for language in LANGUAGES + ["eng", "all"]:
        # data_splits = slot_data_manager.load_data(language, split="split")
        
        # Process each split
        for split_name in splits:
            # split_data = data_splits[split_name]
            sft_data = generate_sft_data(slot_data_manager, language, split_name)
            
            split_name = {"train": "train", "dev": "validation", "test": "test"}[split_name]

            # Save token classification data
            dataset_name = f"sft_{language}_tokenc_{split_name}"
            dataset_info[dataset_name] = {
                "file_name": f"tokenc/{language}/{split_name}.json",
                "formatting": "alpaca",
                # "split": split_name,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system"
                }
            }
            save_sft_data(output_dir, language, "tokenc", split_name, sft_data["tokenc"])
            
            # Save sequence classification data
            dataset_name = f"sft_{language}_seqc_{split_name}"
            dataset_info[dataset_name] = {
                "file_name": f"seqc/{language}/{split_name}.json",
                "formatting": "alpaca",
                # "split": split_name,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system"
                }
            }
            save_sft_data(output_dir, language, "seqc", split_name, sft_data["seqc"])
    
    # Save dataset info
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()