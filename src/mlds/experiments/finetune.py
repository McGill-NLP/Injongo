import argparse
import json
import os
from typing import List

import torch
os.environ["WANDB_SILENT"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["WANDB_DISABLED"] = "true"

from mlds.data_loader import INTENTS, LANGUAGES, SLOTS_MERGED, NLLB_MAPPER, SlotDataManager
from mlds.utils.lr import get_learning_rate
from mlds.utils.path_utils import model_basename
import evaluate

### Receive Augmentation
parser = argparse.ArgumentParser()
parser.add_argument(
    "language", type=str, choices=LANGUAGES + ["all", "eng", "clinc", "eng+clinc", "clinc+extend", "eng+1shot"] + [lang+"_eng" for lang in LANGUAGES] + [f"clinc_{shot}shots" for shot in (5,10, 25, 50, 100)] + [f"eng_{shot}shots" for shot in (5,10,25)], default="amh"
)
parser.add_argument("task", type=str, choices=["seqc", "tokenc"])
parser.add_argument("model", type=str)
parser.add_argument("--fold", type=int, default=-1)
parser.add_argument("--seed", "-s", type=int, default=57706989)
parser.add_argument("--eval", "-e", action="store_true")
parser.add_argument("--lr", type=float, default=0.0)
parser.add_argument("--local-rank", type=int, default=-1)
args = parser.parse_args()


import pandas as pd
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

set_seed(args.seed)

# Define Parameter
model_checkpoint = args.model
model_name = model_basename(model_checkpoint)

if args.language == "eng+1shot":
    if args.task == "seqc":
        args.language = "eng+40shots"
    else:
        args.language = "eng+23shots"

if args.seed == 57706989:
    output_dir_hash = f"finetune/{args.task}/{model_name}/test/{args.language}"
else:
    output_dir_hash = f"finetune/{args.task}/{model_name}/{args.seed}/{args.language}"

# if (os.path.exists("./data/weight/" + output_dir_hash + "/config.json") or 
#     os.path.exists("./data/weight/" + output_dir_hash + "/adapter_config.json")) and not args.eval:
#     print("Results exists... Skip!")
#     exit()

if os.path.exists("./data/results/" + output_dir_hash + ".json") and not args.eval:
    print("Results exists... Skip!")
    exit()

batch_size = 32 // torch.cuda.device_count()
if any(x in model_checkpoint.lower() for x in ["3b", "8b", "70b", "27b"]):
    batch_size //= 2

lr = args.lr if args.lr else get_learning_rate(model_checkpoint, args.task)
gradient_accumulation_steps = 64 // batch_size

intents_list = INTENTS
intents_l2id = {x: i for i, x in enumerate(intents_list)}
intents_id2l = dict(enumerate(intents_list))
intent_label_col_name = "intent"  # seq classification

slots_list = ["O"] + ["B-" + s for s in SLOTS_MERGED] + ["I-" + s for s in SLOTS_MERGED]
slots_l2id = {x: i for i, x in enumerate(slots_list)}
slots_id2l = dict(enumerate(slots_list))
# slot_label_col_name = "spans"  # token classification

from peft import get_peft_model
from peft.tuners.lora.config import LoraConfig

if "NLLB-LLM2Vec-Meta-Llama-31-8B" in model_checkpoint:
    # Only attach LoRAs to the linear layers of LLM2Vec inside NLLB-LLM2Vec
    lora_config = LoraConfig(
        lora_alpha=32,
        target_modules=r".*llm2vec.*(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj).*",
        bias="none",
        task_type="SEQ_CLS",
    )
elif (
    "8b" in model_checkpoint.lower()
    or "70b" in model_checkpoint.lower()
    or "27b" in model_checkpoint.lower()
):
    lora_config = LoraConfig(
        lora_alpha=32,
        # target_modules = r".*llm2vec.*linear.*",
        bias="none",
        # task_type = "SEQ_CLS"
    )
else:
    lora_config = None


if args.task == "seqc":
    config, additional_config = AutoConfig.from_pretrained(
        model_checkpoint,  # id2label=id2l, label2id=l2id, num_labels=len(label_list)
        num_labels=len(intents_list),
        finetuning_task="text-classification",
        return_unused_kwargs=True,
        trust_remote_code=True,
    )
    config.problem_type = "single_label_classification"

elif args.task == "tokenc":
    config, additional_config = AutoConfig.from_pretrained(
        model_checkpoint,  # id2label=id2l, label2id=l2id, num_labels=len(label_list)
        num_labels=len(slots_list),
        finetuning_task="token-classification",
        return_unused_kwargs=True,
        id2label=slots_id2l,
        label2id=slots_l2id,
        trust_remote_code=True,
    )

if (
    config.model_type in {"bloom", "gpt2", "roberta", "deberta"}
    and args.task == "tokenc"
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        use_fast=True,
        trust_remote_code=True,
        add_prefix_space=True,
    )
elif config.model_type in {"nllb-llm2vec"}:
    tokenizer = AutoTokenizer.from_pretrained(
        "fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse",
        use_fast=True,
        trust_remote_code=True,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        # cache_dir=model_args.cache_dir,
        use_fast=True,
        # revision=model_args.model_revision,
        # token=model_args.token,
        trust_remote_code=True,
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"tokenizer.pad_token is None, set to: {tokenizer.eos_token}")

if config.pad_token_id is None:
    config.pad_token_id = tokenizer.pad_token_id
    print(f"config.pad_token_id is None, set to: {tokenizer.pad_token_id}")


# TODO: construct language mappings from your languages to NLLB languages
# https://huggingface.co/facebook/nllb-200-distilled-600M/blob/main/special_tokens_map.json
from transformers import BatchEncoding

def tokenc_preprocess(data: pd.DataFrame):
    from mlds.utils.converter import create_tokens_and_tags

    tokens_ll, tags_ll = [], []

    def to_dict(l):
        res = []
        if not l:
            return res
        for x in l.split(","):
            s, e, _, t = x.split(":")
            res.append({"start": int(s), "end": int(e), "labels": [t]})
        return res

    for text, spans in zip(data["text"], data["spans"]):
        tokens, tags = create_tokens_and_tags(
            text=text, spans=to_dict(spans), format="BIO"
        )
        tokens_ll.append(tokens)
        tags_ll.append(tags)

    data["tokens"] = tokens_ll
    data["spans"] = tags_ll
    
    # INFO: added on-the-fly change of `tokenizer.src_lang`
    if not hasattr(tokenizer, "src_lang"):
        tokenized_inputs = tokenizer(
            data["tokens"], truncation=True, is_split_into_words=True, padding=False
        )
    else:
        if len(set(data["source_language"])) > 1:
            for i, (lan, token) in enumerate(zip(data["source_language"], data["tokens"])):
                tokenizer.src_lang = NLLB_MAPPER[lan]
                if i == 0:
                    # print("tokenizer.src_lang", tokenizer.src_lang)
                    tokenized_inputs = tokenizer(
                        [token], truncation=True, is_split_into_words=True, padding=False
                    )
                else:
                    new_tokenized_inputs = tokenizer(
                        [token], truncation=True, is_split_into_words=True, padding=False
                    )
                    for key in tokenized_inputs.data:
                        tokenized_inputs.data[key] += new_tokenized_inputs.data[key]
                    tokenized_inputs = BatchEncoding(
                        data=tokenized_inputs.data,
                        encoding=tokenized_inputs.encodings + new_tokenized_inputs.encodings
                    )
        else:
            tokenizer.src_lang = NLLB_MAPPER[data["source_language"][0]]
            tokenized_inputs = tokenizer(
                data["tokens"], truncation=True, is_split_into_words=True, padding=False
            )

    labels = []
    word_ids_list = []
    for i, label in enumerate(data["spans"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100) # label[word_idx] if label_all_tokens else
            previous_word_idx = word_idx

        label_ids = [slots_l2id[x] if type(x) != int else x for x in label_ids]
        labels.append(label_ids)
        word_ids_list.append(word_ids)

    tokenized_inputs["labels"] = labels
    tokenized_inputs["word_ids"] = word_ids_list
    return tokenized_inputs


def seqc_preprocess(data: pd.DataFrame):
    # INFO: added on-the-fly change of `tokenizer.src_lang`
    if not hasattr(tokenizer, "src_lang"):
        tokenized_inputs = tokenizer(data["text"], truncation=True)
    else:
        if len(set(data["source_language"])) > 1:
            for i, lan in enumerate(data["source_language"]):
                tokenizer.src_lang = NLLB_MAPPER[lan]
                if i == 0:
                    tokenized_inputs = tokenizer(
                        [data["text"][i]], truncation=True
                    )
                else:
                    new_tokenized_inputs = tokenizer(
                        [data["text"][i]], truncation=True
                    )
                    for key in tokenized_inputs.data:
                        tokenized_inputs.data[key] += new_tokenized_inputs.data[key]
                    tokenized_inputs = BatchEncoding(
                        data=tokenized_inputs.data,
                        encoding=tokenized_inputs.encodings + new_tokenized_inputs.encodings
                    )
        else:
            tokenizer.src_lang = NLLB_MAPPER[data["source_language"][0]]
            tokenized_inputs = tokenizer(data["text"], truncation=True)
    # if hasattr(tokenizer, "src_lang"):
    #     tokenizer.src_lang = NLLB_MAPPER[data["source_language"][0]]
    # tokenized_inputs = tokenizer(data["text"], truncation=True)
    tokenized_inputs["labels"] = [intents_l2id[x] for x in data[intent_label_col_name]]
    return tokenized_inputs

data_preprocess = {
    "tokenc": tokenc_preprocess,
    "seqc": seqc_preprocess,
    # "jointc": tokenc_preprocess,
}

print("CUDA available: ", torch.cuda.is_available())
print("CUDA device count: ", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")

from .trainer import get_datasets, BaseTrainer

# TODO: here you need to pass correct language code
# TODO: construct language mappings from your languages to NLLB languages
# https://huggingface.co/facebook/nllb-200-distilled-600M/blob/main/special_tokens_map.json
tokenized_datasets = get_datasets(args.language).map(
    # FIXME: args.language currently most likely will not be the correct language code
    data_preprocess[args.task],
    # fn_kwargs={"language": args.language},
    batched=True,
)

# if not args.eval:
#     import wandb

#     wandb.init(
#         project="Multilingual-Dataset-FT",
#         name=f"{model_basename(model_checkpoint)}/{args.task}/{args.language}",
#     )
# import random

trainer_args = TrainingArguments(
    output_dir=f"/tmp/data/weight/{output_dir_hash}",  # output tmp dir
    run_name=f"{args.task}/{model_name}/{args.language}",
    # label_names=["seq_labels", "token_labels"],
    label_names=["labels"],
    overwrite_output_dir=True,
    seed=args.seed,  # + random.randint(100, 1000),
    eval_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=20,
    weight_decay=0.01,
    # report_to="wandb" if not args.eval else "none",
    save_safetensors=True,
    save_strategy="epoch",
    # save_total_limit=10,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    metric_for_best_model={"tokenc": "eval_f1", "seqc": "eval_accuracy"}[args.task],
    greater_is_better=True,
    load_best_model_at_end=True,
    local_rank=args.local_rank,
    ddp_find_unused_parameters=False,
)

early_stopping = EarlyStoppingCallback(early_stopping_patience=5)

if args.task == "seqc":
    if args.eval:
        model = AutoModelForSequenceClassification.from_pretrained(
            f"./data/weight/{output_dir_hash}",
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
            config=config,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
            config=config,
        )
        # model.config.num_labels = len(intents_list)
        model.config.intents_l2id = intents_l2id
        model.config.intents_id2l = intents_id2l

        if lora_config:
            model = get_peft_model(model, lora_config)

    import numpy as np
    import evaluate

    seq_metric = evaluate.load(
        "accuracy", experiment_id=output_dir_hash.replace("/", "_")
    )

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = seq_metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        print("compute_metrics: seqc: ", result)
        return result

    trainer = BaseTrainer(
        model,
        trainer_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

elif args.task == "tokenc":
    if args.eval:
        model = AutoModelForTokenClassification.from_pretrained(
            f"./data/weight/{output_dir_hash}",
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
            config=config,
        )
    else:
        # print(model_checkpoint, type(config))
        model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
            config=config,
        )
        if lora_config:
            model = get_peft_model(model, lora_config)

    # https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py
    # data_collator =  #, label_column_name=["seq_labels", "token_labels"])
    # data_collator.num_labels = len(slots_list)
    # from transformers import pipeline
    # from mlds.metrics import span_f1

    # pipe = pipeline(
    #     "ner",
    #     model=model,
    #     tokenizer=tokenizer,
    #     ignore_labels=[-100],
    #     aggregation_strategy="simple",
    #     device="cuda",
    # )

    import numpy as np

    token_metric = evaluate.load(
        "seqeval", experiment_id=output_dir_hash.replace("/", "_")
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [slots_id2l[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [slots_id2l[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = token_metric.compute(
            predictions=true_predictions, references=true_labels
        )
        result = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

        strict_result = token_metric.compute(
            predictions=true_predictions, references=true_labels, mode="strict"
        )
        result["strict_f1"] = strict_result["overall_f1"]
        result["strict_precision"] = strict_result["overall_precision"]
        result["strict_recall"] = strict_result["overall_recall"]

        print("compute_metrics: tokenc: ", result)
        return result

    # print("Padding Side:", tokenizer.padding_side)
    trainer = BaseTrainer(
        model,
        trainer_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],  #  if do_eval else None,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

else:
    raise NotImplementedError


if not args.eval:
    print("!!!!!", args.language, "!!!!!")
    print("lr", trainer_args.to_dict()["learning_rate"])
    trainer.train(resume_from_checkpoint=args.eval)
    if lora_config:
        model.save_pretrained(f"./data/weight/{output_dir_hash}")
    else:
        trainer.save_model(f"./data/weight/{output_dir_hash}")

    os.system(f"rm -rf /tmp/data/weight/{output_dir_hash}")

    if args.lr != 0.0:
        # validation_results = compute_metrics(None, split="validation")
        # if args.task == "seqc":
        #     validation_results = trainer.evaluate(
        #         tokenized_datasets["validation"], metric_key_prefix=""
        #     )
        # elif args.task == "tokenc":
        #     validation_results = compute_metrics(
        #         None, dataset=tokenized_datasets["validation"]
        #     )
        validation_results = trainer.evaluate(
            tokenized_datasets["validation"], metric_key_prefix=""
        )

        with open(f"./data/results/finetune/log-v3.jsonl", "a") as f:
            validation_results["model"] = model_basename(model_checkpoint)
            validation_results["task"] = args.task
            validation_results["language"] = args.language
            validation_results["lr"] = args.lr
            f.write(json.dumps(validation_results) + "\n")
            f.flush()

        exit()

results = {}
# TODO: here you need to pass correct language code
for lan in LANGUAGES + ["eng"]:
    tokenized_test_dataset = get_datasets(lan)["test"].map(
        # FIXME: lan is not correct here
        data_preprocess[args.task],
        # fn_kwargs={"language": lan},
        batched=True,
    )

    # if args.task == "seqc":
    test_results = trainer.evaluate(tokenized_test_dataset, metric_key_prefix="test")
    results[lan] = test_results
    # elif args.task == "tokenc":
    #     results[lan] = compute_metrics(None, dataset=tokenized_test_dataset)

if args.task == "seqc" and ("eng" in args.language or "clinc" in args.language or "shots" in args.language):
    for lan in LANGUAGES:
        print(f"{lan}: {results[lan]}")
        lan = lan + "_eng"
        tokenized_test_dataset = get_datasets(lan)["test"].map(
            data_preprocess[args.task],
            batched=True,
        )

        test_results = trainer.evaluate(tokenized_test_dataset, metric_key_prefix="test")
        results[lan] = test_results

if args.local_rank != -1 and args.local_rank != 0:
    exit()

os.makedirs(os.path.dirname(f"./data/results/{output_dir_hash}.json"), exist_ok=True)

with open(f"./data/results/{output_dir_hash}.json", "w") as f:
    json.dump(results, f)
print(f"./data/results/{output_dir_hash}.json")
