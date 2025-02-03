import argparse
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

from mlds.data_loader import INTENTS, LANGUAGES, SLOTS_MERGED, SlotDataManager
from mlds.utils.lr import get_learning_rate
from mlds.utils.path_utils import model_basename

### Receive Augmentation
parser = argparse.ArgumentParser()
parser.add_argument(
    "language", type=str, choices=LANGUAGES + ["all", "eng"], default="amh"
)
parser.add_argument("task", type=str, choices=["seqc", "tokenc", "jointc"])
parser.add_argument("model", type=str)
parser.add_argument("--fold", type=int, default=-1)
parser.add_argument("--seed", "-s", type=int, default=57706989)
parser.add_argument("--eval", "-e", action="store_true")
parser.add_argument("--lr", type=float, default=0.0)
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

assert args.model in ["google/mt5-large", "castorini/afriteva_v2_large"]

import pandas as pd
from transformers import (
    AutoConfig,
    MT5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
    set_seed,
)
import torch, random

set_seed(args.seed + random.randint(100, 1000))

# Define Parameter
model_checkpoint = args.model
model_name = model_basename(model_checkpoint)

if args.seed == 57706989:
    output_dir_hash = f"finetune/{args.task}/{model_name}/test/{args.language}"
else:
    output_dir_hash = f"finetune/{args.task}/{model_name}/{args.seed}/{args.language}"


batch_size = 32
if any(x in model_checkpoint.lower() for x in ["3b", "8b", "70b", "27b"]):
    batch_size //= 2

lr = args.lr if args.lr else get_learning_rate(model_checkpoint, args.task)
gradient_accumulation_steps = 64 // batch_size

intents_list = INTENTS
intents_l2id = {x: i for i, x in enumerate(intents_list)}
intents_id2l = dict(enumerate(intents_list))
intent_label_col_name = "intent"  # seq classification

slots_list = ["O"] + [s for s in SLOTS_MERGED]
slots_l2id = {x: i for i, x in enumerate(slots_list)}
slots_id2l = dict(enumerate(slots_list))
slot_label_col_name = "spans"  # token classification

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
        # finetuning_task="text-classification",
        return_unused_kwargs=True,
        trust_remote_code=True,
    )
    # config.problem_type = "single_label_classification"
    config.max_position_embeddings = 512

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
    config.max_position_embeddings = 512

tokenizer = T5Tokenizer.from_pretrained(
    model_checkpoint,
    use_fast=True,
    trust_remote_code=True,
    add_prefix_space=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"tokenizer.pad_token is None, set to: {tokenizer.eos_token}")

if config.pad_token_id is None:
    config.pad_token_id = tokenizer.pad_token_id
    print(f"config.pad_token_id is None, set to: {tokenizer.pad_token_id}")


def tokenc_preprocess(data: pd.DataFrame):
    from mlds.utils.converter import create_tokens_and_tags

    tokens_ll, tags_ll = [], []
    target_ll = []

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
            text=text, spans=to_dict(spans), format="IO"
        )
        tokens_ll.append(tokens)
        entities = []
        for entity in to_dict(spans):
            entities.append(
                f"{entity['labels'][0]}: {text[entity['start']:entity['end']]}"
            )

        tags_ll.append(entities)
        target_ll.append(" $$ ".join(entities))

    data["tokens"] = tokens_ll
    data["spans"] = tags_ll
    data["target"] = target_ll

    # print(data["tokens"][0], data["spans"][0], data["target"][0])
    tokenized_inputs = tokenizer(
        ["entity extraction: " + t for t in data["text"]],
        truncation=True,
        text_target=data["target"],
        max_length=config.max_position_embeddings,
    )

    labels = tokenizer(
        [t for t in data["target"]],
        # padding="longest",
        # return_tensors="pt",
    )["input_ids"]
    # labels[labels == 0] = -100
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def seqc_preprocess(data: pd.DataFrame):
    tokenized_inputs = tokenizer(
        ["intent recognition: " + t for t in data["text"]],
        truncation=True,
        text_target=data["intent"],
    )
    labels = tokenizer(
        ["<pad>" + t for t in data["intent"]],
        max_length=8,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"]
    labels[labels == 0] = -100
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


data_preprocess = {
    "tokenc": tokenc_preprocess,
    "seqc": seqc_preprocess,
    # "jointc": tokenc_preprocess,
}

from .trainer import get_datasets, Seq2SeqTrainer

tokenized_datasets = get_datasets(args.language).map(
    data_preprocess[args.task], batched=True
)

import random

trainer_args = Seq2SeqTrainingArguments(
    output_dir=f"/tmp/data/weight/{output_dir_hash}",
    run_name=f"{args.task}/{model_name}/{args.language}",
    overwrite_output_dir=True,
    seed=args.seed,
    eval_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=20,
    weight_decay=0.01,
    save_safetensors=True,
    save_strategy="epoch",
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

trainer_args = Seq2SeqTrainingArguments(
    output_dir=f"/tmp/data/weight/{output_dir_hash}",
    run_name=f"{args.task}/{model_name}/{args.language}",
    overwrite_output_dir=True,
    seed=args.seed,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size // 8,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=20,
    weight_decay=0.01,
    save_safetensors=True,
    save_strategy="epoch",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    # metric_for_best_model={"tokenc": "eval_f1", "seqc": "eval_accuracy"}[args.task],
    # greater_is_better=True,
    # load_best_model_at_end=True,
    local_rank=args.local_rank,
    ddp_find_unused_parameters=False,
)

from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
import evaluate
import numpy as np

early_stopping = EarlyStoppingCallback(early_stopping_patience=5)

def get_model(model_name, model_type):
    if model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(f"{model_name}")
        model = T5ForConditionalGeneration.from_pretrained(
            f"{model_name}", return_dict=True
        )
    elif model_type == "mt5":
        tokenizer = T5Tokenizer.from_pretrained(f"{model_name}")
        model = MT5ForConditionalGeneration.from_pretrained(
            f"{model_name}", return_dict=True
        )
    return model, tokenizer

print("CUDA available: ", torch.cuda.is_available())
print("CUDA device count: ", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")

MODEL_CLASS = {
    "t5": T5ForConditionalGeneration,
    "mt5": MT5ForConditionalGeneration,
}

if args.task == "seqc":
    if args.eval:
        model = MODEL_CLASS[config.model_type].from_pretrained(
            f"./data/weight/{output_dir_hash}",
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
            config=config,
        )
    else:
        model = MODEL_CLASS[config.model_type].from_pretrained(
            model_checkpoint,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
            config=config,
        )
        if lora_config:
            model = get_peft_model(model, lora_config)

    seq_metric = evaluate.load("accuracy", experiment_id=output_dir_hash.replace("/", "_"))

    def convert_to_intent(decoded_str, target_str):
        """
        Convert decoded and target strings to intent labels for metrics computation.

        Args:
            decoded_str (str): The decoded string in format "intent: value"
            target_str (str): The target string in format "intent: value"

        Returns:
            tuple: (decoded_intent, target_intent) where each is a string
        """

        def get_intent(value):
            value = value.strip()
            if value not in INTENTS:
                return None
            return value

        try:
            decoded_str = decoded_str.split(" ")[1]
        except:
            # print("Error", decoded_str)
            pass
        return get_intent(decoded_str), get_intent(target_str)

    def compute_metrics(p):
        preds, labels = p
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = preds.argmax(axis=-1)
        decoded_preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        true_labels = []
        pred_labels = []
        for output, target in zip(decoded_preds, decoded_labels):
            # print(output, "<=#=>", target)
            pred_label, true_label = convert_to_intent(output.strip(), target.strip())
            true_labels.append(true_label)
            pred_labels.append(pred_label)
        print(pred_labels[:5], true_labels[:5])

        pred_labels = [INTENTS.index(label) if label else -1 for label in pred_labels]
        true_labels = [INTENTS.index(label) if label else -2 for label in true_labels]
        # print(output[:5], decoded_str[:5], target_str[:5])
        print(pred_labels[:5], true_labels[:5])
        return seq_metric.compute(predictions=pred_labels, references=true_labels)


    trainer = Seq2SeqTrainer(
        model,
        trainer_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding="longest", pad_to_multiple_of=8),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # callbacks=[early_stopping],
    )

elif args.task == "tokenc":  # AutoModelForSeq2SeqLM
    if args.eval:
        model = MODEL_CLASS[config.model_type].from_pretrained(
            f"./data/weight/{output_dir_hash}",
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
            config=config,
        )
    else:
        model = MODEL_CLASS[config.model_type].from_pretrained(
            model_checkpoint,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
            config=config,
        )
        if lora_config:
            model = get_peft_model(model, lora_config)

    print("Padding Side:", tokenizer.padding_side)
    
    from transformers import pipeline
    from mlds.metrics import span_f1
    # print cude available: devices
    # from accelerate import infer_auto_device_map

    # device_map = infer_auto_device_map(model, max_memory={0: "0GiB", 1: "32GiB", 2: "32GiB", 3: "32GiB", "cpu": "60GiB"}, offload_buffers=True)

    def compute_metrics(p, **kargs):
        del p

        # evaluate the evaluation dataset
        if kargs.get("dataset", None) is None:
            dataset = tokenized_datasets[kargs.get("split", "validation")]
        else:
            dataset = kargs["dataset"]

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            # device=3,  # Use GPU 0 for the pipeline
            # device="cuda:{0}".format(args.local_rank) if args.local_rank != -1 else "cuda",
            device="cuda",
            device_map="balanced_low_0",
            # torch_dtype=torch.float16,
            # device_map=device_map,
            # ignore_labels=[-100],
            # aggregation_strategy="simple",
            max_new_tokens=250
        )

        # if args.local_rank == -1 or args.local_rank == 0:
        output = pipe(["entity extraction: " + t for t in dataset["text"]], batch_size=trainer_args.per_device_eval_batch_size)
        decoded_str = [x["generated_text"] for x in output]
        target_str = dataset["xtreme-up"]

        print(output[:5], decoded_str[:5], target_str[:5])
        return {"f1": span_f1(target_str, decoded_str) / 100}

        return {"f1": 0.0}

    trainer = Seq2SeqTrainer(
        model,
        trainer_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"], #  if do_eval else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # callbacks=[early_stopping],
    )
else:
    raise NotImplementedError


if not args.eval:
    print("!!!!!", args.language, "!!!!!")
    print("lr", trainer_args.to_dict()["learning_rate"])
    trainer.train(resume_from_checkpoint=args.eval)
    trainer.save_model(f"./data/weight/{output_dir_hash}/")
    os.system(f"rm -rf /tmp/data/weight/{output_dir_hash}")

    if args.lr != 0.0:
        if args.task == "seqc":
            validation_results = trainer.evaluate(
                tokenized_datasets["validation"], metric_key_prefix=""
            )
        elif args.task == "tokenc":
            validation_results = compute_metrics(
                None, dataset=tokenized_datasets["validation"]
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
for lan in LANGUAGES + ["eng"]:
    tokenized_test_dataset = get_datasets(lan)["test"].map(
        data_preprocess[args.task], batched=True
    )

    if args.task == "seqc":
        test_results = trainer.evaluate(
            tokenized_test_dataset, metric_key_prefix="test"
        )
        results[lan] = test_results
    elif args.task == "tokenc":
        results[lan] = compute_metrics(None, dataset=tokenized_test_dataset)

os.makedirs(os.path.dirname(f"./data/results/{output_dir_hash}.json"), exist_ok=True)

if args.local_rank == -1 or args.local_rank == 0:

    print(results)
    print(f"./data/results/{output_dir_hash}.json")
    with open(f"./data/results/{output_dir_hash}.json", "w") as f:
        json.dump(results, f)
    print(f"./data/results/{output_dir_hash}.json")
