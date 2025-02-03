import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

from mlds.experiments.prompt_manager import PromptManager
from mlds.data_loader import (
    INTENTS,
    LANGUAGES,
    LANGUAGES_MAPPER,
    SCHEMA,
    SLOTS_MERGED,
    SlotDataManager,
)
from mlds.utils.path_utils import model_basename
from tenacity import retry, stop_after_attempt, wait_random_exponential
from mlds.experiments.prompt import PROMPTS

API_MODEL = None


complete_model = {
    "Lugha-Llama/Lugha-Llama-8B-wura_edu"
}
def main():
    load_dotenv()
    
    print(os.getenv("OPENAI_ORGANIZATION"))

    args = parse_arguments()

    pm = init_prompt_manager(args)

    # Initialize OpenAI client
    if "sk-proj" not in os.getenv("OPENAI_API_KEY"):
        client = OpenAI(
            base_url=args.api_base,
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORGANIZATION"),
            timeout=60,
        )
    else:
        client = OpenAI(base_url=args.api_base, api_key=args.api_key, timeout=60,)

    sm = SlotDataManager("./data/output")
    df = sm.load_data(args.language, split=args.split)
    df.index.name = "id"

    print(f"{args.language} Loaded {len(df)} rows of data")
    print(df.head())

    # Few Shot
    import random
    random.seed(2025)
    shuffled_intents = [i for i in INTENTS]
    random.shuffle(shuffled_intents)
    example_data = []
    print(shuffled_intents)
    train_df = sm.load_data(args.language, split="train")

    # seqc: shot_count = 40 intent
    # seqc: shot_count = 10 domain (5 shots)
    # slot: shot_count = 23 slot type
    # shot_count = 5 random
    train_df.fillna("", inplace=True)
    train_df = train_df[train_df["spans"] != ""]

    if args.shot_count == 40:
        for intent in INTENTS:
            line = train_df[train_df["intent"] == intent]
            line = line[line["text"] != ""]
            line = line.iloc[0]
            example_data.append({"intent": line["intent"], "text": line["text"], "slot": line["xtreme-up"]})
    elif args.shot_count == 10:
        print(len(train_df["domain"].unique()), train_df["domain"].unique())
        for domain in train_df["domain"].unique():
            line = train_df[train_df["domain"] == domain]
            line = line[line["text"] != ""]
            line = line.iloc[0]
            example_data.append({"intent": line["intent"], "text": line["text"], "slot": line["xtreme-up"]})
    elif args.shot_count == 23:
        # select first entity covers 23 slot types
        existed_slots = set()
        for _, line in train_df.iterrows():
            # 17:27:SL:MONEY,36:51:SL:BANK_NAME
            if isinstance(line["spans"], float):
                continue
            slots = [slot.split(":")[-1] for slot in line["spans"].split(",")]
            for slot in slots:
                if slot not in existed_slots:
                    example_data.append({"intent": line["intent"], "text": line["text"], "slot": line["xtreme-up"]})
                    existed_slots |= set(slots)
                    break
            if len(existed_slots) >= 23:
                break
    else:
        for intent in shuffled_intents[:args.shot_count]:
            line = train_df[train_df["intent"] == intent]
            line = line[line["xtreme-up"] != ""]
            line = line.iloc[0]
            example_data.append({"intent": line["intent"], "text": line["text"], "slot": line["xtreme-up"]})
    # print(example_data)
    pm.add_language_example(LANGUAGES_MAPPER[args.language], example_data)

    # Process data
    if args.round >= 0:
        output_file = (
            args.output_file
            or f"data/results/prompt/{model_basename(args.model)}/{args.template}/round_{args.round}/{args.language}.csv"
        )
    else:
        output_file = (
            args.output_file
            or f"data/results/prompt/{model_basename(args.model)}/{args.template}/{args.language}.csv"
        )

    if os.path.exists(output_file) and not args.override:
        print(f"Output file {output_file} already exists. Exiting...")
        return

    assert API_MODEL is not None

    raw_outputs, processed_outputs = process_data(
        df,
        pm,
        client,
        args.num_threads,
        template_name=args.template,
        model_name=API_MODEL if not args.openai_model else args.openai_model,
        shot_count=args.shot_count,
        language=args.language,
        generation_params={
            "max_tokens": 300,
            "temperature": 0.5,
            # "stop": ["\n"],
        },
        guide=args.guide,
        round=args.round,
    )

    # Add results to DataFrame
    df["raw"] = raw_outputs
    # df["processed"] = processed_outputs

    # Save results
    folder = os.path.dirname(output_file)
    os.makedirs(folder, exist_ok=True)
    df.to_csv(output_file, index=False)

    # print(evaluate(df, gt_col="logical_form", pred_col="processed")) # Not working


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=str, choices=LANGUAGES + ["eng"], help="Language code")
    parser.add_argument("--output_file", "-o", help="Path to the output CSV file")
    parser.add_argument(
        "--num_threads", "-t", type=int, default=1, help="Number of threads to use"
    )
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument(
        "--template", type=str, required=True, help="Template name for prompt"
    )
    parser.add_argument(
        "--api_base", type=str, default="http://localhost:8000/v1", help="API base URL"
    )
    parser.add_argument(
        "--shot_count",
        type=int,
        default=0,
        help="Number of shots for few-shot learning",
    )
    parser.add_argument(
        "--guide",
        action="store_true",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to use (train, val, test)",
    )
    parser.add_argument(
        "--override", action="store_true", help="Override existing output file"
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--openai-model", type=str, default="")
    parser.add_argument("--round", type=int, default=-1)

    args = parser.parse_args()

    if "localhost" in args.api_base and not args.openai_model:
        global API_MODEL
        API_MODEL = requests.get(f"{args.api_base}/models").json()["data"][0]["root"]
        if not args.model:
            args.model = API_MODEL
        # args.model = args.model
        args.api_key = "test"
        print(f"Using model: {args.model}")
    elif "google" in args.api_base:
        API_MODEL = "gemini-1.5-pro-002"
        args.model = "gemini-1.5-pro-002"
        args.api_key = os.getenv("GEMINI_API_KEY")
        print(f"Using Google Gemini Model: {args.model}")
    else:
        args.model = args.openai_model
        API_MODEL = args.model
        args.api_key = os.getenv("OPENAI_API_KEY")
        args.api_base = "https://api.openai.com/v1"
        print(f"Using OpenAI API Model: {args.openai_model}")

    if args.shot_count >= 1:
        args.template += f"-{args.shot_count}shot"

    return args


def init_prompt_manager(args):
    pm = PromptManager()
    pm.add_template(
        "slot-filling-five-round" + (f"-{args.shot_count}shot" if args.shot_count >= 1 else ""),
        PROMPTS["slot-filling-five-round"])

    pm.add_template(
        "intent-detection-five-round" + (f"-{args.shot_count}shot" if args.shot_count >= 1 else ""),
        PROMPTS["intent-detection-five-round"]
        # """Classify the sentence by intent, selecting from the available categories. Only return the chosen intent category without additional commentary or formatting.\n\n# Intent Categories\n"""
    )
    # pm.save(f"data/prompts/{args.language}_{int(time.time())%10000000//60}.json")
    return pm


def process_row(
    idx: int, row: pd.Series, prompt_manager: PromptManager, client: OpenAI, **kwargs
) -> Tuple[str, str, str]:
    """
    Process a single row of data using the PromptManager and OpenAI client.

    :param row: A pandas Series representing a row of data
    :param prompt_manager: An instance of PromptManager
    :param client: An instance of OpenAI client
    :param kwargs: Additional arguments including template_name, model_name, etc.
    :return: A tuple containing (id, raw_output, processed_output)
    """
    id_field = kwargs.get("id_field", "id")
    template_name = kwargs["template_name"]
    model_name = kwargs["model_name"]
    shot_count = kwargs.get("shot_count", 0)
    language = kwargs.get("language", "eng")
    generation_params = kwargs.get("generation_params", {})
    round = kwargs.get("round", -1)

    # Generate prompt using PromptManager
    prompt = prompt_manager.generate_prompt(
        template_name=template_name,
        model_name=model_name,
        shot_count=shot_count,
        language=LANGUAGES_MAPPER[language],
        text=row["text"].replace("\n", " ").replace('""', '"').strip(),
        round=round,
        # question=row.get("question", ""),  # Add other fields as necessary
    )

    if idx == 0:
        print(prompt)

    # Use OpenAI client to get response
    if kwargs.get("guide"):
        response = requests.post(
            f"{client.base_url}/completions",
            json={
                "prompt": prompt,
                "schema": SCHEMA,
                **generation_params,
            },
        )
        raw_output = response.json()["text"][0][len(prompt) :]
        processed_output = prompt_manager.process_output(raw_output, "raw")

    else:
        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def chatcompletion_with_backoff(**kwargs):
            return client.chat.completions.create(**kwargs)

        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def completion_with_backoff(**kwargs):
            return client.completions.create(**kwargs)

        try:
            if model_name in complete_model:
                response = completion_with_backoff(
                    model=model_name,
                    prompt=prompt,
                    **generation_params,
                )
                raw_output = response.choices[0].text
            else:
                response = chatcompletion_with_backoff(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    n=1,
                    **generation_params
                )
                raw_output = response.choices[0].message.content

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            raw_output = ""
            exit(1)

        if not raw_output:
            raw_output = ""
            print(f"Empty response for row {idx}")
        # Process the output if needed
        processed_output = prompt_manager.process_output(raw_output, "raw")
        print("====>", processed_output , "<====")
    # row[id_field]
    # print(idx, row)
    return idx, processed_output, processed_output


def process_data(
    df: pd.DataFrame,
    prompt_manager: PromptManager,
    client: OpenAI,
    num_threads: int,
    **kwargs,
) -> Tuple[List[str], List[str]]:
    """
    Process the entire dataset using single or multi-threading.

    :param df: The input DataFrame
    :param prompt_manager: An instance of PromptManager
    :param client: An instance of OpenAI client
    :param num_threads: Number of threads to use (1 for single-threaded)
    :param kwargs: Additional arguments
    :return: Two lists containing raw_outputs and processed_outputs
    """
    ll = len(df)
    ids = list(range(ll))
    df.index = ids

    process_func = lambda row: process_row(
        row[0], row[1], prompt_manager, client, **kwargs
    )

    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_func, row) for row in df.iterrows()]
            results = [
                future.result() for future in tqdm(as_completed(futures), total=len(df))
            ]
    else:
        results = [process_func(row) for row in tqdm(df.iterrows(), total=len(df))]

    # Sort results based on original order
    sorted_results = sorted(results, key=lambda x: (x[0]))
    raw_outputs = [result[1] for result in sorted_results]
    processed_outputs = [result[2] for result in sorted_results]

    return raw_outputs, processed_outputs


if __name__ == "__main__":
    main()
