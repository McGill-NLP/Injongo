from mlds.data_loader import LANGUAGES
from mlds.utils.path_utils import model_basename
# train: 2240 dev 320 test 640
# mT5/T5
import os

def status(task, model, seed, language):
    if seed is None:
        seed = "test"
    output_log = f"{task}/{model_basename(model)}/{seed}/{language}"
    if os.path.exists("./data/results/finetune/" + output_log + ".json"):
        return 1
    if os.path.exists("./data/weight/finetune/" + output_log + "/config.json"):
        # data/weight/finetune/seqc/mt5-large/test/eng/config.json
        return 2
    else:
        return 0

# 1 = done, -1 = need evaluate, 0 = not started
def get_task(seed=None):
    for task in ["seqc", "tokenc"]:
        for language in ["eng"] + LANGUAGES:
            for model in [
                "google/mt5-large",
                "castorini/afriteva_v2_large",
            ]:
                # for seed in range(2024, 2029):
                #     yield (f"python -m mlds.experiments.finetune_seq2seq {language} {task} {model} --seed {seed}")
                run_status = status(task, model, seed, language)

                if run_status == 0:
                    yield (f"python -m mlds.experiments.finetune_seq2seq {language} {task} {model}" + (f" --seed {seed}" if seed else ""))
                elif run_status == 2:
                    yield (f"python -m mlds.experiments.finetune_seq2seq {language} {task} {model} -e" + (f" --seed {seed}" if seed else ""))
                else:
                    print("# Skipping")
                    continue

    # train: 2240 dev 320 test 640
    # other models
    for task in ["tokenc", "seqc"]: # "joinc"
        for language in ["eng"] + LANGUAGES:
            for model in [
                "FacebookAI/xlm-roberta-large",
                "Davlan/afro-xlmr-large",
                "Davlan/afro-xlmr-large-76L",
                "castorini/afriberta_v2_large",
                'fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse',
                # "meta-llama/Llama-3.2-1B-Instruct",
                # "meta-llama/Llama-3.2-3B-Instruct",
                # "google/gemma-2-2b-it",
            ]:
                # yield (f"python -m mlds.experiments.finetune {language} {task} {model}" + (f" --seed {seed}" if seed else ""))
                run_status = status(task, model, seed, language)

                if run_status == 0:
                    yield (f"python -m mlds.experiments.finetune {language} {task} {model}" + (f" --seed {seed}" if seed else ""))
                elif run_status == 2:
                    yield (f"python -m mlds.experiments.finetune {language} {task} {model} -e" + (f" --seed {seed}" if seed else ""))
                else:
                    print("# Skipping")
                    continue


    # task = "seqc"
    # for language in LANGUAGES:
    #     language += "_eng"
    #     for model in [
    #         # "google/mt5-large",
    #         # "castorini/afriteva_v2_large",
    #     ]:
    #         # for seed in range(2024, 2029):
    #         #     yield (f"python -m mlds.experiments.finetune_seq2seq {language} {task} {model} --seed {seed}")
    #         run_status = status(task, model, seed, language)

    #         if run_status == 0:
    #             yield (f"python -m mlds.experiments.finetune_seq2seq {language} {task} {model}" + (f" --seed {seed}" if seed else ""))
    #         elif run_status == 2:
    #             yield (f"python -m mlds.experiments.finetune_seq2seq {language} {task} {model}" + (f" --seed {seed}" if seed else ""))
    #         else:
    #             print("# Skipping")
    #             continue

    # # train: 2240 dev 320 test 640
    # # other models
    # task = "seqc"
    # for language in LANGUAGES:
    #     language += "_eng"
    #     for model in [
    #         # "FacebookAI/xlm-roberta-large",
    #         # "Davlan/afro-xlmr-large",
    #         "Davlan/afro-xlmr-large-76L",
    #         # 'fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse',
    #     ]:
    #         # yield (f"python -m mlds.experiments.finetune {language} {task} {model}" + (f" --seed {seed}" if seed else ""))
    #         run_status = status(task, model, seed, language)

    #         if run_status == 0:
    #             yield (f"python -m mlds.experiments.finetune {language} {task} {model}" + (f" --seed {seed}" if seed else ""))
    #         elif run_status == 2:
    #             yield (f"python -m mlds.experiments.finetune {language} {task} {model}" + (f" --seed {seed}" if seed else ""))
    #         else:
    #             print("# Skipping")
    #             continue

def get_remaining_task():
    import json
    from collections import defaultdict
    import os

    save_dir = f"./data/results/finetune"
    # print("\t".join(["model", "task", "language", "accuracy", "f1", "loss", "epoch"]))


    from mlds.data_loader import LANGUAGES
    from mlds.utils.path_utils import model_basename

    for seed in [2024, 2025, 2026, 2027, 2028]:
        table = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda: defaultdict()))) # l1: model, l2: task
        for task in ["seqc", "tokenc"]:
            for source_language in LANGUAGES + ["eng"]: # ["eng"]:
                # T5
                for model in [
                    "google/mt5-large", # T5
                    "castorini/afriteva_v2_large", # T5
                    "FacebookAI/xlm-roberta-large",
                    "Davlan/afro-xlmr-large",
                    "Davlan/afro-xlmr-large-76L",
                    "castorini/afriberta_v2_large",
                    'fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse',
                    # "meta-llama/Llama-3.2-1B-Instruct",
                    # "meta-llama/Llama-3.2-3B-Instruct",
                    # "google/gemma-2-2b-it",
                ]:
                    try:
                        output_log = f"{save_dir}/{task}/{model_basename(model)}/{seed}/{source_language}.json"
                        if os.path.exists(output_log):
                            with open(output_log) as f:
                                data = json.load(f)
                            if task == "tokenc":
                                key = "test_f1"
                                if "f1" in data["eng"]:
                                    key = "f1"
                            else:
                                key = "test_accuracy"
                            for target_language in LANGUAGES + ["eng"]:
                                table[model][task][source_language][target_language] = data[target_language][key]

                            if table[model][task][source_language][source_language] < 0.65:
                                if model == "google/mt5-large" or model == "castorini/afriteva_v2_large":
                                    yield(f"python -m mlds.experiments.finetune_seq2seq {source_language} {task} {model} --seed {seed} # {table[model][task][source_language][source_language]}")
                                else:
                                    yield(f"python -m mlds.experiments.finetune {source_language} {task} {model} --seed {seed} # {table[model][task][source_language][source_language]}")
                        else:
                            if model == "google/mt5-large" or model == "castorini/afriteva_v2_large":
                                yield(f"python -m mlds.experiments.finetune_seq2seq {source_language} {task} {model} --seed {seed} # Missing")
                            else:
                            # print(f"Missing {output_log}")
                                yield(f"python -m mlds.experiments.finetune {source_language} {task} {model} --seed {seed} # Missing")
                    except Exception as e:
                        print(f"Error: {e}")
                        print(f"Error: {output_log}")
                        pass

if __name__ == "__main__":
    for task in get_task():
        print(task)

    for task in get_remaining_task():
        print(task)