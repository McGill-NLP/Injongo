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
def get_task(*args):
    for seed in range(2024, 2029):
        for task in ["seqc", "tokenc"]:
            for language in ["all"]:
                for model in [
                    # "google/mt5-large",
                    "castorini/afriteva_v2_large",
                ]:
                    # for seed in range(2024, 2029):
                    #     yield (f"python -m mlds.experiments.finetune_seq2seq {language} {task} {model} --seed {seed}")
                    run_status = status(task, model, seed, language)

                    if run_status == 0:
                        yield (f"python -m mlds.experiments.finetune_seq2seq {language} {task} {model}" + (f" --seed {seed}" if seed else ""))
                    elif run_status == 2:
                        yield (f"python -m mlds.experiments.finetune_seq2seq {language} {task} {model}" + (f" --seed {seed}" if seed else ""))
                    else:
                        print("Skipping")
                        continue

        # train: 2240 dev 320 test 640
        # other models
        for task in ["tokenc", "seqc"]: # "joinc"
            for language in ["all"]:
                for model in [
                    # "FacebookAI/xlm-roberta-large",
                    "Davlan/afro-xlmr-large",
                    "Davlan/afro-xlmr-large-76L",
                    'fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse',
                ]:
                    # yield (f"python -m mlds.experiments.finetune {language} {task} {model}" + (f" --seed {seed}" if seed else ""))
                    run_status = status(task, model, seed, language)

                    if run_status == 0:
                        yield (f"python -m mlds.experiments.finetune {language} {task} {model}" + (f" --seed {seed}" if seed else ""))
                    elif run_status == 2:
                        yield (f"python -m mlds.experiments.finetune {language} {task} {model}" + (f" --seed {seed}" if seed else ""))
                    else:
                        print("# Skipping")
                        continue

if __name__ == "__main__":
    for task in get_task():
        print(task)