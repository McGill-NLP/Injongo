
def get_task():# find the best model for each model
    logfile = "data/results/finetune/log-v3.jsonl"
    import json
    from collections import defaultdict

    table = defaultdict(lambda: defaultdict(dict))
    for line in open(logfile):
        data = json.loads(line)
        model = data["model"]
        task = data["task"]
        lr = data["lr"]

        col = "accuray"
        for key in data:
            if "f1" in key:
                col = key
                break
        else:
            for key in data:
                if "accuracy" in key:
                    col = key
                    break

        table[model][task][lr] = data[col]

    def table_get(model, task, lr):
        try:
            model = model.split("/")[-1]
            return table[model][task][lr]
        except KeyError:
            return -1

    for task in ["seqc", "tokenc"]:
        for language in ["swa"]:
            # T5
            for model in [
                "google/mt5-large", # T5
                "castorini/afriteva_v2_large", # T5
            ]:
                for lr in [1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]:
                    # if 0.0 <= table_get(model, task, lr) <= 0.1:
                    #     break
                    if table_get(model, task, lr) == -1:
                        yield (f"python -m mlds.experiments.finetune_seq2seq {language} {task} {model} --lr {lr}")

            for model in [
                "FacebookAI/xlm-roberta-large",
                "Davlan/afro-xlmr-large",
                "Davlan/afro-xlmr-large-76L",
                "castorini/afriberta_v2_large",
                'fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse',
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "google/gemma-2-2b-it",
            ]:
                for lr in [1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4]:
                    # if 0.0 <= table_get(model, task, lr) <= 0.1 and task == "seqc":
                    #     break
                    if table_get(model, task, lr) == -1:
                        yield (f"python -m mlds.experiments.finetune {language} {task} {model} --lr {lr}")

    print("Current Status:")
    # find the best model for each model
    padding = lambda x: x + " " * (30 - len(x))
    lrs = [1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 2.5e-4, 3e-4, 5e-4, 7.5e-4, 1e-3]
    lrs = [1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4]
    print("\t".join([padding("model"), "task"] + [f"{x:.5f}" for x in lrs]))
    for task in ["seqc", "tokenc"]:
        print("\n".join(["\t".join(map(str, [padding(model), task] + [round(table[model][task].get(lr, -1), 5) for lr in lrs])) for model in sorted(table)]))
        print()
        print()

if __name__ == "__main__":
    for x in get_task():
        print(x)