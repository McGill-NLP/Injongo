import argparse
from functools import partial
import os
from tasks import lr, run, run_all

task_mapper = {
    "lr": lr.get_task,
    "test": run.get_task,
    "seed1": partial(run.get_task, seed=2024),
    "seed2": partial(run.get_task, seed=2025),
    "seed3": partial(run.get_task, seed=2026),
    "seed4": partial(run.get_task, seed=2027),
    "seed5": partial(run.get_task, seed=2028),
    "rerun": run.get_remaining_task,
    "all": run_all.get_task,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("ngpus", type=int)
    args = parser.parse_args()
    task_list = list(task_mapper[args.task]())

    for gpu in range(args.ngpus):
        # clean the file
        with open(f"./task/GPU_{gpu}.sh", "w") as f:
            # f.write(f"export CUDA_VISIBLE_DEVICES={gpu}\n")
            f.write("# Run the task with GPU {gpu}\n")
    for index in range(0, len(task_list)):
        with open(f"./task/GPU_{index % args.ngpus}.sh", "a") as f:
            f.write(f"CUDA_VISIBLE_DEVICES={index % args.ngpus} {task_list[index]}\n")

    # give permission
    for gpu in range(args.ngpus):
        os.system(f"chmod +x ./task/GPU_{gpu}.sh")

if __name__ == "__main__":
    main()