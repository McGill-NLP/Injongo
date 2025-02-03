#! /bin/bash

#SBATCH --output=./task/log/%j.out
#SBATCH --error=./task/log/%j.err
#SBATCH --job-name=multilingual

exit_script() {
    echo "Preemption signal, saving myself"
    trap - SIGTERM # clear the trap
    # Optional: sends SIGTERM to child/sub processes
    kill -- -$$
}

trap exit_script SIGTERM

tasks=(
    "torchrun --nproc_per_node=$2 -m mlds.experiments.finetune all tokenc fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse --seed 2024"
    "torchrun --nproc_per_node=$2 -m mlds.experiments.finetune all tokenc fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse --seed 2025"
    "torchrun --nproc_per_node=$2 -m mlds.experiments.finetune all tokenc fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse --seed 2026"
    "torchrun --nproc_per_node=$2 -m mlds.experiments.finetune all tokenc fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse --seed 2027"
    "torchrun --nproc_per_node=$2 -m mlds.experiments.finetune all tokenc fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse --seed 2028"
    "torchrun --nproc_per_node=$2 --master_port 9001 -m mlds.experiments.finetune all seqc fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse --seed 2024"
    "torchrun --nproc_per_node=$2 --master_port 9002 -m mlds.experiments.finetune all seqc fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse --seed 2025"
    "torchrun --nproc_per_node=$2 --master_port 9003 -m mlds.experiments.finetune all seqc fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse --seed 2026"
    "torchrun --nproc_per_node=$2 --master_port 9004 -m mlds.experiments.finetune all seqc fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse --seed 2027"
    "torchrun --nproc_per_node=$2 --master_port 9005 -m mlds.experiments.finetune all seqc fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse --seed 2028"
    "llamafactory-cli train task/config/llama3_tokenc_sft.yaml"
    "llamafactory-cli train task/config/llama3_seqc_sft.yaml"
    "llamafactory-cli train task/config/llama3_tokenc_sft_3b.yaml"
    "llamafactory-cli train task/config/llama3_seqc_sft_3b.yaml"
    "llamafactory-cli train task/config/gemma_seqc_sft_9b.yaml"
    "llamafactory-cli train task/config/gemma_tokenc_sft_9b.yaml"
)

# task $1 th task
# take the task from the array
task=${tasks[$1]}
echo "Running $task"
export FORCE_TORCHRUN=1
$task
wait
echo "Finished $task"

# python -m mlds.experiments.finetune all seqc fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse --seed 2028
# torchrun --nproc_per_node=4 -m mlds.experiments.finetune all seqc fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse --seed 2028