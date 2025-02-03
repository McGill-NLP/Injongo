vllm serve "google/gemma-2-27b-it" -tp 2
vllm serve "meta-llama/Llama-3.3-70B-Instruct" -tp 4 --gpu_memory_utilization 0.85 --max_model_len 4096