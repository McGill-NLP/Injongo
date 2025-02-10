SINGULARITYENV_HF_HOME="~/.cache/huggingface" SINGULARITYENV_CUDA_VISIBLE_DEVICES=0,1 singularity run --nv --bind ~/.cache/huggingface:~/.cache/huggingface docker://ghcr.io/huggingface/text-generation-inference:2.4.0 --model-id CohereForAI/aya-101 --port 8001

python src/mlds/server/tgi_wrapper.py --src-port 8001 --dest-port 8000

curl -X POST "http://localhost:9000/v1/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "CohereForAI/aya-101",
		"prompt": "What is the capital of France?"
	}'

singularity run --nv tensorflow_latest-gpu.sif