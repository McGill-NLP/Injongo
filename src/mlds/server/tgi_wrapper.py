from flask import Flask, jsonify, request

# https://github.com/js8544/vllm/tree/enc_dec_t5

app = Flask(__name__)

tokenizer = None
model = None
source_port = None

import requests

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    messages = data.get("messages", [])

    # Combine all messages into a single prompt
    prompt = "\n".join([f"{m['content']}" for m in messages]) #{m['role']}:     
    
    r = requests.post(f"http://localhost:{source_port}/v1/completions", json={"model": model_name, "prompt": prompt})
    response_text = r.json()["choices"][0]["text"]
    print(response_text)
    new_text = response_text

    return jsonify(
        {
            "id": "chatcmpl-mock-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": new_text.strip()},
                    "finish_reason": "stop",
                }
            ],
        }
    )


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify(
        {
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "mock-org",
                    "root": model_name,
                }
            ]
        }
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="CohereForAI/aya-101")
    parser.add_argument("--src-port", type=int, default=8001)
    parser.add_argument("--dest-port", type=int, default=8000)
    args = parser.parse_args()
    model_name = args.model
    source_port = args.src_port
    app.run(port=args.dest_port)

# python src/mlds/server/tgi_wrapper.py --src-port 8001 --dest-port 8000