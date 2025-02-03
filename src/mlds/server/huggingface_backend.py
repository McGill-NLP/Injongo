from flask import Flask, jsonify, request
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# https://github.com/js8544/vllm/tree/enc_dec_t5

app = Flask(__name__)

tokenizer = None
model = None


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 500)

    # Combine all messages into a single prompt
    prompt = "\n".join([f"{m['content']}" for m in messages])
    # {m['role']}:
    # Tokenize and generate
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    output = model.generate(input_ids, max_new_tokens=max_tokens)
    # Decode the output
    response_text = tokenizer.decode(output[0].cpu(), skip_special_tokens=True)

    # Extract only the newly generated text
    new_text = response_text  # [len(prompt):]
    print(new_text, len(new_text))

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
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    app.run(port=args.port)
