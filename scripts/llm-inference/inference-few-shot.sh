script=mlds.experiments.inference_llm
best_intent_round=2
best_slot_round=3

for lang in eng amh ewe hau ibo kin lin lug orm sna sot swa twi wol xho yor zul
do
    python -m $script $lang -t 10 --template "intent-detection-five-round" --round $best_intent_round --openai-model gpt-4o-mini-2024-07-18 --shot_count 40
    python -m $script $lang -t 15 --template "slot-filling-five-round" --round $best_slot_round --openai-model gpt-4o-mini-2024-07-18 --shot_count 23
    python -m $script $lang -t 10 --template "intent-detection-five-round" --round $best_intent_round --openai-model gpt-4o-mini-2024-07-18 --shot_count 10
    python -m $script $lang -t 15 --template "slot-filling-five-round" --round $best_slot_round  --openai-model gpt-4o-mini-2024-07-18 --shot_count 10
done


google_base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
for lang in eng amh ewe hau ibo kin lin lug orm sna sot swa twi wol xho yor zul
do
    python -m $script $lang -t 10 --template "intent-detection-five-round" --round $best_intent_round --api_base $google_base_url --shot_count 40
    python -m $script $lang -t 15 --template "slot-filling-five-round" --round $best_slot_round --api_base $google_base_url --shot_count 23
    python -m $script $lang -t 10 --template "intent-detection-five-round" --round $best_intent_round --api_base $google_base_url --shot_count 10
    python -m $script $lang -t 15 --template "slot-filling-five-round" --round $best_slot_round --api_base $google_base_url --shot_count 10
done
