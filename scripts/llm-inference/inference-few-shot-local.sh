script=mlds.experiments.inference_llm
best_intent_round=2
best_slot_round=3
for lang in eng amh ewe hau ibo kin lin lug orm sna sot swa twi wol xho yor zul
do
    python -m $script $lang -t 50 --template "intent-detection-five-round" --round $best_intent_round --shot_count 40
    python -m $script $lang -t 50 --template "slot-filling-five-round" --round $best_slot_round --shot_count 23
    python -m $script $lang -t 50 --template "intent-detection-five-round" --round $best_intent_round --shot_count 10
    python -m $script $lang -t 50 --template "slot-filling-five-round" --round $best_slot_round --shot_count 10
done
