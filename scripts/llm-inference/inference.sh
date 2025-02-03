script=mlds.experiments.inference_llm
for round in 1 2 3 4 5
do
    for lang in eng amh ewe hau ibo kin lin lug orm sna sot swa twi wol xho yor zul
    do
        python -m $script $lang -t 30 --template "slot-filling-five-round" --openai-model gpt-4o-2024-08-06 --round $round
        python -m $script $lang -t 20 --template "slot-filling-five-round" --openai-model gpt-4o-mini-2024-07-18 --round $round
        python -m $script $lang -t 30 --template "intent-detection-five-round" --openai-model gpt-4o-2024-08-06 --round $round
        python -m $script $lang -t 20 --template "intent-detection-five-round" --openai-model gpt-4o-mini-2024-07-18 --round $round
    done
done


google_base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
for round in 1 2 3 4 5
do
    for lang in eng amh ewe hau ibo kin lin lug orm sna sot swa twi wol xho yor zul
    do
        python -m $script $lang -t 10 --template "slot-filling-five-round" --round $round --api_base $google_base_url
        python -m $script $lang -t 10 --template "intent-detection-five-round" --round $round --api_base $google_base_url
    done
done
