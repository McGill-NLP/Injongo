script=mlds.experiments.inference_llm
for round in 1 2 3 4 5
do
    for lang in eng amh ewe hau ibo kin lin lug orm sna sot swa twi wol xho yor zul
    do
        python -m $script $lang -t 100 --template "slot-filling-five-round" --round $round --api_base http://localhost:8000/v1
        python -m $script $lang -t 100 --template "intent-detection-five-round" --round $round --api_base http://localhost:8000/v1
    done
done
