# Example
# python -m mlds.experiments.finetune eng seqc Davlan/afro-xlmr-large-76L --seed 2024

python -m mlds.experiments.finetune eng+1shot tokenc Davlan/afro-xlmr-large-76L --seed 2024
python -m mlds.experiments.finetune eng+1shot tokenc Davlan/afro-xlmr-large-76L --seed 2025
python -m mlds.experiments.finetune eng+1shot tokenc Davlan/afro-xlmr-large-76L --seed 2026
python -m mlds.experiments.finetune eng+1shot tokenc Davlan/afro-xlmr-large-76L --seed 2027
python -m mlds.experiments.finetune eng+1shot tokenc Davlan/afro-xlmr-large-76L --seed 2028

python -m mlds.experiments.finetune eng+1shot seqc Davlan/afro-xlmr-large-76L --seed 2024
python -m mlds.experiments.finetune eng+1shot seqc Davlan/afro-xlmr-large-76L --seed 2025
python -m mlds.experiments.finetune eng+1shot seqc Davlan/afro-xlmr-large-76L --seed 2026
python -m mlds.experiments.finetune eng+1shot seqc Davlan/afro-xlmr-large-76L --seed 2027
python -m mlds.experiments.finetune eng+1shot seqc Davlan/afro-xlmr-large-76L --seed 2028


for lang in eng clinc eng+clinc clinc+extend clinc_5shots clinc_10shots clinc_25shots clinc_50shots clinc_100shots eng_5shots eng_10shots eng_25shots
do
    for seed in 2024 2025 2026 2027 2028
    do
        for model in Davlan/afro-xlmr-large-76L
        do
            echo "python -m mlds.experiments.finetune $lang seqc $model --seed $seed"
        done
    done
done

