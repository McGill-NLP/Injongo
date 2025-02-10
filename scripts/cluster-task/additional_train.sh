# clinc

python -m mlds.experiments.finetune clinc seqc Davlan/afro-xlmr-large-76L
python -m mlds.experiments.finetune clinc seqc Davlan/afro-xlmr-large-76L --seed 2024
python -m mlds.experiments.finetune clinc seqc Davlan/afro-xlmr-large-76L --seed 2025
python -m mlds.experiments.finetune clinc seqc Davlan/afro-xlmr-large-76L --seed 2026
python -m mlds.experiments.finetune clinc seqc Davlan/afro-xlmr-large-76L --seed 2027
python -m mlds.experiments.finetune clinc seqc Davlan/afro-xlmr-large-76L --seed 2028

python -m mlds.experiments.finetune eng+clinc seqc Davlan/afro-xlmr-large-76L --seed 2024
python -m mlds.experiments.finetune eng+clinc seqc Davlan/afro-xlmr-large-76L --seed 2025
python -m mlds.experiments.finetune eng+clinc seqc Davlan/afro-xlmr-large-76L --seed 2026
python -m mlds.experiments.finetune eng+clinc seqc Davlan/afro-xlmr-large-76L --seed 2027
python -m mlds.experiments.finetune eng+clinc seqc Davlan/afro-xlmr-large-76L --seed 2028


python -m mlds.experiments.finetune clinc+extend seqc Davlan/afro-xlmr-large-76L --seed 2024
python -m mlds.experiments.finetune clinc+extend seqc Davlan/afro-xlmr-large-76L --seed 2025
python -m mlds.experiments.finetune clinc+extend seqc Davlan/afro-xlmr-large-76L --seed 2026
python -m mlds.experiments.finetune clinc+extend seqc Davlan/afro-xlmr-large-76L --seed 2027
python -m mlds.experiments.finetune clinc+extend seqc Davlan/afro-xlmr-large-76L --seed 2028

for lang in eng clinc eng+clinc clinc+extend
do
    for seed in 2024 2025 2026 2027 2028
    #    1047 || 1040 || 1040 clinc + 1047 eng || 2080
    do
        for model in Davlan/afro-xlmr-large-76L
        do
            echo "python -m mlds.experiments.finetune $lang seqc $model --seed $seed -e"
        done
    done
done

python -m mlds.experiments.finetune eng seqc Davlan/afro-xlmr-large-76L --seed 2024 -e
python -m mlds.experiments.finetune eng seqc Davlan/afro-xlmr-large-76L --seed 2025 -e
python -m mlds.experiments.finetune eng seqc Davlan/afro-xlmr-large-76L --seed 2026 -e
python -m mlds.experiments.finetune eng seqc Davlan/afro-xlmr-large-76L --seed 2027 -e
python -m mlds.experiments.finetune eng seqc Davlan/afro-xlmr-large-76L --seed 2028 -e
python -m mlds.experiments.finetune clinc seqc Davlan/afro-xlmr-large-76L --seed 2024 -e
python -m mlds.experiments.finetune clinc seqc Davlan/afro-xlmr-large-76L --seed 2025 -e
python -m mlds.experiments.finetune clinc seqc Davlan/afro-xlmr-large-76L --seed 2026 -e
python -m mlds.experiments.finetune clinc seqc Davlan/afro-xlmr-large-76L --seed 2027 -e
python -m mlds.experiments.finetune clinc seqc Davlan/afro-xlmr-large-76L --seed 2028 -e
python -m mlds.experiments.finetune eng+clinc seqc Davlan/afro-xlmr-large-76L --seed 2024 -e
python -m mlds.experiments.finetune eng+clinc seqc Davlan/afro-xlmr-large-76L --seed 2025 -e
python -m mlds.experiments.finetune eng+clinc seqc Davlan/afro-xlmr-large-76L --seed 2026 -e
python -m mlds.experiments.finetune eng+clinc seqc Davlan/afro-xlmr-large-76L --seed 2027 -e
python -m mlds.experiments.finetune eng+clinc seqc Davlan/afro-xlmr-large-76L --seed 2028 -e
python -m mlds.experiments.finetune clinc+extend seqc Davlan/afro-xlmr-large-76L --seed 2024 -e
python -m mlds.experiments.finetune clinc+extend seqc Davlan/afro-xlmr-large-76L --seed 2025 -e
python -m mlds.experiments.finetune clinc+extend seqc Davlan/afro-xlmr-large-76L --seed 2026 -e
python -m mlds.experiments.finetune clinc+extend seqc Davlan/afro-xlmr-large-76L --seed 2027 -e
python -m mlds.experiments.finetune clinc+extend seqc Davlan/afro-xlmr-large-76L --seed 2028 -e


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

for seed in 2024 2025 2026 2027 2028
do
    python -m mlds.experiments.finetune clinc_5shots seqc Davlan/afro-xlmr-large-76L --seed $seed -e
done

for seed in 2024 2025 2026 2027 2028
do
    python -m mlds.experiments.finetune clinc_10shots seqc Davlan/afro-xlmr-large-76L --seed $seed -e
done

for seed in 2024 2025 2026 2027 2028
do
    python -m mlds.experiments.finetune clinc_25shots seqc Davlan/afro-xlmr-large-76L --seed $seed -e
done

for seed in 2024 2025 2026 2027 2028
do
    python -m mlds.experiments.finetune clinc_50shots seqc Davlan/afro-xlmr-large-76L --seed $seed -e
done

for seed in 2024 2025 2026 2027 2028
do
    python -m mlds.experiments.finetune clinc_100shots seqc Davlan/afro-xlmr-large-76L --seed $seed -e
done

for seed in 2024 2025 2026 2027 2028
do
    python -m mlds.experiments.finetune eng_5shots seqc Davlan/afro-xlmr-large-76L --seed $seed -e
done

for seed in 2024 2025 2026 2027 2028
do
    python -m mlds.experiments.finetune eng_10shots seqc Davlan/afro-xlmr-large-76L --seed $seed -e
done

for seed in 2024 2025 2026 2027 2028
do
    python -m mlds.experiments.finetune eng_25shots seqc Davlan/afro-xlmr-large-76L --seed $seed -e
done
