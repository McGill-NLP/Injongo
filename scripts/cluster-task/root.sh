#! /bin/bash

# SBATCH --output=./task/log/%j.out
# SBATCH --error=./task/log/%j.err
# SBATCH --job-name=multilingual

exit_script() {
    echo "Preemption signal, saving myself"
    trap - SIGTERM # clear the trap
    # Optional: sends SIGTERM to child/sub processes
    kill -- -$$
}

trap exit_script SIGTERM

#  if first parameter is not empty, then use it as task
rm -f ./task/GPU_*.sh

python ./task/create.py $1 $2

for task in ./task/GPU_*.sh; do
    echo "Running $task"
    save_path=$(basename $task .sh)_$(date +%s).log
    cat $task | tee $save_path
    # srun -l --overlap --output=$save_path 
    bash $task >> $save_path &
done

wait