#!/bin/bash

export PYTHONPATH=/root/autodl-tmp/acdc
source /etc/network_turbo

# Define parameters
RESET_NETWORKS=(0 1)
ZERO_ABLATIONS=(0 1)
LOSS_TYPES=("kl_div")
THRESHOLDS=(0.01 0.0178 0.0316 0.0562 0.1 0.1778 0.3162 0.5623 1.0 1.7783 3.1623)  # Corresponds to 10^np.linspace(-2, 0.5, 11)
SEED=424671755
DEVICE="cuda" 
CPU=6  # Number of PyTorch threads
WANDB_MODE="online"  # WandB mode, can be set to "offline" or "online"

# Create log directory
LOG_DIR="logs"

# Clear log directory and its subdirectories
if [ -d "$LOG_DIR" ]; then
    rm -rf "$LOG_DIR"
fi
mkdir -p "$LOG_DIR"

# Set the maximum number of parallel tasks
MAX_PARALLEL_JOBS=3  # Adjust the number of parallel tasks as needed
SEMAPHORE="/tmp/semaphore"  # Used to control the number of parallel tasks

# If the semaphore already exists, delete it first
if [ -e "$SEMAPHORE" ]; then
    rm -f "$SEMAPHORE"
fi

# Initialize the semaphore
mkfifo "$SEMAPHORE" || exit 1
exec 3<>"$SEMAPHORE"
for ((i = 0; i < MAX_PARALLEL_JOBS; i++)); do
    echo >&3
done

# Iterate through all parameter combinations and generate commands
command_id=0  # Used to generate WandB run names

# Define task list
TASKS=('ioi' 'docstring' 'induction' 'tracr-reverse' 'tracr-proportion' 'greaterthan' 'or_gate')

# Iterate through tasks
for task in "${TASKS[@]}"; do
    # Create a separate log directory for each task
    task_log_dir="$LOG_DIR/$task"
    mkdir -p "$task_log_dir"

    for reset_network in "${RESET_NETWORKS[@]}"; do
        for zero_ablation in "${ZERO_ABLATIONS[@]}"; do
            for loss_type in "${LOSS_TYPES[@]}"; do
                for threshold in "${THRESHOLDS[@]}"; do
                    # Construct the command
                    command="python acdc/main.py \
                        --task=${task} \
                        --threshold=${threshold} \
                        --using-wandb \
                        --wandb-run-name=naive-acdc-$(printf "%03d" $command_id) \
                        --device=${DEVICE} \
                        --reset-network=${reset_network} \
                        --seed=${SEED} \
                        --metric=${loss_type} \
                        --torch-num-threads=${CPU} \
                        --wandb-dir=/autodl-tmp/acdc \
                        --wandb-mode=${WANDB_MODE}"
                    
                    # If zero_ablation is 1, add the parameter
                    if [ "$zero_ablation" -eq 1 ]; then
                        command="$command --zero-ablation"
                    fi

                    # Log file name
                    log_file="$task_log_dir/threshold_${threshold}_reset${reset_network}_zero${zero_ablation}_loss_${loss_type}.log"
                    
                    # Error log file name
                    error_file="$task_log_dir/threshold_${threshold}_reset${reset_network}_zero${zero_ablation}_loss_${loss_type}_error.log"

                    # Acquire a semaphore to ensure the number of parallel tasks does not exceed the limit
                    read -u 3

                    # Execute the task
                    {
                        echo "Running: $command"
                        $command > "$log_file" 2> "$error_file"
                        echo >&3  # Release the semaphore
                    } &  # **Key Point**: The task is executed in a child process, with `&` added

                    # Increment the command ID
                    command_id=$((command_id + 1))
                done
            done
        done
    done
done

# Wait for all tasks to complete
wait

# Close the semaphore
exec 3>&-

echo "All tasks are complete! Logs are stored in $LOG_DIR, and error logs are stored separately by task and parameters."
