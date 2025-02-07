#!/bin/bash

export PYTHONPATH=/root/autodl-tmp/acdc
source /etc/network_turbo

# 定义参数
ZERO_ABLATIONS=(0)
SEED=424671755
DEVICE="cuda"
CPU=6  # PyTorch线程数
WANDB_MODE="online"  # WandB模式，可设置为 "offline" 或 "online"

# 创建日志目录
LOG_DIR="logs-16heads"

# 清空日志目录及其子目录
if [ -d "$LOG_DIR" ]; then
    rm -rf "$LOG_DIR"
fi
mkdir -p "$LOG_DIR"

# 设置最大并行任务数
MAX_PARALLEL_JOBS=3  # 可根据需求调整并行任务数
SEMAPHORE="/tmp/semaphore"  # 用于控制并行任务数

# 如果管道已存在，先删除
if [ -e "$SEMAPHORE" ]; then
    rm -f "$SEMAPHORE"
fi

# 初始化信号量
mkfifo "$SEMAPHORE" || exit 1
exec 3<>"$SEMAPHORE"
for ((i = 0; i < MAX_PARALLEL_JOBS; i++)); do
    echo >&3
done

# 遍历所有任务组合并生成命令
command_id=0  # 用于生成 WandB 运行名称

# 定义任务列表
TASKS=('ioi' 'docstring' 'induction' 'greaterthan' 'or_gate')

# 遍历任务
for task in "${TASKS[@]}"; do
    # 为每个任务创建单独的日志目录
    task_log_dir="$LOG_DIR/$task"
    mkdir -p "$task_log_dir"

    for zero_ablation in "${ZERO_ABLATIONS[@]}"; do
        # 构造命令
        command="python experiments/launch_sixteen_heads.py\
            --task=${task} \
            --using-wandb \
            --wandb-run-name=naive-acdc-16heads-$(printf "%03d" $command_id) \
            --device=${DEVICE} \
            --seed=${SEED} \
            --torch-num-threads=${CPU} \
            --wandb-dir=/autodl-tmp/acdc-16head \
            --wandb-mode=${WANDB_MODE}\
            --wandb-project=acdc-16head"
            
        # 如果 zero_ablation 为 1，则添加参数
        if [ "$zero_ablation" -eq 1 ]; then
            command="$command --zero-ablation"
        fi

        # 日志文件名
        log_file="$task_log_dir/zero${zero_ablation}.log"
        
        # 错误日志文件名
        error_file="$task_log_dir/zero${zero_ablation}_error.log"

        # 获取信号量，确保并行任务数不超过限制
        read -u 3

        # 执行任务
        {
            echo "Running: $command"
            $command > "$log_file" 2> "$error_file"
            echo >&3  # 释放信号量
        } &  # **关键点**：任务在子进程中执行，加入 `&` 符号

        # 增加命令编号
        command_id=$((command_id + 1))
    done
done

# 等待所有任务完成
wait

# 关闭信号量
exec 3>&-

echo "所有任务已完成！日志保存在 $LOG_DIR 中，错误日志按任务和参数分别存储。"
