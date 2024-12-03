#!/bin/bash

# 定义模型前缀列表
prefix_list=(
    "temp/resnet18" "temp/resnet50"
    "temp/vgg16" "temp/unet"
    "temp/bert" "temp/BSRN"
    "temp/mobilenet" "temp/mobilevit"
    "temp/NAFNet" "temp/NeRF"
    "temp/restormer" "temp/swin"
    "temp/vit" "temp/Conformer"
)

# 定义脚本列表
script_list=("run_blade.py" "run_nimble.py" "run_torch.py")

# 定义 fp16 列表
fp16_list=("true" "false")

# 定义批量大小列表
bs_list=(1 64)

# 遍历所有脚本文件
for script in "${script_list[@]}"; do
    # 提取模型名称以便生成输出文件名
    model_name=$(basename "$script" ".py" | sed 's/^run_//')
    output_file="results_${model_name}.txt"

    # 清空或创建输出文件
    >"$output_file"
    echo "Running script: $script"

    # 遍历所有模型前缀
    for prefix in "${prefix_list[@]}"; do
        for fp16 in "${fp16_list[@]}"; do
            for bs in "${bs_list[@]}"; do
                # 构建基础命令行参数
                command="python $script --prefix $prefix --bs $bs"

                # 根据 fp16 的值添加 --fp16 参数
                if [ "$fp16" == "true" ]; then
                    command+=" --fp16"
                fi

                echo "Running with cmd: $command"

                # 执行 Python 脚本并捕获输出
                result=$(eval "$command" 2>&1)

                # 使用正则表达式提取 time cost 信息
                avg_time=$(echo "$result" | grep -oP 'avg: \K[0-9.]+')
                min_time=$(echo "$result" | grep -oP 'min: \K[0-9.]+')
                max_time=$(echo "$result" | grep -oP 'max: \K[0-9.]+')

                # 检查是否成功提取了所有时间信息
                if [[ -n "$avg_time" && -n "$min_time" && -n "$max_time" ]]; then
                    # 将结果写入文件
                    echo "cmd: $command, avg_time=$avg_time ms, min_time=$min_time ms, max_time=$max_time ms" >>"$output_file"
                else
                    # 如果提取失败，记录错误信息
                    echo -e "cmd: $command, log:\n$(tail -n 2 <<< "$result")\n\n" >>"$output_file"
                fi
            done
        done
    done
done

echo "All runs completed. Results are saved in $output_file"