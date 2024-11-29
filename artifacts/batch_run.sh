#!/bin/bash

# 定义参数列表
prefix_list=("temp/resnet18" "temp/resnet50" "temp/vgg16" "temp/unet" "temp/bert/1" "temp/bert/1_fp16" "temp/bert/64" "temp/bert/64_fp16" "temp/BSRN/1" "temp/BSRN/1_fp16" "temp/mobilenet/1" "temp/mobilenet/1_fp16" "temp/mobilenet/64" "temp/mobilenet/64_fp16" "temp/mobilevit/1" "temp/mobilevit/1_fp16" "temp/mobilevit/64" "temp/mobilevit/64_fp16" "temp/NAFNet/1" "temp/NAFNet/1_fp16" "temp/NAFNet/64" "temp/NAFNet/64_fp16" "temp/NeRF/1" "temp/NeRF/1_fp16" "temp/restormer/1" "temp/restormer/1_fp16" "temp/swin/1" "temp/swin/1_fp16" "temp/swin/64" "temp/swin/64_fp16" "temp/vit/1" "temp/vit/1_fp16" "temp/vit/64" "temp/vit/64_fp16" "temp/Conformer/1" "temp/Conformer/1_fp16" "temp/Conformer/64" "temp/Conformer/64_fp16") # 替换为实际的 prefix 列表


# 定义脚本列表
script_list=("run_ansor.py" "run_onnxrt.py" "run_welder.py") # 替换为实际的脚本文件名


# 遍历所有脚本文件
for script in "${script_list[@]}"; do
    # 定义输出文件
    output_file="results(${script}).txt"
    # 清空或创建输出文件
    >"$output_file"
    echo "Running script: $script"

    # 遍历所有参数组合
    for prefix in "${prefix_list[@]}"; do
        for device in "${device_list[@]}"; do
            # 构建命令行参数
            command="python $script --prefix $prefix --device $device"

            echo "Running with prefix=$prefix, device=$device"
            # 运行 Python 脚本并捕获输出
            result=$(eval "$command" 2>&1)

            # 使用正则表达式提取 time cost 信息
            avg_time=$(echo "$result" | grep -oP 'avg: \K[0-9.]+')
            min_time=$(echo "$result" | grep -oP 'min: \K[0-9.]+')
            max_time=$(echo "$result" | grep -oP 'max: \K[0-9.]+')

            # 检查是否提取成功
            if [[ -n "$avg_time" && -n "$min_time" && -n "$max_time" ]]; then
                # 将结果写入文件
                echo "script=$script, prefix=$prefix, device=$device, avg_time=$avg_time ms, min_time=$min_time ms, max_time=$max_time ms" >>"$output_file"
            else
                # 如果提取失败，记录错误信息
                echo -e "script=$script, prefix=$prefix, device=$device, log:\n$(tail -n 2 <<< "$result")\n\n" >>"$output_file"
            fi
        done
    done
done

echo "All runs completed. Results are saved in $output_file"
