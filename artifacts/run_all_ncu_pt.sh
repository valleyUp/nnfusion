#!/bin/bash
ncu_dir="/home/sosp/env/spack/opt/spack/linux-ubuntu22.04-zen2/gcc-11.4.0/cuda-12.4.0-ypujjdfaen2zwiplopzke4ud33wddscv/bin"

root_dir=$(pwd)
log_dir="$root_dir/logs"
benchmark_dir="FractalTensor/benchmarks"
mha_dir="$benchmark_dir/multi-head_attention/baseline"

bigbird_dir="$benchmark_dir/blocked_sparse_attention/pytorch"

# 2. ncu test the bigbird benchmark
echo "NCU profiling BigBird benchmark"
$ncu_dir/ncu --section "MemoryWorkloadAnalysis" \
    --csv --set full python3 $bigbird_dir/main.py > $log_dir/pt_bigbird_ncu.csv

