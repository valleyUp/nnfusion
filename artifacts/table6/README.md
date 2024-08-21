## Profile about memory behavior

The profiling results shown in Table 6 are based on [NVIDIA Nsight Compute (ncu)](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html), which offers a non-interactive method to analyze NVIDIA CUDA kernels via the command line. Below, we describe the steps to use ncu to reproduce the results presented in Table 6.

### Environment preparation

1. Ensure that your test cluster account has root privileges, as ncu tools require these permissions.
2. Use the root account to install FractalTensor and the other baseline tools on the cluster.

### Usage

1. Locate the executable file for the ncu tool: it is typically found in the `bin` directory of the CUDA toolkit.
   
   - For instance, the ncu executable is usually located in `$CUDA_HOME/bin/ncu`.

2. Profile the kernel: To measure the memory traffic through the kernel, use ncu's `--section "MemoryWorkloadAnalysis"` parameter.
   
   - For example, to test the memory behavior of FractalTensor on the FlashAttention2 benchmark, the following command can be used:
  
      ```bash
      mha_dir="$benchmark_dir/multi-head_attention/fractaltensor/build"
      mha_exe="$mha_dir/main"
      ncu --section "MemoryWorkloadAnalysis" --csv --set full $mha_exe > profile_ft.csv
      ```
      
   - For example, to test the memory behavior of the Triton baseline on the FlashAttention2 benchmark, the following command can be used:
   
      ```bash
      benchmark_dir="FractalTensor/benchmarks"
      mha_dir="$benchmark_dir/multi-head_attention/baseline"
      ncu --section "MemoryWorkloadAnalysis" \
          --csv --set full python3 $mha_dir/test_triton_model.py > profile_triton.csv
      ```

3. Analyze the profile results

   In the output file of the profile results, you will find the memory traffic behavior of the kernel of interest. You can then further process and analyze these results.
   
   We cannot pre-assign names due to libraries like Triton having internal implementations that call extra kernels. Filtering based on names is not feasible. To address this, we run profiling multiple times (e.g., three) to observe log outputs, then run the tested program several times (e.g., five) to identify patterns. This helps us pinpoint actual kernel calls and post-process the ncu profiling logs to compute network traffic over the memory hierarchy.
