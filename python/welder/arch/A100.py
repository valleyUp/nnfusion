import tvm


class A100:
    def __init__(self):
        self.reg_cap = 256000  # 256K registers per SM
        self.smem_cap = 102400  # 100KB shared memory per SM
        self.compute_max_core = 108  # 108 Streaming Multiprocessors (SMs)
        self.warp_size = 32
        self.sm_partition = 7  # Each SM can be partitioned into 7 GPCs
        self.transaction_size = [16, 128, 256]  # in bytes
        self.max_smem_usage = 100 * 1024  # Maximum shared memory usage per block
        self.bandwidth = [1938, 20480]  # GB/s (memcopy, peak HBM bandwidth)
        self.platform = "CUDA"
        self.compute_capability = "80"
        self.target = tvm.target.cuda(model="A100", arch="sm_80")

        self.cutlass_mma = [16, 16, 16]  # MMA tile sizes for Tensor Cores