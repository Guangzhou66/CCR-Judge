# CCR-Judge Ablation Blockers

| Testbed | Variant | Error Type | Error Message |
| --- | --- | --- | --- |
| layer1_mmlu_progressive_shuffle | ccr_random_shortlist_seed42 | RetryError | RetryError[<Future at 0x7f8f456beda0 state=finished raised RuntimeError>] |
| layer2_mmlu_parallel_shuffle | ccr_full | RuntimeError | NVML_SUCCESS == r INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp":1154, please report a bug to PyTorch.  |
| layer2_mmlu_parallel_shuffle | ccr_wo_comparative_grouping | RuntimeError | NVML_SUCCESS == r INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp":1154, please report a bug to PyTorch.  |
| layer2_mmlu_parallel_shuffle | ccr_wo_support_scoring | RuntimeError | NVML_SUCCESS == r INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp":1154, please report a bug to PyTorch.  |
| layer2_mmlu_parallel_shuffle | ccr_wo_shortlist_compression | RuntimeError | NVML_SUCCESS == r INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp":1154, please report a bug to PyTorch.  |
