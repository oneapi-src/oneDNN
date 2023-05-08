# Introducing Arm Compute Library verbose (RFC)

## Introduction

The goal for this RFC is to demonstrate an approach for enabling ACL verbose functionality that can capture low-level kernel information from [Compute Library for the Arm® Architecture (ACL)](https://github.com/ARM-software/ComputeLibrary) and kernel timing measurements and present the results in a oneDNN-style verbose output.

The implementation available [here](https://github.com/oneapi-src/oneDNN/compare/master...cfRod:oneDNN:acl_benchmark_scheduler) can be tested on a oneDNN build with ACL backend using examples and benchdnn, by setting `ONEDNN_VERBOSE=2`
By enabling ACL verbose in oneDNN, this implementation provides developers a unified logging interface to capture both high-level oneDNN primitive information with low-level ACL kernel information in one log.

This includes a minimal set of changes to oneDNN's `src/cpu/cpu_engine.hpp`, and, the addition of a benchmark scheduler implementation to oneDNN's `src/cpu/aarch64` directory. This scheduler implementation acts a wrapper scheduler that intercepts the actual scheduler used during primitive execution and is based on ACL's public [scheduler interface](https://github.com/ARM-software/ComputeLibrary/blob/master/arm_compute/runtime/IScheduler.h). A similar implementation exists in ACL's testing framework `libarm_compute_test_framework.a` called [Interceptor](https://github.com/ARM-software/ComputeLibrary/blob/cfb1c3035cbfc31a2fe8491c7df13e911698e2b6/tests/framework/instruments/SchedulerTimer.cpp#L53). However, using the Interceptor class introduces an additional build dependency on ACL's test framework, changes to the `execute()` method for each acl-based primitive and issues with supporting schedulers external to ACL (E.g `acl_threadpool_scheduler`).

### Motivation

Currently, oneDNN verbose provides primitive-level information for different ACL integrated primitives. To improve developer experience in terms of debugging and benchmarking, this implementation will expose finer-grained ACL information for higher settings of oneDNN verbose (i.e greater than or equal to 2). Example scenarios to use this functionality:

- A single primitive is mapped to multiple ACL kernels
This will give the user a breakdown time spent in different ACL kernels and enable them to compare the cummulative execution time of each ACL kernel with overall time from oneDNN.
- To identify if the right BF16 kernel is called for ONEDNN_DEFAULT_FPMATH_MODE.
Example logs:
```
ONEDNN_DEFAULT_FPMATH_MODE=BF16 OMP_NUM_THREADS=8 ./tests/benchdnn/benchdnn  --conv --stag=acdb --dtag=acdb mb1_ic3oc64_ih224oh112kh7sh2dh0ph3_iw224ow112kw7sw2dw0pw3
onednn_verbose,info,oneDNN v3.0.0 (commit 6ccb2b5d98f56d8286a4a939c66c2b597a849919)
onednn_verbose,info,cpu,runtime:OpenMP,nthr:8
onednn_verbose,info,cpu,isa:AArch64 SVE (256 bits)
onednn_verbose,info,gpu,runtime:none
onednn_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
onednn_verbose,create:cache_miss,cpu,convolution,gemm:acl,forward_training,src_f32::blocked:acdb:f0 wei_f32::blocked:acdb:f0 bia_f32::blocked:a:f0 dst_f32::blocked:acdb:f0,attr-fpmath:bf16 ,alg:convolution_direct,mb1_ic3oc64_ih224oh112kh7sh2dh0ph3_iw224ow112kw7sw2dw0pw3,9.99194
onednn_verbose,create:cache_hit,cpu,convolution,gemm:acl,forward_training,src_f32::blocked:acdb:f0 wei_f32::blocked:acdb:f0 bia_f32::blocked:a:f0 dst_f32::blocked:acdb:f0,attr-fpmath:bf16 ,alg:convolution_direct,mb1_ic3oc64_ih224oh112kh7sh2dh0ph3_iw224ow112kw7sw2dw0pw3,22.9929
onednn_verbose,create:cache_miss,cpu,reorder,simple:any,undef,src_f32::blocked:a:f0 dst_f32::blocked:a:f0,attr-fpmath:bf16 ,,64,0.171875
onednn_verbose,exec,cpu,reorder,simple:any,undef,src_f32::blocked:a:f0 dst_f32::blocked:a:f0,attr-fpmath:bf16 ,,64,23.4939
onednn_verbose,create:cache_miss,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:acdb:f0,attr-fpmath:bf16 ,,64x3x7x7,0.49707
onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:acdb:f0,attr-fpmath:bf16 ,,64x3x7x7,0.236816
onednn_verbose,create:cache_miss,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:acdb:f0,attr-fpmath:bf16 ,,1x3x224x224,0.197021
onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:acdb:f0,attr-fpmath:bf16 ,,1x3x224x224,0.0600586
onednn_acl_verbose,acl_kernel,CpuWeightsReshapeKernel,exec_time,0.431152
onednn_acl_verbose,acl_kernel,CpuIm2ColKernel,exec_time,0.693115
onednn_acl_verbose,acl_kernel,CpuGemmAssemblyWrapperKernel/a64_interleaved_bf16fp32_mmla_8x12,exec_time,0.37085
onednn_verbose,exec,cpu,convolution,gemm:acl,forward_training,src_f32::blocked:acdb:f0 wei_f32::blocked:acdb:f0 bia_f32::blocked:a:f0 dst_f32::blocked:acdb:f0,attr-fpmath:bf16 ,alg:convolution_direct,mb1_ic3oc64_ih224oh112kh7sh2dh0ph3_iw224ow112kw7sw2dw0pw3,1.76587
onednn_verbose,create:cache_miss,cpu,reorder,jit:uni,undef,src_f32::blocked:acdb:f0 dst_f32::blocked:abcd:f0,attr-fpmath:bf16 ,,1x64x112x112,0.359863
onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:acdb:f0 dst_f32::blocked:abcd:f0,attr-fpmath:bf16 ,,1x64x112x112,0.106934
0:PASSED __REPRO: --conv --stag=acdb --dtag=acdb --attr-fpmath=bf16 mb1ic3ih224oc64oh112kh7sh2ph3
tests:1 passed:1 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:0 listed:0
total compute_ref: sum(s):0.31
```
## Proposal

The key changes required are listed below:

- Adding a custom BenchmarkScheduler implementation inside `src/cpu/aarch64/`
- Method to set BenchmarkScheduler `acl_set_benchmark_scheduler_tp()` for threadpool scheduler and `acl_set_benchmark_scheduler_default()` for default schedulers in `src/cpu/aarch64/acl_thread.*`.
_Note: ACL can be built with different schedulers such as [OMPScheduler](https://github.com/ARM-software/ComputeLibrary/blob/main/arm_compute/runtime/OMP/OMPScheduler.h), [CPPScheduler](https://github.com/ARM-software/ComputeLibrary/blob/main/arm_compute/runtime/CPP/CPPScheduler.h) or linked to an external custom scheduler [ThreadpoolScheduler](https://github.com/oneapi-src/oneDNN/blob/4a7b48c4b6bfafcd07c222d7365fc42c1d9b224c/src/cpu/aarch64/acl_threadpool_scheduler.hpp)
- Call `acl_set_threading()` in `src/cpu/cpu_engine.hpp`

### Adding a custom BenchmarkScheduler implementation inside `src/cpu/aarch64/`

The BenchmarkScheduler behaves as a wrapper scheduler with extra instrumentation code added to `schedule()` and `schedule_op()` methods. It stores the address to the actual scheduler used by oneDNN+ACL referred to as `real_scheduler`.
~~~c++

class BenchmarkScheduler final : public arm_compute::IScheduler {
public:
    BenchmarkScheduler(IScheduler &real_scheduler);

    ~BenchmarkScheduler();

    void set_num_threads(unsigned int num_threads) override;
    unsigned int num_threads() const override;
    void set_num_threads_with_affinity(
            unsigned int num_threads, BindFunc func) override;

    void schedule(arm_compute::ICPPKernel *kernel,
            const arm_compute::IScheduler::Hints &hints) override;

    void schedule_op(arm_compute::ICPPKernel *kernel,
            const arm_compute::IScheduler::Hints &hints,
            const arm_compute::Window &window,
            arm_compute::ITensorPack &tensors) override;

protected:
    void run_workloads(std::vector<Workload> &workloads) override;
    void run_tagged_workloads(
            std::vector<Workload> &workloads, const char *tag) override;

private:
    IScheduler &_real_scheduler;
};
#endif
~~~
During execution, the `schedule()` and `schedule_op()` methods will be intercepted by the BenchmarkScheduler which then calls the equivalent methods for the actual scheduler in addtion to  capturing kernel-level information and timings as shown below:
~~~c++
void BenchmarkScheduler::schedule(ICPPKernel *kernel, const Hints &hints) {
    double start_ms = get_msec();
    _real_scheduler.schedule(kernel, hints);
    double duration_ms = get_msec() - start_ms;
    const char *name = kernel->name();
    printf("onednn_acl_verbose,acl_kernel,%s,exec_time,%g\n", name,
            duration_ms);
}
~~~
All lines beginning with `onednn_acl_verbose` contain ACL specific information.

### Method to set benchmark scheduler for ACL in `src/cpu/aarch64/acl_thread.*`

ACL provides the necessary infrastructure to plug in an external custom scheduler via methods such as [Scheduler::set()](https://github.com/ARM-software/ComputeLibrary/blob/7dcb9fadb98cad05fca72de3273311d570d98b4e/src/runtime/Scheduler.cpp#L128)
which sets a custom scheduler in ACL and a [Scheduler::get()](https://github.com/ARM-software/ComputeLibrary/blob/7dcb9fadb98cad05fca72de3273311d570d98b4e/src/runtime/Scheduler.cpp#L94) that can retrieve the scheduler used.
In the case where ACL is built with its own internal ACL scheduler (CPPScheduler or OMPScheduler), the `Scheduler::get()` will retrieve the raw pointer to the `real_scheduler` from ACL and passed it to the constructor of BenchmarkScheduler.
In the case of an external custom scheduler, the ThreadpoolScheduler is instantiated outside of ACL (i.e in oneDNN in [acl_set_custom_scheduler()](https://github.com/oneapi-src/oneDNN/blob/master/src/cpu/aarch64/acl_thread.cpp#L44), therefore, the raw pointer to the custom scheduler can be directly passed after its creation to the BenchmarkScheduler.
~~~c++
void acl_set_benchmark_scheduler_tp() {
    static std::once_flag flag_once;
    // Create threadpool scheduler
    std::unique_ptr<arm_compute::IScheduler> threadpool_scheduler
            = std::make_unique<ThreadpoolScheduler>();
    arm_compute::IScheduler *_real_scheduler = nullptr;
    _real_scheduler = threadpool_scheduler.release();
    // Create benchmark scheduler and set TP as real scheduler
    std::shared_ptr<arm_compute::IScheduler> benchmark_scheduler
            = std::make_unique<BenchmarkScheduler>(*_real_scheduler);
    std::call_once(flag_once,
            [&]() { arm_compute::Scheduler::set(benchmark_scheduler); });
}
void acl_set_benchmark_scheduler_default() {
    arm_compute::IScheduler *_real_scheduler = &arm_compute::Scheduler::get();
    std::shared_ptr<arm_compute::IScheduler> benchmark_scheduler
            = std::make_unique<BenchmarkScheduler>(*_real_scheduler);
    // set Benchmark scheduler in ACL
    std::call_once(flag_once, [&]() {
        arm_compute::Scheduler::set(
                std::static_pointer_cast<arm_compute::IScheduler>(
                        benchmark_scheduler));
    });
#endif
}
~~~

### Call `acl_set_threading()` in `src/cpu/cpu_engine.hpp`
`acl_set_threading()` will set BenchmarkScheduler only when oneDNN verbose is set to 2 or greater.

## Limitations of other approaches
- As mentioned previously, ACL's testing framework `libarm_compute_test_framework.a` provides a similar [Interceptor](https://github.com/ARM-software/ComputeLibrary/blob/cfb1c3035cbfc31a2fe8491c7df13e911698e2b6/tests/framework/instruments/SchedulerTimer.cpp#L53) class which can also be leveraged. However, using the Interceptor class introduces an additional build dependency on ACL's test framework and changes to the `execute()` method for each acl-based primitive to add profiling start and stop methods.
- Currently ACL can not support more than one external custom scheduler as it stores only one reference to the custom scheduler [here](https://github.com/ARM-software/ComputeLibrary/blob/cfb1c3035cbfc31a2fe8491c7df13e911698e2b6/src/runtime/Scheduler.cpp#L50) which can't be replaced by another custom scheduler. This issue exists with the current interceptor class, which is also defined as a custom scheduler inside the testing framework, see [here](https://github.com/ARM-software/ComputeLibrary/blob/cfb1c3035cbfc31a2fe8491c7df13e911698e2b6/tests/framework/instruments/SchedulerTimer.cpp#L198). This problem is circumvented by providing a similar light-weight "Interceptor" scheduler, which we refer to as `BenchmarkScheduler` in oneDNN.

## Results/validation

oneDNN's standard test suites were used for validation including ctests and benchdnn with verbose=2 to ensure accuracy of the implementation with ONEDNN_VERBOSE=2.
The cost of running this is small, typically <2% increase in runtime.

## Limited impact

The scope of this RFC, and the PRs outlined below, is limited such that:

- There are no changes to the API.
- There is no impact on non-AArch64 builds.

## Implementation plan
We propose implementing the changes using a single PR with all the above changes for ACL builds.
