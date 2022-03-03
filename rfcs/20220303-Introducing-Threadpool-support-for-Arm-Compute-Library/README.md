# Introducing Threadpool support for Arm Compute Library (RFC)

## Introduction

The goal for this RFC is to demonstrate an approach for enabling `DNNL_RUNTIME_THREADPOOL` support in oneDNN built with [Compute Library for Arm® Architecture (ACL)](https://github.com/ARM-software/ComputeLibrary).

The implementation available [here](https://github.com/cfRod/oneDNN/tree/poc-threadpool) can be tested using oneDNN's [threadpool](https://github.com/oneapi-src/oneDNN/blob/master/tests/test_thread.cpp#L85) test class available in `tests/test_thread.cpp`.
By enabling support for threadpool, Compute Library can leverage the existing Eigen threadpool defined in oneDNN.

This is presented as a basis for future work and includes a minimal set of changes to certain files in oneDNN's `src/cpu` directory, and, the addition of a custom scheduler implementation to oneDNN's `src/cpu/aarch64` directory, that enables ACL to run operations using oneDNN's threadpool via the [threadpool_iface](https://github.com/oneapi-src/oneDNN/blob/c0c5582d671bf573a576f4e63b2edeb02fbd0f8a/include/oneapi/dnnl/dnnl_threadpool_iface.hpp#L33). This scheduler implementation is based on ACL's public [scheduler interface](https://github.com/ARM-software/ComputeLibrary/blob/master/arm_compute/runtime/IScheduler.h). Additionally, some changes to `CMakeLists` are also required to exclude the scheduler sources for other runtime builds.

### Motivation

Currently our oneDNN+ACL integration stack depends on Compute Library's [CPPScheduler](https://github.com/ARM-software/ComputeLibrary/blob/master/src/runtime/CPP/CPPScheduler.cpp) to perform workload decomposition and scheduling during oneDNN's primitive execution phase. Frameworks such as TensorFlow, are built around the [threadpool implementation](https://github.com/tensorflow/tensorflow/blob/0b6b491d21d6a4eb5fbab1cca565bc1e94ca9543/tensorflow/core/platform/threadpool.cc) provided by the [Eigen Library](https://gitlab.corp.e-scopics.com/mirrors/eigen/-/tree/6d2dbfc45381141281625bd72a74f3d61d11bf1d/unsupported/Eigen/CXX11/src/ThreadPool) and hence, adding support for the use of external threadpools by ACL would enable the same threading runtime to be used across the stack. Additionally, when using the default CPPScheduler to execute workloads on a complete TensorFlow stack that includes oneDNN and ACL, resource contention is an issue between the different runtimes used across the stack. To mitigate some of these issues, our benchmarking has shown that capping the number of threads used by ACL below the number of available cores has a marked impact on performance.

## Proposal

The key changes required are listed below:

- Adding a custom ThreadpoolScheduler implementation inside `src/cpu/aarch64/`
- CMake changes to `src/cpu/aarch64/CMakeLists`
- Method to set custom scheduler for ACL in `src/cpu/aarch64/acl_utils.*`
- Add ACL runtime specific methods to oneDNN's engine creation step: `src/cpu/cpu_engine.hpp`
- Add `acl_set_num_threads()` to ThreadpoolScheduler's `schedule_op()` and `schedule_common()`
- Minor changes to `src/cpu/gemm/gemm.hpp`
- Minor changes to `src/cpu/platform.cpp`

### Adding a custom ThreadpoolScheduler implementation inside `src/cpu/aarch64/`

ACL provides a public [Scheduler interface](https://github.com/ARM-software/ComputeLibrary/blob/master/arm_compute/runtime/IScheduler.h) that is responsible for splitting workloads and scheduling. Several scheduler implementations (i.e OMP, CPP, ST etc.) of this interface are available in ACL and are defined as [scheduler types](https://github.com/ARM-software/ComputeLibrary/blob/91ee4d0a9ef128b16936921470a0e3ffef347536/arm_compute/runtime/Scheduler.h#L44).
The current version of a oneDNN+ACL build makes use of the default [CPPScheduler](https://github.com/ARM-software/ComputeLibrary/blob/master/arm_compute/runtime/CPP/CPPScheduler.h), which is shipped as part of the default ACL build.

 To support a runtime based on an external threadpool provided by oneDNN, a custom scheduler `ThreadpoolScheduler`, is implemented in oneDNN in `acl_threadpool_scheduler.[cpp/hpp]` residing in the AArch64 specific directory and is linked to ACL via the "CUSTOM" scheduler type option available from ACL.

 `ThreadpoolScheduler` class definition: [link](https://github.com/cfRod/oneDNN/blob/poc-threadpool/src/cpu/aarch64/acl_threadpool_scheduler.hpp)

 The [`run_workload()`](https://github.com/cfRod/oneDNN/blob/caa172e5c86977edb398cbf16d26e443d51b2da4/src/cpu/aarch64/acl_threadpool_scheduler.cpp#L101) of the `ThreadpoolScheduler` is the key function that  is responsible for the distribution of work across threads and will use oneDNN's threapool object to run the operations in parallel. The threadpool object is retrieved by the function `get_active_threadpool()` and is used to call `parallel_for()` which will call Eigen's [`Schedule()`](https://github.com/oneapi-src/oneDNN/blob/c0c5582d671bf573a576f4e63b2edeb02fbd0f8a/tests/test_thread.cpp#L104). The thread ID via `ithr` is then used by `process_workload()` to assign workloads to threads.
 ~~~c++
void ThreadpoolScheduler::run_workloads(
        std::vector<arm_compute::IScheduler::Workload> &workloads) {
...
    using namespace dnnl::impl::threadpool_utils;
    dnnl::threadpool_interop::threadpool_iface *tp = get_active_threadpool();
    bool is_async = tp->get_flags()
            & dnnl::threadpool_interop::threadpool_iface::ASYNCHRONOUS;
    counting_barrier_t b;
    if (is_async) b.init(num_threads);
    tp->parallel_for(num_threads, [&](int ithr, int nthr) {
        bool is_main = get_active_threadpool() == tp;
        if (is_main) activate_threadpool(tp);
        // Make ThreadInfo local to avoid race conditions
        ThreadInfo info;
        info.cpu_info = &cpu_info();
        info.num_threads = nthr;
        info.thread_id = ithr;
        process_workloads(workloads, feeder, info);
        if (is_main) deactivate_threadpool();
        if (is_async) b.notify();
    });
    if (is_async) b.wait();
}
~~~

###  Changes to `src/cpu/aarch64/CMakeLists`

A modified `CMakeLists` ensures that the ThreadpoolScheduler sources, i.e. `src/cpu/aarch64/acl_threadpool_scheduler.[cpp/hpp]` are excluded during the build phase for runtimes other than THREADPOOL. For non-ACL builds, these sources prefixed with `acl_*` are automatically excluded [here](https://github.com/oneapi-src/oneDNN/blob/c0c5582d671bf573a576f4e63b2edeb02fbd0f8a/src/cpu/aarch64/CMakeLists.txt#L29).


The following steps will be required to get a oneDNN and
ACL build with threadpool support:

- The user, having built ACL as per the [instructions](https://arm-software.github.io/ComputeLibrary/latest/how_to_build.xhtml), will need to set an environment variable
specifying ACL's root directory:

`export ACL_ROOT_DIR=/path/to/ComputeLibrary`;

- Use the CMake option to enable THREADPOOL builds: `-DDNNL_CPU_RUNTIME=THREADPOOL`.

For example:

`cmake -DCMAKE_BUILD_TYPE=RELEASE -DDNNL_CPU_RUNTIME=THREADPOOL -DDNNL_AARCH64_USE_ACL=ON -D_DNNL_TEST_THREADPOOL_IMPL=EIGEN -DEigen3_DIR=/path/to/EIGEN_CMAKE_INSTALL/share/eigen3/cmake ..`

### Method to set custom scheduler for ACL in `src/cpu/aarch64/acl_utils.*`

ACL provides the necessary infrastructure to plug in an external custom scheduler via methods such as [Scheduler::set()](https://github.com/ARM-software/ComputeLibrary/blob/7dcb9fadb98cad05fca72de3273311d570d98b4e/src/runtime/Scheduler.cpp#L128)
which sets a custom scheduler in ACL and a [Scheduler::get()](https://github.com/ARM-software/ComputeLibrary/blob/7dcb9fadb98cad05fca72de3273311d570d98b4e/src/runtime/Scheduler.cpp#L94) that can retrieve the custom scheduler object.
The `Scheduler::set()` method is used to instantiate the `ThreadpoolScheduler` object in oneDNN (in `src/cpu/aarch64/acl_utils.cpp`) via [`acl_set_custom_scheduler()`](https://github.com/cfRod/oneDNN/blob/caa172e5c86977edb398cbf16d26e443d51b2da4/src/cpu/aarch64/acl_utils.cpp#L123) as shown below:

~~~c++
void acl_set_custom_scheduler() {
if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    static std::once_flag flag_once;
    // Create custom threadpool scheduler
    std::shared_ptr<IScheduler> threadpool_scheduler
            = std::make_unique<ThreadpoolScheduler>();
    // Set custom scheduler in ACL
    std::call_once(
            flag_once, [&]() { arm_compute::Scheduler::set(threadpool_scheduler); });
}
~~~

### Add ACL runtime specific methods to oneDNN's engine creation step: `src/cpu/cpu_engine.hpp`

The creation of the ThreadpoolScheduler object and setting is done during the execution creation phase, i.e. at the [`engine_create(https://github.com/oneapi-src/oneDNN/blob/b7fd832ace0d893fe44ea3688de75ff04747eae0/src/cpu/cpu_engine.hpp#L163)`](). A single scheduler is set, ensuring that it is only called once and is used across different ACL-enabled primitives.

~~~c++
    status_t engine_create(engine_t **engine, size_t index) const override {
         assert(index == 0);
        *engine = new cpu_engine_t();

#if DNNL_AARCH64 && DNNL_AARCH64_USE_ACL
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
            // Number of threads in Compute Library is set by OMP_NUM_THREADS
            // dnnl_get_max_threads() == OMP_NUM_THREADS
            dnnl::impl::cpu::aarch64::acl_common_utils::acl_thread_bind();
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
           // Set ACL scheduler for threadpool runtime
           dnnl::impl::cpu::aarch64::acl_common_utils::acl_set_custom_scheduler();
#endif
#endif
        return status::success;
    };
~~~

The `Scheduler::get()` method is used at various points in ACL stack and is linked to the external custom `ThreadpoolScheduler` object.

### Add `acl_set_num_threads()` to ThreadpoolScheduler's `schedule_op()` and `schedule_common()`

During the primitive execution phase, a method called [`acl_set_threadpool_num_threads()`](https://github.com/cfRod/oneDNN/blob/caa172e5c86977edb398cbf16d26e443d51b2da4/src/cpu/aarch64/acl_utils.cpp#L133), defined in `src/cpu/aarch64/acl_utils.hpp`, sets the number of threads for the ThreadpoolScheduler based on the active threadpool. This is required so that ACL's scheduler interface can use this information to do workload decomposition and scheduling.

The example below shows ThreadpoolScheduler's `schedule()` and `schedule_op()` where the number of threads is set before the call to `ThreadpoolScheduler::run_workloads()`:

~~~c++
void ThreadpoolScheduler::schedule(ICPPKernel *kernel, const Hints &hints) {
    ITensorPack tensors;
    // Retrieve threadpool size during primitive execution and set ThreadpoolScheduler num_threads
    acl_common_utils::acl_set_threadpool_num_threads();
    schedule_common(kernel, hints, kernel->window(), tensors);
}

void ThreadpoolScheduler::schedule_op(ICPPKernel *kernel, const Hints &hints,
        const Window &window, ITensorPack &tensors) {
    // Retrieve threadpool size during primitive execution and set ThreadpoolScheduler num_threads
    acl_common_utils::acl_set_threadpool_num_threads();
    schedule_common(kernel, hints, window, tensors);
}
...
~~~


### Minor changes to `src/cpu/gemm/gemm.hpp`

Changes to `src/cpu/gemm/gemm.hpp`  conditionally adds `cpu_isa_traits.hpp` for AArch64 builds to ensure the header `common/dnnl_thread.hpp` that contains the namespace `threadpool_utils` is available for both reference and ACL builds with a threadpool runtime.

~~~c++
#if DNNL_AARCH64
#include "cpu/aarch64/cpu_isa_traits.hpp"
#endif
~~~

### Minor changes to `src/cpu/platform.cpp`

Adding of the function [`num_threads_hint()`]((https://github.com/ARM-software/ComputeLibrary/blob/c66f0e08bacd1041a4e5a7fda0eadbd868521a18/src/common/cpuinfo/CpuInfo.cpp#L362)) function from ACL ensures that oneDNN's `get_num_cores()` returns the number of threads available for use on AArch64 platforms. If a threadpool is available, [def_max_threads](https://github.com/oneapi-src/oneDNN/blob/3ca144dd41819a100b857260d03c5e627e4c2c83/src/common/dnnl_thread.hpp#L107) is the maximum number of threads available on the platform, as identified by ACL.

~~~c++
unsigned get_num_cores() {
#if DNNL_X64
    return x64::cpu().getNumCores(Xbyak::util::CoreLevel);
#elif DNNL_AARCH64_USE_ACL
    return arm_compute::cpuinfo::num_threads_hint();
#else
    return 1;
#endif
}
~~~
## Limitations

The main limitation of the implementation outlined in this RFC is that during the instantiation of the ThreadpoolScheduler (Note: which happens in oneDNN's primitive creation phase), the information about threadpool size is required in order for ACL's internal heuristics to select an optimized kernel for a given platform and provide hints on how to do workload partition. Since we do not have this information during primitive creation, ACL's `num_thread_hint()` function is used to provide the number of threads supported by the system. This function allows ACL to tune internal heuristics for kernel selection at ACL's configure stage.

See below the constructor of the ThreadpoolScheduler class:
~~~c++
ThreadpoolScheduler::ThreadpoolScheduler() {
    _num_threads = num_threads_hint();
}
~~~

However, during primitive execution we may use a different number of threads based on the available "active" threadpool. This implies that a sub-optimal kernel selection can be made by ACL during primitive creation or configure stage. Furthermore, combining both configure and execution phases also leads to performance overhead.

## Results/validation

oneDNN's standard test suites were used for validation including ctests and benchdnn.

Overall, the same number of oneDNN tests passed as the default ACL build using the OMP runtime.

## Expected performance benefits

Preliminary tests on matmul demonstrated performance comparable to that of using ACL's CPPScheduler and for high-core counts a slightly better performance.

## Limited impact

The scope of this RFC, and the PRs outlined below, is limited such that:

- There are no changes to the API.
- There is no impact on non-AArch64 builds.

## Implementation plan

We propose implementing the changes using a single PR with all the above changes for all ACL-enabled primitives.

With this initial implementation in place, future PRs will focus on improvements to the work scheduling in multi-threading
regime.
