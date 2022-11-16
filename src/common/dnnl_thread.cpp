#include <functional>

#include "dnnl_thread.hpp"

#if defined(DNNL_ENABLE_ITT_TASKS)
#include "common/ittnotify.hpp"
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "counting_barrier.hpp"
#endif

namespace dnnl {
namespace impl {

void parallel(int nthr, const std::function<void(int, int)> &f) {
    nthr = adjust_num_threads(nthr, INT64_MAX);
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
    for (int i = 0; i < nthr; ++i) {
        f(i, nthr);
    }
#else
#if defined(DNNL_ENABLE_ITT_TASKS)
    auto task_primitive_kind = itt::primitive_task_get_current_kind();
    bool itt_enable = itt::get_itt(itt::__itt_task_level_high);
#endif
    if (nthr == 1) {
        f(0, 1);
        return;
    }
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#pragma omp parallel num_threads(nthr)
    {
        int nthr_ = omp_get_num_threads();
        int ithr_ = omp_get_thread_num();
        assert(nthr_ == nthr);
#if defined(DNNL_ENABLE_ITT_TASKS)
        if (ithr_ && itt_enable) itt::primitive_task_start(task_primitive_kind);
#endif
        f(ithr_, nthr_);
#if defined(DNNL_ENABLE_ITT_TASKS)
        if (ithr_ && itt_enable) itt::primitive_task_end();
#endif
    }
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
    tbb::parallel_for(
            0, nthr,
            [&](int ithr) {
#if defined(DNNL_ENABLE_ITT_TASKS)
                bool mark_task = itt::primitive_task_get_current_kind()
                        == primitive_kind::undefined;
                if (mark_task && itt_enable)
                    itt::primitive_task_start(task_primitive_kind);
#endif
                f(ithr, nthr);
#if defined(DNNL_ENABLE_ITT_TASKS)
                if (mark_task && itt_enable) itt::primitive_task_end();
#endif
            },
            tbb::static_partitioner());
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB_AUTO
    tbb::parallel_for(
            0, nthr, [&](int ithr) { f(ithr, nthr); });
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    using namespace dnnl::impl::threadpool_utils;
    dnnl::threadpool_interop::threadpool_iface *tp = get_active_threadpool();
    if (!tp || dnnl_in_parallel()) {
        threadpool_utils::deactivate_threadpool();
        for (int ithr = 0; ithr < nthr; ithr++) {
            f(ithr, nthr);
        }
        threadpool_utils::activate_threadpool(tp);
    } else {
        bool async = tp->get_flags()
                & dnnl::threadpool_interop::threadpool_iface::ASYNCHRONOUS;
        counting_barrier_t b;
        if (async) b.init(nthr);
        tp->parallel_for(nthr, [&, tp](int ithr, int nthr) {
            bool is_master = threadpool_utils::get_active_threadpool() == tp;
            if (!is_master) {
                threadpool_utils::activate_threadpool(tp);
#if defined(DNNL_ENABLE_ITT_TASKS)
                if (itt_enable) itt::primitive_task_start(task_primitive_kind);
#endif
            }
            f(ithr, nthr);
            if (!is_master) {
#if defined(DNNL_ENABLE_ITT_TASKS)
                if (itt_enable) itt::primitive_task_end();
#endif
                threadpool_utils::deactivate_threadpool();
            }
            if (async) b.notify();
        });
        if (async) b.wait();
    }
#endif
#endif
}

} // namespace impl
} // namespace dnnl
