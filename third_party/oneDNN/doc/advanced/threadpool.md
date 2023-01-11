Using oneDNN with Threadpool-Based Threading {#dev_guide_threadpool}
====================================================================

When oneDNN is built with the threadpool CPU runtime (see @ref
dev_guide_build_options), oneDNN requires the user to implement a threadpool
interface to enable the library to perform computations using multiple
threads.

The threadpool interface is defined in
``include/oneapi/dnnl/dnnl_threadpool_iface.hpp``. Below is a sample
implementation based on the Eigen threadpool that is also used for testing (see
`tests/test_thread.cpp`).

~~~cpp
#include "dnnl_threadpool_iface.hpp"

class threadpool_t : public dnnl::threadpool_interop::threadpool_iface {
private:
    // Change to Eigen::NonBlockingThreadPool if using Eigen <= 3.3.7
    std::unique_ptr<Eigen::ThreadPool> tp_;

public:
    explicit threadpool_t(int num_threads = 0) {
        if (num_threads <= 0)
            num_threads = (int)std::thread::hardware_concurrency();
        tp_.reset(new Eigen::ThreadPool(num_threads));
    }
    int get_num_threads() const override { return tp_->NumThreads(); }
    bool get_in_parallel() const override {
        return tp_->CurrentThreadId() != -1;
    }
    uint64_t get_flags() override { return ASYNCHRONOUS; }
    void parallel_for(int n, const std::function<void(int, int)> &fn) override {
        int nthr = get_num_threads();
        int njobs = std::min(n, nthr);

        for (int i = 0; i < njobs; i++) {
            tp_->Schedule([i, n, njobs, fn]() {
                int start, end;
                impl::balance211(n, njobs, i, start, end);
                for (int j = start; j < end; j++)
                    fn(j, n);
            });
        }
    };
};
~~~
