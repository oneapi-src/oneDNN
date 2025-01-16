#include "simple_reduction.hpp"

#include "gpu/generic/sycl/engine.hpp"
#include "gpu/generic/sycl/simple_reduction_kernels.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t simple_reduction_t::pd_t::init_conf() {
    conf_.alg = desc()->alg_kind;
    conf_.src_md = xpu::sycl::md_t(src_md());
    conf_.dst_md = xpu::sycl::md_t(dst_md());
    conf_.p = desc()->p;
    conf_.eps = desc()->eps;

    auto src_wrap = memory_desc_wrapper(src_md());
    auto dst_wrap = memory_desc_wrapper(dst_md());
    dst_nelems_ = dst_wrap.nelems();

    const auto ndims = dst_wrap.ndims();
    for (int d = 0; d < xpu::sycl::md_t::max_dims; d++) {
        conf_.reduce_dims[d] = dim_t {1};
        if (d < ndims) {
            if (src_wrap.dims()[d] != dst_wrap.dims()[d]) {
                conf_.reduce_dims[d] = src_wrap.dims()[d];
                conf_.reduce_size *= conf_.reduce_dims[d];
            }
        }
    }

    conf_.post_ops = sycl_post_ops_t(attr(), dst_wrap);

    return status::success;
}

status_t simple_reduction_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<simple_reduction_kernel_fwd_t>();
    CHECK(create_kernel(engine, kid, &kernel_));

    return status::success;
}

status_t simple_reduction_t::execute(const exec_ctx_t &ctx) const {
    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        simple_reduction_kernel_fwd_t reduction_kernel(pd()->conf_, cgh, ctx);
        cgh.parallel_for(::sycl::range<1>(pd()->dst_nelems_), reduction_kernel);
    });
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
