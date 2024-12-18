
#ifndef GPU_GENERIC_SYCL_SIMPLE_REDUCTION_KERNELS_HPP
#define GPU_GENERIC_SYCL_SIMPLE_REDUCTION_KERNELS_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_math_utils.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "xpu/sycl/memory_storage_base.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct Reducer {
    dnnl_alg_kind_t alg_;
    float p_, eps_;

    Reducer(dnnl_alg_kind_t alg, float p, float eps)
        : alg_(alg), p_(p), eps_(eps) {}

    float identity() const {
        if (alg_ == dnnl_reduction_min) {
            return std::numeric_limits<float>::max();
        } else if (alg_ == dnnl_reduction_max) {
            return std::numeric_limits<float>::lowest();
        } else if (alg_ == dnnl_reduction_mul) {
            return 1.f;
        }

        return 0.f;
    }

    float reduce(float lhs, float rhs) const {
        if (alg_ == dnnl_reduction_sum || alg_ == dnnl_reduction_mean) {
            return lhs + rhs;
        } else if (alg_ == dnnl_reduction_min) {
            return ::sycl::min(lhs, rhs);
        } else if (alg_ == dnnl_reduction_max) {
            return ::sycl::max(lhs, rhs);
        } else if (alg_ == dnnl_reduction_mul) {
            return lhs * rhs;
        } else if (alg_ == dnnl_reduction_norm_lp_max
                || alg_ == dnnl_reduction_norm_lp_sum
                || alg_ == dnnl_reduction_norm_lp_power_p_max
                || alg_ == dnnl_reduction_norm_lp_power_p_sum) {
            return lhs + ::sycl::pow(::sycl::fabs(rhs), p_);
        }

        return ::sycl::nan(0U);
    }

    float finalize(float val, int size) const {
        if (alg_ == dnnl_reduction_mean) {
            return val / size;
        } else if (alg_ == dnnl_reduction_norm_lp_max) {
            return ::sycl::rootn(::sycl::max(val, eps_), p_);
        } else if (alg_ == dnnl_reduction_norm_lp_sum) {
            return ::sycl::rootn(val + eps_, p_);
        } else if (alg_ == dnnl_reduction_norm_lp_power_p_max) {
            return ::sycl::max(val, eps_);
        } else if (alg_ == dnnl_reduction_norm_lp_power_p_sum) {
            return val + eps_;
        }

        return val;
    }
};

struct reduction_kernel_fwd_t {
    sycl_simple_reduction_conf_t conf_;
    xpu::sycl::in_memory_arg_t src_;
    xpu::sycl::out_memory_arg_t dst_;
    post_op_input_args po_args_;

    reduction_kernel_fwd_t(const sycl_simple_reduction_conf_t &conf,
            ::sycl::handler &cgh, const exec_ctx_t &ctx)
        : conf_(conf)
        , src_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC))
        , dst_(CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST))
        , po_args_(cgh, ctx, conf_.post_ops) {}

    void operator()(::sycl::item<1> item) const {
        Reducer reducer(conf_.alg, conf_.p, conf_.eps);

        memory_tensor_t<::sycl::access_mode::read> src(src_, conf_.src_md);
        memory_tensor_t<::sycl::access_mode::write> dst(dst_, conf_.dst_md);
        const int id = item.get_linear_id();

        const auto &dst_md = conf_.dst_md;
        dims_t pos;
        int l_offset = id;
        for (int i = 0; i < dst_md.ndims(); i++) {
            const int d = dst_md.ndims() - 1 - i;
            const dim_t cur_dim = dst_md.dims()[d];
            pos[d] = l_offset % cur_dim;
            l_offset = l_offset / cur_dim;
        }

        float acc = reducer.identity();
        for (off_t d0 = 0; d0 < conf_.reduce_dims[0]; d0++)
            for (off_t d1 = 0; d1 < conf_.reduce_dims[1]; d1++)
                for (off_t d2 = 0; d2 < conf_.reduce_dims[2]; d2++)
                    for (off_t d3 = 0; d3 < conf_.reduce_dims[3]; d3++)
                        for (off_t d4 = 0; d4 < conf_.reduce_dims[4]; d4++)
                            for (off_t d5 = 0; d5 < conf_.reduce_dims[5];
                                    d5++) {
                                dims_t src_off = {pos[0] + d0, pos[1] + d1,
                                        pos[2] + d2, pos[3] + d3, pos[4] + d4,
                                        pos[5] + d5};
                                const float val = src.load_md(src_off);
                                acc = reducer.reduce(acc, val);
                            }

        float result = reducer.finalize(acc, conf_.reduce_size);
        result = conf_.post_ops.apply(result, dst.load_md(pos), po_args_, pos);
        dst.store_md(result, pos);
    }
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
