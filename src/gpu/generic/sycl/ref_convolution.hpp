/*******************************************************************************
* Copyright 2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_SYCL_REF_CONVOLUTION_HPP
#define GPU_SYCL_REF_CONVOLUTION_HPP

#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_q10n.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

inline bool check_convolution_data_types(const memory_desc_wrapper &src0,
        const memory_desc_wrapper &src1, const memory_desc_wrapper &dst) {
    for (const auto &mdw : {src0, src1, dst}) {
        if (!is_supported_type(mdw.data_type())) return false;
    }

    return true;
}

inline bool check_convolution_formats(const memory_desc_wrapper &src0,
        const memory_desc_wrapper &src1, const memory_desc_wrapper &dst) {
    using namespace format_tag;

    for (const auto &mdw : {src0, src1, dst}) {
        if (!mdw.is_plain()) { return false; }
    }
    return true;
}

inline bool check_convolution_work_amount(
        const memory_desc_wrapper &weights, dim_t OC) {
    auto elems = weights.nelems();
    auto work_per_output = elems / OC;
    // arbitrarily chosen threshold to avoid unreasonably long runtimes
    // such cases should use a different implementation
    return work_per_output < 200000;
}

inline bool check_convolution_scales_types(const primitive_attr_t *attr) {
    const std::vector<int> supported_args
            = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};

    const auto &scales = attr->scales_;
    for (auto arg : supported_args) {
        auto dt = scales.get(arg).data_type_;
        if (!is_supported_type(dt)) { return false; }
    }
    return true;
}

struct ref_convolution_fwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public convolution_fwd_pd_t {
        using convolution_fwd_pd_t::convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_convolution_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            const memory_desc_wrapper data_d(src_md());
            const memory_desc_wrapper weights_d(weights_md());
            const memory_desc_wrapper dst_d(dst_md());

            VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(check_convolution_work_amount(weights_d, OC()),
                    VERBOSE_IMPL_HEURISTIC_FAIL,
                    "number of elements exceeds threshold");
            VDISPATCH_CONV(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(md_dims_in_range(src_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "src");
            VDISPATCH_CONV_SC(attr_.set_default_formats(dst_md()),
                    VERBOSE_UNSUPPORTED_TAG_S, "dst");
            VDISPATCH_CONV(
                    check_convolution_data_types(data_d, weights_d, dst_d),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_CONV(check_convolution_formats(data_d, weights_d, dst_d),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(attr()->has_default_values(sm::scales_runtime
                                   | sm::zero_points_runtime | sm::post_ops
                                   | sm::sum_dt),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONV(
                    IMPLICATION(!attr()->scales_.has_default_values(),
                            attr_scales_ok()
                                    && check_convolution_scales_types(attr())),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_CONV(sycl_post_ops_t::post_ops_ok(attr(), false),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);

            return init_conf();
        }

        sycl_convolution_fwd_conf_t conf_;

    private:
        status_t init_conf();

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    kernel_t kernel_;
};

struct ref_convolution_bwd_data_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public convolution_bwd_data_pd_t {
        using convolution_bwd_data_pd_t::convolution_bwd_data_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_convolution_bwd_data_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            const memory_desc_wrapper diff_data_d(diff_src_md());
            const memory_desc_wrapper weights_d(weights_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());

            VDISPATCH_CONV(is_bwd_d(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(check_convolution_work_amount(weights_d, OC()),
                    VERBOSE_IMPL_HEURISTIC_FAIL,
                    "number of elements exceed threshold");
            VDISPATCH_CONV(md_dims_in_range(src_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "src");
            VDISPATCH_CONV(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(check_convolution_data_types(
                                   diff_data_d, weights_d, diff_dst_d),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_CONV(check_convolution_formats(
                                   diff_data_d, weights_d, diff_dst_d),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(attr()->has_default_values(sm::scales_runtime
                                   | sm::zero_points_runtime | sm::sum_dt
                                   | sm::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONV(sycl_post_ops_t::post_ops_ok(attr(), false),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_CONV(
                    IMPLICATION(!attr()->scales_.has_default_values(),
                            attr_scales_ok()
                                    && check_convolution_scales_types(attr())),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(sycl_post_ops_t::post_ops_ok(attr(), false),
                    VERBOSE_UNSUPPORTED_POSTOP);

            return init_conf();
        }

        sycl_convolution_bwd_data_conf_t conf_;

    private:
        status_t init_conf();

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    kernel_t kernel_;
};

struct ref_convolution_bwd_weights_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public convolution_bwd_weights_pd_t {
        using convolution_bwd_weights_pd_t::convolution_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_convolution_bwd_weights_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            const memory_desc_wrapper data_d(src_md());
            const memory_desc_wrapper diff_weights_d(diff_weights_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());

            VDISPATCH_CONV(is_bwd_w(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(check_convolution_work_amount(diff_weights_d, OC()),
                    VERBOSE_IMPL_HEURISTIC_FAIL,
                    "number of elements exceed threshold");
            VDISPATCH_CONV(md_dims_in_range(src_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "src");
            VDISPATCH_CONV(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(check_convolution_data_types(
                                   data_d, diff_weights_d, diff_dst_d),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_CONV(check_convolution_formats(
                                   data_d, diff_weights_d, diff_dst_d),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);

            return init_conf();
        }

        sycl_convolution_bwd_weights_conf_t conf_;

    private:
        status_t init_conf();

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    kernel_t kernel_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
