/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_CONVOLUTION_INNER_PRODUCT_HPP
#define GPU_INTEL_OCL_CONVOLUTION_INNER_PRODUCT_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_inner_product_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct convolution_inner_product_fwd_t : public gpu_primitive_t {
    struct pd_t : public gpu_inner_product_fwd_pd_t {
        using gpu_inner_product_fwd_pd_t::gpu_inner_product_fwd_pd_t;

        pd_t(const pd_t &rhs) = default;

        DECLARE_COMMON_PD_T("ocl:conv", convolution_inner_product_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::scales
                    | primitive_attr_t::skip_mask_t::post_ops;

            VDISPATCH_INNER_PRODUCT(
                    utils::one_of(desc()->prop_kind, forward_training,
                            forward_inference),
                    VERBOSE_BAD_PROPKIND);

            VDISPATCH_INNER_PRODUCT_SC(
                    set_default_params(true), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_INNER_PRODUCT(
                    IMPLICATION(with_bias(),
                            utils::one_of(desc()->bias_desc.data_type, u8, s8,
                                    bf16, f16, f32)),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);
            VDISPATCH_INNER_PRODUCT(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_INNER_PRODUCT(
                    post_ops_with_binary_ok(attr(), desc()->dst_desc.data_type),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_INNER_PRODUCT_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_INNER_PRODUCT(
                    IMPLICATION(desc()->src_desc.data_type == f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_INNER_PRODUCT(
                    (invariant_src_md()->format_desc.blocking.inner_nblks > 0
                            || invariant_wei_md()
                                            ->format_desc.blocking.inner_nblks
                                    > 0
                            || (src_md_.format_kind == format_kind::any
                                    && weights_md_.format_kind
                                            == format_kind::any)),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            VDISPATCH_INNER_PRODUCT_SC(
                    init_conf(engine), VERBOSE_PRIMITIVE_CREATION_FAIL, "ip");
            VDISPATCH_INNER_PRODUCT_SC(
                    init_scratchpad(), VERBOSE_SCRATCHPAD_INIT);
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        inner_product_conf_t conf;

        std::shared_ptr<primitive_desc_t> cpd_;
        std::shared_ptr<primitive_desc_t> rpd_postop_;
        std::shared_ptr<primitive_desc_t> rpd_dst_;

    private:
        status_t init_scratchpad();
    };

    convolution_inner_product_fwd_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(impl::engine_t *engine) override {
        CHECK(create_nested_primitive(conv_, pd()->cpd_, engine));
        if (pd()->rpd_postop_)
            CHECK(create_nested_primitive(
                    postop_reorder_, pd()->rpd_postop_, engine));
        if (pd()->rpd_dst_)
            CHECK(create_nested_primitive(
                    dst_reorder_, pd()->rpd_dst_, engine));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> conv_;
    std::shared_ptr<impl::primitive_t> postop_reorder_;
    std::shared_ptr<impl::primitive_t> dst_reorder_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
