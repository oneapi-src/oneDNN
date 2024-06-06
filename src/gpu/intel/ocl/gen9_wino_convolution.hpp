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

#ifndef GPU_INTEL_OCL_GEN9_WINO_CONVOLUTION_HPP
#define GPU_INTEL_OCL_GEN9_WINO_CONVOLUTION_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct gen9_wino_convolution_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:gen9:wino", gen9_wino_convolution_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            auto src_data_t = this->desc()->src_desc.data_type;
            auto dst_data_t = this->desc()->dst_desc.data_type;

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;

            VDISPATCH_CONV(utils::one_of(this->desc()->prop_kind,
                                   forward_training, forward_inference),
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(
                    (this->desc()->alg_kind == alg_kind::convolution_winograd
                            || this->desc()->alg_kind
                                    == alg_kind::convolution_auto),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(utils::one_of(true,
                                   expect_data_types(f32, f32, f32, f32, f32),
                                   expect_data_types(f16, f16, f16, f16, f32)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_CONV(compute_engine->mayiuse(
                                   compute::device_ext_t::intel_subgroups),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "subgroups");
            VDISPATCH_CONV(
                    IMPLICATION(src_data_t == f16,
                            true
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16)
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::
                                                    intel_subgroups_short)),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "subgroups");
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(
                    attr()->has_default_values(attr_skip_mask, dst_data_t),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONV(post_ops_with_binary_ok(attr(), dst_data_t),
                    VERBOSE_UNSUPPORTED_POSTOP);

            VDISPATCH_CONV_SC(init_conf(compute_engine),
                    VERBOSE_PRIMITIVE_CREATION_FAIL, "convolution");

            int sub_group_size = conf.wino_ic_block / 2; // LWX
            VDISPATCH_CONV(compute_engine->mayiuse_sub_group(sub_group_size),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "subgroups");

            init_scratchpad();

            bool ok = set_default_formats_common(
                    conf.src_tag, conf.wei_tag, conf.dst_tag);
            VDISPATCH_CONV(ok, VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_CONV_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);

            return status::success;
        }

        status_t init_conf(compute::compute_engine_t *engine);
        void init_scratchpad();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        conv_conf_t conf;
    };

    status_t init(impl::engine_t *engine) override {
        bool is_fused = pd()->conf.is_fused;
        bool is_nonfused_2x3 = pd()->conf.wino_m == 2 && !is_fused;

        std::vector<const char *> kernel_names;
        if (is_fused) {
            kernel_names.push_back("gen9_wino_conv_fwd");
            kernel_names.push_back("gen9_wino_wei_transform");
        } else if (is_nonfused_2x3) {
            kernel_names.push_back("gen9_wino_conv_fwd_2x3");
            kernel_names.push_back("gen9_wino_wei_transform_2x3");
            kernel_names.push_back("gen9_wino_src_transform_2x3");
            kernel_names.push_back("gen9_wino_dst_transform_2x3");
        } else {
            assert(!"Invalid Winograd version chosen by init_conf");
            return status::unimplemented;
        }

        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        std::vector<compute::kernel_t> kernels;
        CHECK(create_kernels(engine, &kernels, kernel_names, kernel_ctx));
        kernel_ = kernels[0];
        wei_trans_kernel_ = kernels[1];
        if (!kernel_ || !wei_trans_kernel_) return status::runtime_error;
        if (!is_fused) {
            src_trans_kernel_ = kernels[2];
            dst_trans_kernel_ = kernels[3];
            if (!src_trans_kernel_ || !dst_trans_kernel_)
                return status::runtime_error;
        }

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
    compute::kernel_t wei_trans_kernel_;
    compute::kernel_t src_trans_kernel_;
    compute::kernel_t dst_trans_kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
