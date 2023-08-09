/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_OCL_GEMM_CONV_GEMM_HPP
#define GPU_OCL_GEMM_CONV_GEMM_HPP

#ifdef DNNL_DEV_MODE

#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_gemm_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/jit/conv/gen_convolution.hpp"
#include "gpu/primitive_conf.hpp"
#include "gpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct conv_gemm_t : public gpu_gemm_t {
    using gpu_gemm_t::gpu_gemm_t;
    struct pd_t : public gpu_gemm_pd_t {
        using gpu_gemm_pd_t::gpu_gemm_pd_t;

        DECLARE_COMMON_PD_T("ocl:conv:ir", conv_gemm_t);

        status_t init(engine_t *engine) {
            // This is currently only used for experimentation purposes
            bool enable_conv_gemm
                    = gpu_utils::dev_getenv("enable_conv_gemm", false);
            if (!enable_conv_gemm) return status::unimplemented;

            if (!attr()->has_default_values(
                        primitive_attr_t::skip_mask_t::gpu_attr))
                return status::unimplemented;

            auto conv_desc = convolution_desc_t();

            auto *gemm_a_desc = src_md(0);
            auto *gemm_b_desc = src_md(1);
            auto *gemm_bias_desc = src_md(2);
            auto *gemm_c_desc = dst_md();

            auto with_bias = gemm_bias_desc->format_kind != format_kind::undef;

            auto add_width = [&](memory_desc_t &desc) {
                if (desc.ndims != 2) return status::unimplemented;

                // Add width dimension with size 1
                constexpr int width_idx = 2;
                constexpr int width_size = 1;
                desc.ndims++;
                desc.dims[width_idx] = width_size;
                desc.padded_dims[width_idx] = width_size;

                if (desc.format_kind == format_kind::blocked) {
                    auto &blk = desc.format_desc.blocking;
                    blk.strides[width_idx] = blk.strides[0];
                    return status::success;
                } else if (desc.format_kind == format_kind::any)
                    return status::success;
                else
                    return status::unimplemented;
            };

            auto transpose = [&](memory_desc_t &desc, int i, int j) {
                std::swap(desc.dims[i], desc.dims[j]);
                std::swap(desc.padded_dims[i], desc.padded_dims[j]);
                std::swap(desc.padded_offsets[i], desc.padded_offsets[j]);
                if (desc.format_kind == format_kind::blocked) {
                    auto &blk = desc.format_desc.blocking;
                    std::swap(blk.strides[i], blk.strides[j]);
                    for (int idx = 0; idx < blk.inner_nblks; idx++) {
                        if (blk.inner_idxs[idx] == i)
                            blk.inner_idxs[idx] = j;
                        else if (blk.inner_idxs[idx] == j)
                            blk.inner_idxs[idx] = i;
                    }
                } else {
                    assert(desc.format_kind == format_kind::any);
                }
            };

            // Enable using blocked format, otherwise, prefer spatial dimensions
            // as mb=1 is a more common optimization target than w=1.
            bool use_spatial_m = gpu_utils::dev_getenv("use_spatial_m",
                    !(gemm_a_desc->format_kind == format_kind::any
                            && gemm_a_desc->dims[0] > 8));

            // M x K x N -> use_spatial_m ? iw/ow x ic x oc : mb x ic x oc
            memory_desc_t src_desc = *gemm_a_desc, weights_desc = *gemm_b_desc,
                          dst_desc = *gemm_c_desc, bias_desc = *gemm_bias_desc;
            CHECK(add_width(src_desc));
            if (use_spatial_m) transpose(src_desc, 0, 2);
            CHECK(add_width(weights_desc));
            transpose(weights_desc, 0, 1);
            CHECK(add_width(dst_desc));
            if (use_spatial_m) transpose(dst_desc, 0, 2);

            if (with_bias) {
                // GEMM Bias has dimensions mxn with broadcasting semantics, but
                // Conv bias only has 1 dimension along oc. This could likely be
                // replaced with a binary add post-op for full support.
                if (bias_desc.ndims != 2 || bias_desc.dims[0] != 1)
                    return status::unimplemented;
                if (bias_desc.format_kind == format_kind::any) {
                    bias_desc.format_kind = format_kind::blocked;
                    auto &blk = bias_desc.format_desc.blocking;
                    blk = {{bias_desc.dims[1], 1}, 0, {}, {}};
                }
                transpose(bias_desc, 0, 1);
                bias_desc.ndims = 1;
            }

            dims_t zeroes {}, strides {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

            CHECK(dnnl::impl::conv_desc_init(&conv_desc,
                    with_bias ? prop_kind::forward_training
                              : prop_kind::forward_inference,
                    alg_kind::convolution_direct, &src_desc, &weights_desc,
                    &bias_desc, &dst_desc, strides, zeroes, zeroes, zeroes));

            primitive_desc_iterator_t it(
                    engine, (op_desc_t *)&conv_desc, attr(), nullptr);

            conv_pd = *(++it);
            if (!conv_pd) return status::unimplemented;
            if (strstr(conv_pd->name(), "jit:ir") == nullptr) {
                return status::unimplemented;
            }

            desc_.a_desc = *conv_pd->src_md();
            if (use_spatial_m) transpose(desc_.a_desc, 0, 2);
            desc_.a_desc.ndims = 2;

            desc_.b_desc = *conv_pd->weights_md();
            desc_.b_desc.ndims = 2;
            transpose(desc_.b_desc, 0, 1);

            desc_.c_desc = *conv_pd->dst_md();
            if (use_spatial_m) transpose(desc_.c_desc, 0, 2);
            desc_.c_desc.ndims = 2;

            if (with_bias) {
                desc_.bias_desc = bias_desc;
                transpose(desc_.bias_desc, 0, 1);
                desc_.bias_desc.ndims = 2;
            }

            init_scratchpad();

            return status::success;
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd->scratchpad_registry());
        }

        std::shared_ptr<primitive_desc_t> conv_pd;
    };

    status_t init(engine_t *engine) override {
        return create_nested_primitive(conv_, pd()->conv_pd, engine);
    }

    status_t execute(const gemm_exec_ctx_t &ctx) const override {
        exec_args_t args;
        memory_t a(ctx.stream()->engine(), pd()->conv_pd->src_md(0),
                ctx.args().a->clone());
        memory_t b(ctx.stream()->engine(), pd()->conv_pd->src_md(1),
                ctx.args().b->clone());
        memory_t c(ctx.stream()->engine(), pd()->conv_pd->dst_md(),
                ctx.args().c->clone());

        std::unique_ptr<memory_t> bias = [&] {
            if (ctx.args().bias) {
                return utils::make_unique<memory_t>(ctx.stream()->engine(),
                        pd()->conv_pd->src_md(2), ctx.args().bias->clone());
            } else {
                return std::unique_ptr<memory_t>();
            }
        }();

        args[DNNL_ARG_SRC] = {&a, true};
        args[DNNL_ARG_WEIGHTS] = {&b, true};
        args[DNNL_ARG_DST] = {&c, false};
        if (bias) args[DNNL_ARG_BIAS] = {bias.get(), true};

        auto exec_ctx = ctx.into_exec_ctx_t(std::move(args));

        nested_scratchpad_t ns(
                exec_ctx, memory_tracking::names::key_nested, conv_);
        exec_ctx.set_scratchpad_grantor(ns.grantor());

        CHECK(conv_->execute(exec_ctx));

        return status::success;
    };

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> conv_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
#endif
