/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_GENERIC_REF_CONCAT_HPP
#define GPU_GENERIC_REF_CONCAT_HPP

#include "common/primitive.hpp"
#include "common/reorder.hpp"
#include "common/reorder_pd.hpp"
#include "common/stream.hpp"
#include "gpu/gpu_concat_pd.hpp"
#include "gpu/gpu_engine.hpp"
#include "gpu/gpu_primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {

struct ref_concat_t : public gpu::primitive_t {
    using gpu::primitive_t::primitive_t;
    struct pd_t : public gpu_concat_pd_t {
        using gpu_concat_pd_t::gpu_concat_pd_t;

        DECLARE_CONCAT_PD_T("ref:any", ref_concat_t);

        status_t init(impl::engine_t *engine) {
            using sm = primitive_attr_t::skip_mask_t;

            VDISPATCH_CONCAT(attr()->has_default_values(sm::scales_runtime),
                    VERBOSE_UNSUPPORTED_ATTR);

            tent_dst_md_ = types::zero_md();

            if (gpu_concat_pd_t::init() != status::success) {
                assert(dst_md_.format_kind != format_kind::undef);
                VDISPATCH_CONCAT_SC(
                        memory_desc_init_by_strides(tent_dst_md_, dst_md_.ndims,
                                dst_md_.dims, dst_md_.data_type, nullptr),
                        VERBOSE_UNSUPPORTED_MEM_STRIDE);

                VDISPATCH_CONCAT_SC(gpu_concat_pd_t::init(&tent_dst_md_),
                        VERBOSE_PRIMITIVE_CREATION_FAIL, "concat");
            }

            const auto &sc = attr()->scales_;
            reorder_pds_.resize(n_ + use_tent_dst());
            for (int i = 0; i < n_; ++i) {
                primitive_attr_t r_attr;
                int mask = 0;
                bool is_set = false;
                VDISPATCH_CONCAT_SC(
                        sc.get(DNNL_ARG_MULTIPLE_SRC + i, &mask, &is_set),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                if (is_set) {
                    VDISPATCH_CONCAT(mask == 0, "non-zero mask");
                    VDISPATCH_CONCAT_SC(r_attr.scales_.set(DNNL_ARG_SRC, mask),
                            VERBOSE_UNSUPPORTED_SCALES_CFG);
                }
                VDISPATCH_CONCAT_SC(
                        reorder_primitive_desc_create(reorder_pds_[i], engine,
                                src_md(i), src_image_md(i), &r_attr),
                        "reorder_primitive_desc_create()");
            }

            if (use_tent_dst()) {
                assert(tent_dst_md_.format_kind != format_kind::undef);
                assert(dst_md_.format_kind != format_kind::undef);
                VDISPATCH_CONCAT_SC(
                        reorder_primitive_desc_create(reorder_pds_[n_], engine,
                                &tent_dst_md_, &dst_md_),
                        "reorder_primitive_desc_create");
            }
            init_scratchpad(engine);
            return status::success;
        }

        // if dst is forced and cannot be used directly.
        bool use_tent_dst() const { return !types::is_zero_md(&tent_dst_md_); }

        std::vector<std::shared_ptr<primitive_desc_t>> reorder_pds_;
        memory_desc_t tent_dst_md_;

    private:
        void init_scratchpad(impl::engine_t *engine) {
            auto *gpu_engine = utils::downcast<gpu::engine_t *>(engine);
            auto scratchpad = scratchpad_registry().registrar();

            if (use_tent_dst()) {
                const memory_desc_wrapper wtent_dst_md(tent_dst_md_);
                scratchpad.book(memory_tracking::names::key_concat_tent_dst,
                        wtent_dst_md.size(), 1,
                        gpu_engine->get_buffer_alignment());
            }

            for (size_t i = 0; i < reorder_pds_.size(); i++) {
                scratchpad.book(
                        memory_tracking::names::key_nested_multiple + (int)i,
                        reorder_pds_[i]->scratchpad_registry());
            }
        }
    };

    status_t init(impl::engine_t *engine) override {
        const size_t n = pd()->reorder_pds_.size();
        reorders_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            CHECK(create_nested_primitive(
                    reorders_[i], pd()->reorder_pds_[i], engine));
        }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        if (memory_desc_wrapper(pd()->dst_md()).size() == 0)
            return status::success;

        using namespace memory_tracking::names;
        impl::engine_t *engine = ctx.stream()->engine();
        const auto n = pd()->n_inputs();

        auto execute_reorder
                = [&](const std::shared_ptr<impl::primitive_t> &reorder,
                          const memory_arg_t &src, const memory_arg_t &dst,
                          const memory_arg_t *src_scales, int r_num) {
                      exec_args_t r_args;
                      r_args[DNNL_ARG_SRC] = src;
                      r_args[DNNL_ARG_DST] = dst;
                      if (src_scales)
                          r_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC]
                                  = *src_scales;
                      exec_ctx_t r_ctx(ctx, std::move(r_args));

                      nested_scratchpad_t ns(
                              ctx, key_nested_multiple + r_num, reorder);
                      r_ctx.set_scratchpad_grantor(ns.grantor());
                      return reorder->execute(r_ctx);
                  };

        if (pd()->use_tent_dst()) {
            auto scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                    memory_tracking::names::key_concat_tent_dst);

            std::unique_ptr<memory_t, memory_deleter_t> tent_dst;
            CHECK(safe_ptr_assign(tent_dst,
                    new memory_t(engine, &pd()->tent_dst_md_,
                            std::move(scratchpad))));

            for (int i = 0; i < n; ++i) {
                const auto &src_scales_arg = ctx.args().find(
                        DNNL_ARG_ATTR_SCALES | (DNNL_ARG_MULTIPLE_SRC + i));

                const memory_arg_t *src_scales = nullptr;
                if (src_scales_arg != ctx.args().end())
                    src_scales = &src_scales_arg->second;
                CHECK(execute_reorder(reorders_[i],
                        ctx.args().at(DNNL_ARG_MULTIPLE_SRC + i),
                        {tent_dst.get(), false}, src_scales, i));
            }

            CHECK(execute_reorder(reorders_[n], {tent_dst.get(), true},
                    ctx.args().at(DNNL_ARG_DST), nullptr, n));
        } else {
            for (int i = 0; i < n; ++i) {
                const auto &src_scales_arg = ctx.args().find(
                        DNNL_ARG_ATTR_SCALES | (DNNL_ARG_MULTIPLE_SRC + i));

                const memory_arg_t *src_scales = nullptr;
                if (src_scales_arg != ctx.args().end()) {
                    src_scales = &src_scales_arg->second;
                }
                CHECK(execute_reorder(reorders_[i],
                        ctx.args().at(DNNL_ARG_MULTIPLE_SRC + i),
                        ctx.args().at(DNNL_ARG_DST), src_scales, i));
            }
        }

        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<std::shared_ptr<impl::primitive_t>> reorders_;
};

} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
