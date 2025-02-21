/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef GPU_INTEL_OCL_MULTI_PO_REORDER_SUM_HPP
#define GPU_INTEL_OCL_MULTI_PO_REORDER_SUM_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/reorder.hpp"
#include "common/reorder_pd.hpp"
#include "common/stream.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/gpu_sum_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct multi_po_reorder_sum_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_sum_pd_t {
        using gpu_sum_pd_t::gpu_sum_pd_t;

        pd_t(const pd_t &rhs) = default;
        ~pd_t() override = default;

        DECLARE_SUM_PD_T("multi_po_reorder_sum", multi_po_reorder_sum_t);

        status_t init(impl::engine_t *engine) {
            VDISPATCH_SUM_SC(
                    gpu_sum_pd_t::init(engine), VERBOSE_BAD_ENGINE_KIND);

            if (has_zero_dim_memory()) return status::success;

            auto lin = [](primitive_attr_t &attr, float scale) {
                if (scale == 1.f) return status::success;
                return attr.post_ops_.append_eltwise(
                        1.f, alg_kind_t::dnnl_eltwise_linear, scale, 0.f);
            };
            auto new_reorder = [&](int bgn, int end, bool scalar_scales) {
                const bool first_r = reorder_pds_.empty();
                const bool final_r = end == n_inputs();

                primitive_attr_t attr;
                const auto scale = scales()[bgn];
                for (int i = bgn + 1; i < end; i++) {
                    if (scale != scales()[i]) return status::unimplemented;
                    CHECK(attr.post_ops_.append_binary(
                            dnnl::impl::alg_kind::binary_add, src_md(i)));
                }
                if (!scalar_scales || (first_r && final_r))
                    CHECK(lin(attr, scale));
                reorder_pds_.emplace_back(nullptr);
                if (first_r || !final_r || !need_output_reorder()) {
                    if (!first_r) CHECK(attr.post_ops_.append_sum(1.f));
                    if (scalar_scales && !first_r && final_r)
                        CHECK(lin(attr, scale));
                    CHECK(reorder_primitive_desc_create(reorder_pds_.back(),
                            engine, src_md(bgn),
                            (final_r) ? dst_md() : dst_acc_md(), &attr));
                } else { // !first_r && final_r && need_output_reorder
                    if (attr.post_ops_.len() < post_ops_t::post_ops_limit) {
                        CHECK(attr.post_ops_.append_binary(
                                dnnl::impl::alg_kind::binary_add,
                                dst_acc_md()));
                        if (scalar_scales) CHECK(lin(attr, scale));
                        CHECK(reorder_primitive_desc_create(reorder_pds_.back(),
                                engine, src_md(bgn), dst_md(), &attr));
                    } else {
                        CHECK(attr.post_ops_.append_sum(1.f));
                        if (scalar_scales) CHECK(lin(attr, scale));
                        CHECK(reorder_primitive_desc_create(reorder_pds_.back(),
                                engine, src_md(bgn), dst_acc_md(), &attr));
                        reorder_pds_.emplace_back(nullptr);
                        CHECK(reorder_primitive_desc_create(reorder_pds_.back(),
                                engine, dst_acc_md(), dst_md()));
                    }
                }
                return status::success;
            };
            const auto *s = scales();
            bool scalar_scales = true;
            for (int i = 1; scalar_scales && (i < n_inputs()); i++) {
                scalar_scales &= s[0] == s[i];
            }
            int bgn = 0;
            memory_desc_t dst_md_type(*dst_md());
            for (int i = 1; i < n_inputs(); i++) {
                dst_md_type.data_type = src_md(i)->data_type;
                if (!dnnl_memory_desc_equal(&dst_md_type, src_md(i))
                        || (s[i] != s[bgn])
                        || (i - bgn > post_ops_t::post_ops_limit)) {
                    VDISPATCH_SUM_SC(new_reorder(bgn, i, scalar_scales),
                            "new_reorder()");
                    bgn = i;
                }
            }
            VDISPATCH_SUM_SC(new_reorder(bgn, n_inputs(), scalar_scales),
                    "new_reorder()");

            init_scratchpad();
            return status::success;
        }

        std::vector<std::shared_ptr<primitive_desc_t>> reorder_pds_;

    private:
        void init_scratchpad() {
            if (reorder_pds_.size() < 2) return;

            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            if (need_output_reorder()) {
                const memory_desc_wrapper dst_acc_d(dst_acc_md());
                scratchpad.book(key_sum_reduction, dst_acc_d.size(), 1,
                        OCL_BUFFER_ALIGNMENT);
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
        using namespace memory_tracking::names;

        if (pd()->has_zero_dim_memory()) return status::success;

        std::unique_ptr<memory_t, memory_deleter_t> p_temp_dst_acc;
        const bool need_output_reorder
                = pd()->need_output_reorder() && (reorders_.size() > 1);
        if (need_output_reorder) {
            auto scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                    key_sum_reduction);
            CHECK(safe_ptr_assign(p_temp_dst_acc,
                    new memory_t(ctx.stream()->engine(), pd()->dst_acc_md(),
                            std::move(scratchpad))));
        }

        auto dst = ctx.args().at(DNNL_ARG_DST);
        memory_arg_t dst_acc = {p_temp_dst_acc.get(), false};

        auto has_bin = [](const post_ops_t &po) {
            bool retn = false;
            for (int i = 0; !retn && (i < po.len()); i++)
                retn |= po.entry_[i].is_binary();
            return retn;
        };
        for (int r = 0, s = 0; r < int(reorders_.size()); r++) {
            exec_args_t r_args;
            const bool final_reorder = r == int(reorders_.size()) - 1;
            const auto &post_ops = reorders_[r]->pd()->attr()->post_ops_;
            if (final_reorder) dst_acc = {p_temp_dst_acc.get(), true};
            r_args[DNNL_ARG_SRC] = dst_acc;
            r_args[DNNL_ARG_DST] = dst;
            if (!final_reorder || !need_output_reorder || has_bin(post_ops)) {
                r_args[DNNL_ARG_SRC]
                        = ctx.args().at(DNNL_ARG_MULTIPLE_SRC + s++);
                if (need_output_reorder && !final_reorder)
                    r_args[DNNL_ARG_DST] = dst_acc;
                for (int p = 0, pl = post_ops.len(); p < pl; p++) {
                    if (!post_ops.entry_[p].is_binary()) continue;
                    memory_arg_t arg = dst_acc;
                    if (!need_output_reorder || !final_reorder || (p < pl - 1))
                        arg = ctx.args().at(DNNL_ARG_MULTIPLE_SRC + s++);
                    r_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(p) | DNNL_ARG_SRC_1]
                            = arg;
                }
            }
            exec_ctx_t r_ctx(ctx, std::move(r_args));
            nested_scratchpad_t ns(ctx, key_nested_multiple + r, reorders_[r]);
            r_ctx.set_scratchpad_grantor(ns.grantor());
            CHECK(reorders_[r]->execute(r_ctx));
        }
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<std::shared_ptr<impl::primitive_t>> reorders_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
