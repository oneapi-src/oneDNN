/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef GPU_OCL_REF_SUM_HPP
#define GPU_OCL_REF_SUM_HPP

#include "common/primitive.hpp"
#include "common/reorder_pd.hpp"
#include "common/stream.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_sum_pd.hpp"
#include "gpu/ocl/ocl_resource.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ref_sum_t : public gpu_primitive_t {
    struct pd_t : public gpu_sum_pd_t {
        using gpu_sum_pd_t::gpu_sum_pd_t;
        pd_t(const pd_t &rhs) : gpu_sum_pd_t(rhs) { clone_reorder_pds(rhs); }

        ~pd_t() = default;

        pd_t &operator=(const pd_t &rhs) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(rhs);
            gpu_sum_pd_t::operator=(rhs);
            clone_reorder_pds(rhs);
            return *this;
        }

        DECLARE_SUM_PD_T("ref:any", ref_sum_t);

        status_t init(engine_t *engine) {
            bool ok = gpu_sum_pd_t::init(engine) == status::success;
            if (!ok) return status::unimplemented;

            for (int i = 0; i < n_; ++i) {
                auto r_impls = engine->get_reorder_implementation_list(
                        src_md(i), dst_md());
                for (auto r = r_impls; *r; ++r) {
                    primitive_attr_t r_attr;
                    r_attr.set_scratchpad_mode(scratchpad_mode::user);
                    r_attr.output_scales_.set(scales_[i]);
                    if (i != 0) r_attr.post_ops_.append_sum(1.0);

                    reorder_pd_t *r_pd;
                    if ((*r)(&r_pd, engine, &r_attr, engine, src_md(i), engine,
                                dst_md())
                            == status::success) {
                        reorder_pds_.emplace_back(r_pd);
                        break;
                    }
                }
            }
            ok = utils::everyone_is(reorder_pds_.size(), scales_.size());
            return ok ? status::success : status::unimplemented;
        }

        void clone_reorder_pds(const pd_t &rhs) {
            reorder_pds_.clear();
            for (size_t i = 0; i < rhs.reorder_pds_.size(); ++i)
                reorder_pds_.emplace_back(rhs.reorder_pds_[i]->clone());
        }

        std::vector<std::unique_ptr<primitive_desc_t>> reorder_pds_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            for (size_t i = 0; i < reorder_pds_.size(); i++) {
                scratchpad.book(
                        memory_tracking::names::key_nested_multiple + (int)i,
                        reorder_pds_[i]->scratchpad_registry().size());
            }
        }
    };

    ref_sum_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        const int n = pd()->n_inputs();
        reorders_.resize(n);
        for (int i = 0; i < n; ++i) {
            pd()->reorder_pds_[i]->create_primitive(reorders_[i], engine);
            reorders_[i]->init(engine);
        }
        return status::success;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_tracking::names;
        const auto n = pd()->n_inputs();
        for (int i = 0; i < n; ++i) {
            exec_args_t r_args;
            r_args[DNNL_ARG_SRC] = ctx.args().at(DNNL_ARG_MULTIPLE_SRC + i);
            r_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DST);
            exec_ctx_t r_ctx(ctx, std::move(r_args));

            nested_scratchpad_t ns(ctx, key_nested_multiple + i, reorders_[i]);
            r_ctx.set_scratchpad_grantor(ns.grantor());
            reorders_[i]->execute(r_ctx);
            ctx.stream()->wait();
        }
        return status::success;
    }

protected:
    primitive_list_t nested_primitives() const override {
        std::vector<const primitive_t *> _reorders;
        _reorders.reserve(reorders_.size());
        for (const auto &r : reorders_)
            _reorders.push_back(r.get());
        return _reorders;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<std::shared_ptr<primitive_t>> reorders_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
