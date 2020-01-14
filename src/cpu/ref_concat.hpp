/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#ifndef REF_CONCAT_HPP
#define REF_CONCAT_HPP

#include "engine.hpp"
#include "reorder_pd.hpp"

#include "cpu_concat_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct ref_concat_t : public primitive_impl_t {
    struct pd_t : public cpu_concat_pd_t {
        pd_t(engine_t *engine, const primitive_attr_t *attr,
                const memory_desc_t *dst_md, int n, int concat_dim,
                const memory_desc_t *src_mds)
            : cpu_concat_pd_t(engine, attr, dst_md, n, concat_dim, src_mds)
            , tent_dst_md_(types::zero_md()) {}
        pd_t(const pd_t &rhs) : cpu_concat_pd_t(rhs) { copy(rhs); }
        ~pd_t() { clear(); }

        pd_t &operator=(const pd_t &rhs) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(rhs);
            cpu_concat_pd_t::operator=(rhs);
            clear();
            copy(rhs);
            return *this;
        }

        DECLARE_CONCAT_PD_T("ref:any", ref_concat_t);

        status_t init() {
            status_t status = cpu_concat_pd_t::init();
            if (status != status::success) {
                assert(dst_md_.format_kind != format_kind::undef);
                status = dnnl_memory_desc_init_by_strides(&tent_dst_md_,
                        dst_md_.ndims, dst_md_.dims, dst_md_.data_type,
                        nullptr);
                if (status != status::success) return status::unimplemented;

                status = cpu_concat_pd_t::init(&tent_dst_md_);
                if (status != status::success) return status::unimplemented;
            }

            for (int i = 0; i < n_; ++i) {
                auto r_impls = engine_->get_reorder_implementation_list(
                        src_md(i), src_image_md(i));
                for (auto r = r_impls; *r; ++r) {
                    const primitive_attr_t attr; /* alpha == 1. */
                    reorder_pd_t *r_pd = nullptr;
                    if ((*r)(&r_pd, engine_, &attr, engine_, src_md(i), engine_,
                                src_image_md(i))
                            == status::success) {
                        reorder_pds_.push_back(r_pd);
                        break;
                    }
                }
            }
            if (reorder_pds_.size() != (size_t)n_) return status::unimplemented;

            if (use_tent_dst()) {
                assert(tent_dst_md_.format_kind != format_kind::undef);
                assert(dst_md_.format_kind != format_kind::undef);

                primitive_desc_t *r_pd = nullptr;
                status = dnnl_reorder_primitive_desc_create(&r_pd,
                        &tent_dst_md_, engine_, &dst_md_, engine_, nullptr);
                if (status != status::success) return status;
                reorder_pds_.push_back((const reorder_pd_t *)r_pd);

                using namespace memory_tracking::names;
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.book(memory_tracking::names::key_concat_tent_dst,
                        dnnl_memory_desc_get_size(&tent_dst_md_));
            }

            return status;
        }

        // if dst is forced and cannot be used directly.
        bool use_tent_dst() const { return !types::is_zero_md(&tent_dst_md_); }

        std::vector<const reorder_pd_t *> reorder_pds_;
        memory_desc_t tent_dst_md_;

    private:
        void copy(const pd_t &rhs) {
            tent_dst_md_ = rhs.tent_dst_md_;
            for (size_t i = 0; i < rhs.reorder_pds_.size(); ++i)
                reorder_pds_.push_back(
                        (const reorder_pd_t *)rhs.reorder_pds_[i]->clone());
        }

        void clear() {
            for (auto &rpd : reorder_pds_)
                delete rpd;
        }
    };

    ref_concat_t(const pd_t *apd) : primitive_impl_t(apd) {
        const size_t n = pd()->reorder_pds_.size();
        reorders_.resize(n);
        for (size_t i = 0; i < n; ++i)
            pd()->reorder_pds_[i]->create_primitive(&reorders_[i]);
    }

    ~ref_concat_t() {
        for (auto &r : reorders_)
            r->release();
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        const auto n = pd()->n_inputs();

        auto execute_reorder
                = [&](const primitive_t *reorder, const memory_arg_t &src,
                          const memory_arg_t &dst) {
                      exec_args_t r_args;
                      r_args[DNNL_ARG_SRC] = src;
                      r_args[DNNL_ARG_DST] = dst;
                      exec_ctx_t r_ctx(ctx, std::move(r_args));
                      reorder->execute(r_ctx);
                  };

        if (pd()->use_tent_dst()) {
            using namespace memory_tracking::names;
            auto scratchpad = ctx.get_scratchpad_grantor();
            auto tent_dst_storage
                    = scratchpad.get_memory_storage(key_concat_tent_dst);

            for (int i = 0; i < n; ++i) {
                memory_t tent_dst_i(engine(), pd()->src_image_md(i),
                        tent_dst_storage->clone(), false);
                execute_reorder(reorders_[i],
                        ctx.args().at(DNNL_ARG_MULTIPLE_SRC + i),
                        {&tent_dst_i, false});
            }

            memory_t tent_dst(pd()->engine(), &pd()->tent_dst_md_,
                    tent_dst_storage->clone(), false);
            execute_reorder(reorders_[n], {&tent_dst, true},
                    ctx.args().at(DNNL_ARG_DST));
        } else {
            auto &dst_mem_storage = CTX_OUT_STORAGE(DNNL_ARG_DST);
            for (int i = 0; i < n; ++i) {
                memory_t tent_dst_i(engine(), pd()->src_image_md(i),
                        dst_mem_storage.clone(), false);
                execute_reorder(reorders_[i],
                        ctx.args().at(DNNL_ARG_MULTIPLE_SRC + i),
                        {&tent_dst_i, false});
            }
        }

        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    std::vector<primitive_t *> reorders_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
