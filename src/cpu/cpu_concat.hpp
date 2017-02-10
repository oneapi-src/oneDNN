/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#ifndef CPU_CONCAT_HPP
#define CPU_CONCAT_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_primitive.hpp"
#include "event.hpp"
#include "memory_pd.hpp"
#include "reorder_pd.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_memory.hpp"

/* FIXME: tentative performance fix
 * The idea is that concat is simply a number of reorders.
 * The problem with this approach is omp parallelization: it is much better to
 * have one omp section with all the reorders, and not one omp section per the
 * reorder. `cpu_simple_concat.hpp addresses this problem, but the solution is
 * too stupid and isolated... */
#include "cpu_simple_concat.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;

struct cpu_concat_t: public cpu_primitive_t {
    struct pd_t: public concat_pd_t {
        pd_t(engine_t *engine, const memory_desc_t *output_d, int n,
                int concat_dim, const cpu_memory_t::pd_t **input_pds)
            : concat_pd_t(engine, n, concat_dim), dst_pd_(engine_)
        {
            for (int i = 0; i < n_; ++i)
                src_pds_.push_back(*input_pds[i]); /* make a copy */
            dst_pd_ = cpu_memory_t::pd_t(engine, output_d);
            if (output_d->format == memory_format::any) {
                /* the stupidest ever heuristics */
                memory_format_t out_fmt = output_d->format;
                for (int i = 0; i < n_; ++i)
                    out_fmt = nstl::max(out_fmt, src_pds_[i].desc()->format);
                dst_pd_.set_format(out_fmt); /* TODO: check status */
            } else {
                dst_pd_ = cpu_memory_t::pd_t(engine, output_d);
            }

            const int ndims = dst_pd_.desc()->ndims;
            int current_concat_dim_offset = 0;
            for (int i = 0; i < n_; ++i) {
                const int dim = src_pds_[i].desc()->dims[concat_dim];
                dims_t dims, offsets = {};
                utils::array_copy(dims, dst_pd_.desc()->dims, ndims);
                dims[concat_dim] = dim;
                offsets[concat_dim] = current_concat_dim_offset;

                cpu_view_t::pd_t v_pd(engine_, &dst_pd_, dims, offsets);
                src_image_pds_.push_back(*v_pd.dst_pd());

                current_concat_dim_offset += dim;
            }

            use_simple_concat_ =
                cpu_simple_concat_t<data_type::f32>::applicable(src_pds_,
                        src_image_pds_, concat_dim);
            if (use_simple_concat_) return;

            for (int i = 0; i < n_; ++i) {
                auto r_impls = engine_->get_reorder_implementation_list();
                for (auto r = r_impls; *r; ++r) {
                    reorder_pd_t *r_pd;
                    if ((*r)(&r_pd, &src_pds_[i], &src_image_pds_[i], 1.0, 0.0) ==
                            status::success) {
                        reorder_pds_.push_back(r_pd);
                        break;
                    }
                }
            }
        }
        pd_t(const pd_t &rhs)
            : concat_pd_t(rhs), use_simple_concat_(rhs.use_simple_concat_)
            , src_pds_(rhs.src_pds_), src_image_pds_(rhs.src_image_pds_)
            , dst_pd_(rhs.dst_pd_)
        {
            for (size_t i = 0; i < rhs.reorder_pds_.size(); ++i) {
                reorder_pds_.push_back(
                        (const reorder_pd_t *)rhs.reorder_pds_[i]->clone());
            }
        }

        virtual ~pd_t() {
            for (size_t i = 0; i < reorder_pds_.size(); ++i) {
                delete reorder_pds_[i];
            }
        }

        virtual pd_t *clone() const override { return nullptr; /* FIXME */ }
        virtual status_t create_primitive(primitive_t **primitive,
                const primitive_at_t *inputs, const primitive_t **outputs)
            const override
        {
            nstl::vector<primitive_t *> reorders;
            if (use_simple_concat_ == false) {
                reorders.resize(n_);
                for (int i = 0; i < n_; ++i)
                    CHECK(reorder_pds_[i]->create_primitive(&reorders[i],
                                &inputs[i], outputs));
            }

            primitive_t::input_vector ins(inputs, inputs + n_);
            primitive_t::output_vector outs(outputs, outputs + 1);
            return safe_ptr_assign<primitive_t>(*primitive,
                    new cpu_concat_t(this, ins, outs, reorders));
        }

        virtual const cpu_memory_t::pd_t *src_pd(int index = 0) const override
        { return index < this->n_ ? &src_pds_[index] : nullptr; }
        virtual const cpu_memory_t::pd_t *dst_pd(int index = 0) const override
        { return index == 0 ? &dst_pd_ : nullptr; }

        bool use_simple_concat_; /* FIXME: improve */
        nstl::vector<cpu_memory_t::pd_t> src_pds_;
        nstl::vector<cpu_memory_t::pd_t> src_image_pds_;
        nstl::vector<const reorder_pd_t *> reorder_pds_;
        cpu_memory_t::pd_t dst_pd_;
    };

    cpu_concat_t(const pd_t *conf, const input_vector &inputs,
            const output_vector &outputs,
            const nstl::vector<primitive_t *> &reorders)
        : cpu_primitive_t(&conf_, inputs, outputs)
        , conf_(*conf), reorders_(reorders) {}
    virtual ~cpu_concat_t() {
        for (size_t i = 0; i < reorders_.size(); ++i)
            delete reorders_[i];
    }

    virtual void execute(event_t *e) {
        if (conf_.use_simple_concat_) {
            cpu_simple_concat_t<data_type::f32>::execute(conf_.src_pds_,
                    conf_.src_image_pds_, this);
        } else {
            for (size_t i = 0; i < reorders_.size(); ++i) {
                event_t ei;
                reorders_[i]->execute(&ei);
            }
        }
        e->set_state(event_t::ready);
    }

private:
    pd_t conf_;
    nstl::vector<primitive_t *> reorders_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
