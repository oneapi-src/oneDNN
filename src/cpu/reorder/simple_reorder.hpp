/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef CPU_REORDER_SIMPLE_REORDER_HPP
#define CPU_REORDER_SIMPLE_REORDER_HPP

#include "cpu/reorder/cpu_reorder_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct simple_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("simple:any", simple_reorder_t);

        status_t init(const engine_t *engine, const engine_t *src_engine,
                const engine_t *dst_engine);

        enum class kernel_kind_t {
            unknown,
            // Copies tensors whose src/dst layouts are identical for every
            // dimension but the outermost one, which may carry a stride gap.
            direct_copy,
            // Copies between a plain layout and a blocked layout whose inner
            // blocks cover one or two distinct logical dims (e.g. nchw <->
            // nChw16c, oihw <-> OIhw16i16o, goihw <-> gOIhw16i16o).
            plain_to_blocked,
            // Re-blocks a single blocked logical dim (channels) between two
            // blocked layouts with different block sizes (e.g. nChw8c <->
            // nChw16c).
            // TODO: remove this kernel kind, along with its `is_*`/`execute_*`
            // helpers and dispatch wiring, once blocked activation layouts are
            // dropped from the library.
            blocked_to_blocked,
            // Copies a plain layout into a dense (plain or blocked), s8
            // convolution-weights layout and fills the appended convolution
            // compensation buffer(s) (s8s8 and/or asymmetric-src zero-point).
            // The data movement is a plain->dense copy; the compensation is
            // reduced per output channel (the masked logical dims).
            plain_to_dense_comp,
        };

        kernel_kind_t kernel_kind_ = kernel_kind_t::unknown;

    private:
        // Each `is_*` helper performs the applicability check for one kernel
        // kind. It returns `status::success` when the given problem can be
        // served by the matching kernel kind.
        status_t is_direct_copy(const engine_t *engine,
                const memory_desc_wrapper &input_d,
                const memory_desc_wrapper &output_d,
                const primitive_attr_t *attr);
        status_t is_plain_to_blocked(const engine_t *engine,
                const memory_desc_wrapper &input_d,
                const memory_desc_wrapper &output_d,
                const primitive_attr_t *attr);
        status_t is_blocked_to_blocked(const engine_t *engine,
                const memory_desc_wrapper &input_d,
                const memory_desc_wrapper &output_d,
                const primitive_attr_t *attr);
        status_t is_plain_to_dense_comp(const engine_t *engine,
                const memory_desc_wrapper &input_d,
                const memory_desc_wrapper &output_d,
                const primitive_attr_t *attr);
    };

    simple_reorder_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    status_t execute_direct_copy(const exec_ctx_t &ctx) const;
    status_t execute_plain_to_blocked(const exec_ctx_t &ctx) const;
    status_t execute_blocked_to_blocked(const exec_ctx_t &ctx) const;
    status_t execute_plain_to_dense_comp(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
