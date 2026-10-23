/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#ifndef CPU_X64_ZEN64_REORDER_ZEN_REORDER_HPP
#define CPU_X64_ZEN64_REORDER_ZEN_REORDER_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/memory_desc.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/reorder/cpu_reorder_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zen {
namespace reorder {

// ---------------------------------------------------------------------------
// Zen weight pre-pack / unpack reorder
// ---------------------------------------------------------------------------
// Dispatched only when one side uses the opaque `format_kind::zen_packed`
// format (see src/cpu/x64/zen64/common/zen_format_tag.hpp).
//
// Prepack (dst is zen_packed): bytes come from the Zen backend
// `reorder_direct` (prepack mode). The matmul backend consumes the buffer
// with mem_format_b='r'. A dtype change is applied to a plain copy of each
// slice before packing.
//
// Unpack (src is zen_packed): out-of-place inverse into a caller-provided
// dense plain buffer of K*N elements per slice (`ab`/`ba`, `abc`/`acb`),
// via `reorder_direct` in unreorder mode. f32, bf16, f16 and s8 (s8s8 and
// u8s8) only; a conversion between f32 and bf16/f16 is applied after
// unpacking.
//
// Packed to an identical packed descriptor is a byte copy.
struct zen_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("zen:reorder", zen_reorder_t);

        status_t init(const engine_t *engine, const engine_t *src_engine,
                const engine_t *dst_engine);

    private:
        status_t init_unpack(const engine_t *engine);
    };

    zen_reorder_t(const pd_t *apd) : primitive_t(apd) {}
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    status_t execute_unpack(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace reorder
} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
