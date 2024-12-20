/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_OCL_MICRO_GATED_MLP_HPP
#define GPU_OCL_MICRO_GATED_MLP_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/gemm_types.hpp"
#include "common/gemm_utils.hpp"
#include "common/primitive.hpp"
#include "common/gated_mlp_pd.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/microkernels/shim.hpp"
#include "gpu/intel/ocl/ocl_utils.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct micro_gated_mlp_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;

    struct pd_t : public gated_mlp_pd_t {
        using gated_mlp_pd_t::gated_mlp_pd_t;
        static constexpr int mask_mb_indes = 0; //TODO: fix typo
        static constexpr int mask_q_index = 2;
        static constexpr int mask_k_index = 3;

        DECLARE_COMMON_PD_T("ocl:micro:any", micro_gated_mlp_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            VDISPATCH_GATED_MLP(utils::everyone_is(2, src_md()->ndims,
                                        W_gate_md()->ndims, W_up_md()->ndims,
                                        W_down_md()->ndims, dst_md()->ndims),
                    VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_GATED_MLP(utils::everyone_is(data_type::f16,
                                   src_md()->data_type,
                                   W_gate_md()->data_type, W_up_md()->data_type,
                                   W_down_md()->data_type, dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_GATED_MLP(set_default_formats() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);

            CHECK(init_microkernels(engine));
            init_scratchpad();
            return status::success;
        }

        status_t set_default_format(memory_desc_t &md, bool allow_transpose) {
            using namespace format_tag;
            memory_desc_wrapper mdw(md);
            if (mdw.format_any()) return status::unimplemented;
            if (!is_md_gemm_compatible_plain_format(&md))
                return status::unimplemented;
            if (gemm_desc_t::get_trans(md) == dnnl_trans && !allow_transpose)
                return status::unimplemented;
            return status::success;
        }

        status_t set_default_formats() {
            CHECK(set_default_format(desc_.src_desc, false));
            CHECK(set_default_format(desc_.W_gate_desc, false));
            CHECK(set_default_format(desc_.W_up_desc, false));
            CHECK(set_default_format(desc_.W_down_desc, false));
            CHECK(set_default_format(desc_.dst_desc, false));
            return status::success;
        }

        const micro::Package &gemm_gateup() const { return gemm_gateup_; }
        const micro::Package &gemm_down() const { return gemm_down_; }

        int sg_size() const { return sg_size_; }

        // Block size for head_size, which must be hard-coded into the kernel.
        // TODO: can?/does this need to be hardcoded for gated mlp?
        int b_max() const {
            int b_size = desc()->mb_sz();
            for (int i = 32; i <= 1024; i *= 2)
                if (b_size <= i) return i;
            return b_size;
        }
        compute::gpu_arch_t arch() const { return arch_; }

    private:
        micro::Package gemm_gateup_, gemm_down_;
        int sg_size_ = 0;
        compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;

        status_t init_microkernels(impl::engine_t *engine);
        void init_scratchpad();
    };

    status_t init(impl::engine_t *engine) override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute(const exec_ctx_t &ctx) const override;

    compute::kernel_t fused_mlp_kernel_, fc_down_mm_kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
