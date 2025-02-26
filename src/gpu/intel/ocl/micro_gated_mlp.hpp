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
#include "gpu/intel/gemm/gpu_gemm.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/microkernels/shim.hpp"
#include "gpu/intel/ocl/utils.hpp"
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
                                   src_md()->data_type, dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_GATED_MLP(
                    utils::one_of(W_gate_md()->data_type, f16, u8, s8, u4, s4),
                    VERBOSE_UNSUPPORTED_DT);
            //TODO: enable unique types for all gates?
            //VDISPATCH_GATED_MLP(utils::everyone_is(W_gate_md()->data_type,
                                   //W_up_md()->data_type, W_down_md()->data_type),
                    //VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_GATED_MLP(set_default_formats() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);

            CHECK(init_microkernels(engine));

//            // initialize fc_down gemm descriptor and primitive
//            auto *d = desc();
//            // auto fpmath_mode = this->attr()->fpmath_.mode_; //TODO: use as part of attr?
//            int threads_per_eu = 0;
//
//            auto create_gemm_pd =
//                    [&](std::shared_ptr<primitive_desc_t> &gemm_pd, dim_t m, dim_t n,
//                            dim_t k, strides_t<2> a_strides, strides_t<2> b_strides,
//                            strides_t<2> c_strides, data_type_t a_dt, data_type_t b_dt,
//                            data_type_t c_dt, float beta) -> status_t {
//
//                memory_desc_t a_md, b_md, c_md;
//                dims_t a_dims = {m, k};
//                dims_t a_strides_md = {m, 1}; //transpose??
//                //dims_t b_strides_md = {b_strides[0], b_strides[1]};
//                CHECK(memory_desc_init_by_strides(a_md, 2, a_dims, a_dt, a_strides_md));
//
//                // dims_t a_strides_md = {a_strides[0], a_strides[1]};
//                // CHECK(memory_desc_init_by_strides(a_md, 2, a_dims, a_dt, a_strides_md));
//
//                dims_t c_dims = {m, n};
//                dims_t c_strides_md = {c_strides[0], c_strides[1]};
//                CHECK(memory_desc_init_by_strides(c_md, 2, c_dims, c_dt, c_strides_md));
//
//                /*
//            printf("a%d %d \n b%d %d \n c%d %d\n",
//                    a_md.dims[0],
//                    a_md.dims[1],
//                    W_down_md()->dims[0],
//                    W_down_md()->dims[1],
//                    c_md.dims[0],
//                    c_md.dims[1]);
//                    */
//
//                primitive_attr_t attr;
//                //CHECK(attr.post_ops_.append_sum(beta));
//                //CHECK(attr.set_fpmath_mode(fpmath_mode));
//                //attr.deterministic_ = this->attr()->deterministic_;
//                CHECK(dnnl::impl::create_gemm_pd(gemm_pd, engine, &a_md, W_down_md(), &c_md,
//                        &glob_zero_md, c_dt, &attr));
//
//                if (threads_per_eu == 0)
//                    CHECK(gemm_pd->query(
//                            query::preferred_gpu_threads_per_eu, 0, &threads_per_eu));
//                else if (get_verbose_dev_mode(verbose_t::debuginfo) > 1) {
//                    auto t = 0;
//                    CHECK(gemm_pd->query(query::preferred_gpu_threads_per_eu, 0, &t));
//                    if (t != threads_per_eu)
//                        verbose_printf("[WARNING] GEMM grf modes are inconsistent");
//                }
//                return status::success;
//            };
//
            /*
            VDISPATCH_GATED_MLP_SC(
                create_gemm_pd(
                        gemm_fc_down_pd_,
                        d->mb_sz(), d->ic_sz(), d->oc_sz(),
                        //{1, d->mb_sz()}, // a strides for scratchpad mem
                        {32, 1}, // a strides for scratchpad mem
                        {32, 1}, // bstride irrelevant
                        //{d->ic_sz(), 1},
                        {1, d->ic_sz()},
                        data_type::f16,
                        W_down_md()->data_type,
                        dst_md()->data_type,
                        0.f),
                "create_gemm_pd(gemm_fc_down_pd_)");

            */
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

        int sg_size() const { return sg_size_; }

        compute::gpu_arch_t arch() const { return arch_; }

        std::shared_ptr<primitive_desc_t> gemm_fc_down_pd_;
    private:
        micro::Package gemm_gateup_;
        int sg_size_ = 0;
        compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;

        status_t init_microkernels(impl::engine_t *engine);
        void init_scratchpad();
    };

    status_t init(impl::engine_t *engine) override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute(const exec_ctx_t &ctx) const override;

    compute::kernel_t fused_mlp_kernel_;

    std::shared_ptr<impl::primitive_t> gemm_fc_down_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
