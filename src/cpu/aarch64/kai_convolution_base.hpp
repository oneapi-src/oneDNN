/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_KAI_CONVOLUTION_BASE_HPP
#define CPU_AARCH64_KAI_CONVOLUTION_BASE_HPP

#include <cstddef>
#include <memory>
#include <string>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"

#include "cpu/aarch64/post_ops_fallback.hpp"
#include "cpu/cpu_convolution_pd.hpp"

namespace kai {
namespace ops {
// NOLINTBEGIN(readability-identifier-naming)
struct GemmArgs;
class IGemmCommon;
// NOLINTEND(readability-identifier-naming)
} // namespace ops
} // namespace kai

namespace dnnl {
namespace impl {
namespace memory_tracking {
struct grantor_t;
struct registrar_t;
} // namespace memory_tracking
namespace cpu {
namespace aarch64 {

struct kai_convolution_fwd_base_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const op_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd)
            , name_("unknown:kai") {}

        status_t init(const engine_t *engine);
        std::unique_ptr<kai::ops::IGemmCommon> create_kai_gemm() const;
        int kernel_maxthreads() const;
        bool fixed_format() const;

        bool run_weight_reorder_ = false;
        data_type_t gemm_weights_dt_ = data_type::undef;
        data_type_t gemm_dst_dt_ = data_type::undef;
        bool use_dst_reorder_ = false;
        bool has_post_ops_fallback_ = false;
        post_ops_fallback_t post_ops;
        memory_desc_t tmp_dst_md_;
        std::shared_ptr<primitive_desc_t> dst_reorder_pd_;

    protected:
        virtual const char *impl_base_name() const = 0;
        virtual bool uses_indirect_gemm() const { return false; }
        virtual status_t init_datapath(const engine_t *) {
            return status::success;
        }
        virtual void book_datapath_scratchpad(
                memory_tracking::registrar_t &scratchpad,
                size_t src_dt_size) const {}
        virtual unsigned int gemm_m() const;
        virtual unsigned int gemm_k() const;
        virtual unsigned int gemm_k_sections() const { return 1; }
        virtual unsigned int gemm_n_batches() const { return 1; }
        virtual unsigned int gemm_n_multi() const { return 1; }
        bool direct_1x1_kernel_ok() const;
        bool direct_1x1_padding_ok() const;
        bool direct_1x1_output_samples_in_bounds() const;

        std::string name_;

    private:
        std::shared_ptr<kai::ops::GemmArgs> args_ = nullptr;
    };

    kai_convolution_fwd_base_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

protected:
    struct kernel_call_args_t {
        const exec_ctx_t &ctx;
        const pd_t &pd;
        kai::ops::IGemmCommon &kernel;
        const memory_tracking::grantor_t &scratchpad;
        int max_threads;
        int num_threads;
        const char *src_base;
        void *wei_base;
        void *kernel_dst_base;
        const void *bias_base;
        int ld_src;
        int ld_wei;
        int ld_dst;
        int src_h_stride;
        int src_batch_stride;
        int dst_h_stride;
        int dst_batch_stride;
        size_t src_dt_size;
        size_t src_col_stride_bytes;
        size_t src_h_stride_bytes;
        size_t src_batch_stride_bytes;
    };

    virtual status_t setup_kernel_arrays(const kernel_call_args_t &args) const
            = 0;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

private:
    std::shared_ptr<impl::primitive_t> dst_reorder_;
    post_ops_fallback_t post_ops_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
