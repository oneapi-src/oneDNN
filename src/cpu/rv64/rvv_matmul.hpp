/*******************************************************************************
* Copyright 2019 Intel Corporation
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
#ifndef CPU_RV64_RVV_MATMUL_HPP
#define CPU_RV64_RVV_MATMUL_HPP

#include <memory>

#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "cpu/matmul/cpu_matmul_pd.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/jit_uni_postops_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

struct rvv_matmul_t : public primitive_t {
    struct pd_t : public ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t {
        using ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa_, ""), rvv_matmul_t)

        // Bias is always f32: the f32 path uses it directly, the int8 path
        // converts it inside the JIT kernel before adding to the s32 acc.
        static constexpr data_type_t bias_d_type = data_type::f32;

        status_t init(const engine_t *engine) {
            UNUSED(engine);
            using smask_t = primitive_attr_t::skip_mask_t;
            using namespace data_type;

            // Vector kernels are JIT-emitted; the rv64gc baseline build means a
            // non-V CPU must defer to the next (reference) implementation.
            VDISPATCH_MATMUL(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);

            const memory_desc_wrapper src_mdw(src_md(0));
            const memory_desc_wrapper weights_mdw(weights_md(0));
            const memory_desc_wrapper dst_mdw(dst_md(0));
            const memory_desc_wrapper bias_mdw = bias_md_;

            VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            VDISPATCH_MATMUL(!src_mdw.has_runtime_dims_or_strides()
                            && !weights_mdw.has_runtime_dims_or_strides()
                            && !dst_mdw.has_runtime_dims_or_strides()
                            && !bias_mdw.has_runtime_dims_or_strides(),
                    VERBOSE_UNSUPPORTED_TAG);

            // Sparse memory descriptors (format_kind::sparse with a CSR/COO
            // encoding) carry no blocking descriptor; calling blocking_desc()
            // on them triggers an assertion in memory_desc_wrapper. The RV64
            // matmul kernel only handles dense memory, so let the dispatch
            // fall through to the reference matmul (which does support
            // sparse).
            VDISPATCH_MATMUL(!src_mdw.is_sparse_desc()
                            && !weights_mdw.is_sparse_desc()
                            && !dst_mdw.is_sparse_desc(),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            // Determine which dispatch path this pd serves. f32 stays on the
            // existing f32 GEMM kernel; (s8|u8) src + (s8|u8) weights + a dst
            // in the int8-supported set (s32|f32|s8|u8|f16|bf16) runs through
            // the int8 GEMM kernel. f16/bf16 src and weights with a dst of
            // the same type runs through the half-precision GEMM kernel
            // (f32 accumulation). f16 and bf16 additionally require Zvfh /
            // Zvfbfwma, which we gate below.
            const auto src_dt = src_mdw.data_type();
            const auto wei_dt = weights_mdw.data_type();
            const auto dst_dt = dst_mdw.data_type();
            const auto acc_dt = desc()->accum_data_type;
            is_f32_path_ = src_dt == f32 && wei_dt == f32 && dst_dt == f32
                    && acc_dt == f32;
            const bool dst_in_int8_set
                    = utils::one_of(dst_dt, s32, f32, s8, u8, f16, bf16);
            is_int8_path_ = utils::one_of(src_dt, s8, u8)
                    && utils::one_of(wei_dt, s8, u8) && dst_in_int8_set
                    && acc_dt == s32;
            is_hp_path_ = utils::one_of(src_dt, f16, bf16) && wei_dt == src_dt
                    && dst_dt == src_dt && acc_dt == f32;
            VDISPATCH_MATMUL(is_f32_path_ || is_int8_path_ || is_hp_path_,
                    VERBOSE_UNSUPPORTED_DT);
            // Half-precision dst requires the corresponding vector fp
            // extension. The CPU is probed once in init() so this is cheap.
            if (is_int8_path_ && dst_dt == f16) {
                VDISPATCH_MATMUL(mayiuse(zvfh), VERBOSE_UNSUPPORTED_ISA);
            }
            if (is_int8_path_ && dst_dt == bf16) {
                VDISPATCH_MATMUL(mayiuse(zvfbfwma), VERBOSE_UNSUPPORTED_ISA);
            }
            if (is_hp_path_) {
                VDISPATCH_MATMUL(mayiuse(src_dt == f16 ? zvfh : zvfbfwma),
                        VERBOSE_UNSUPPORTED_ISA);
                // The JIT currently implements the common f32 1xN bias. More
                // general matmul bias broadcasts fall back to another path.
                VDISPATCH_MATMUL(bias_mdw.is_zero()
                                || (bias_mdw.data_type() == data_type::f32
                                        && is_bias_1xN()),
                        VERBOSE_UNSUPPORTED_BIAS_CFG);
            }
            // The int8 path rejects per-oc / per-tensor scales, zero-points,
            // and post-ops in this MVP; only optional f32 bias is supported.
            const auto attr_skip_mask
                    = is_f32_path_ ? smask_t::post_ops : smask_t::none;
            VDISPATCH_MATMUL(attr()->has_default_values(attr_skip_mask, dst_dt),
                    VERBOSE_UNSUPPORTED_ATTR);

            // Resolve the primary and the post-op binary src1 formats before any
            // layout / post-op check: a post-op binary src1 may be format_any and
            // must be matched against dst before binary_broadcast_ok() inspects
            // its layout (x64 calls attr_.set_default_formats first as well).
            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);

            // Post-ops are applied per output row (length N) by the unified
            // injector kernel: a chain of supported eltwise ops plus any number
            // of binaries whose src1 is scalar or per-N (broadcast over M/batch).
            // The int8 path rejects post-ops entirely (caught above by the
            // stricter attr skip mask).
            if (is_f32_path_) {
                const dim_t N = weights_mdw.dims()[weights_mdw.ndims() - 1];
                VDISPATCH_MATMUL(jit_uni_postops_kernel_t::post_ops_supported(
                                         attr()->post_ops_, N),
                        VERBOSE_UNSUPPORTED_POSTOP);
                // A non-scalar binary rhs must be strict per-N (broadcast over
                // M and batch): the per-row execute uses a fixed offset, so
                // e.g. per-M [M,1] (which slips past nelems==N when M==N) must
                // fall back.
                VDISPATCH_MATMUL(
                        jit_uni_postops_kernel_t::binary_per_last_dim_ok(
                                attr()->post_ops_, N),
                        VERBOSE_UNSUPPORTED_POSTOP);
            }

            VDISPATCH_MATMUL(check_layouts(src_mdw, weights_mdw, dst_mdw),
                    VERBOSE_UNSUPPORTED_TAG);

            {
                const auto wei_ndims = weights_mdw.ndims();
                bool bc_ok = true;
                for (int i = 0; i < wei_ndims - 2; ++i) {
                    if (src_mdw.dims()[i] != weights_mdw.dims()[i]
                            && weights_mdw.dims()[i] != 1) {
                        bc_ok = false;
                        break;
                    }
                }
                VDISPATCH_MATMUL(bc_ok, VERBOSE_UNSUPPORTED_TAG);
            }

            VDISPATCH_MATMUL(check_bias(dst_mdw, bias_mdw),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);

            init_gemm_conf(src_mdw, weights_mdw);
            init_scratchpad();

            // Include half-precision output conversion on the int8 path.
            // This field describes the implementation; ISA checks stay above.
            isa_ = dst_dt == f16 ? zvfh : (dst_dt == bf16 ? zvfbfwma : v);

            return status::success;
        }

        bool is_row_major(const memory_desc_wrapper &mdw) const {
            const int ndims = mdw.ndims();
            if (ndims < 2) return false;

            const auto &strides = mdw.blocking_desc().strides;
            if (strides[ndims - 1] != 1) return false;

            dim_t expected_stride = mdw.dims()[ndims - 1];
            for (int d = ndims - 2; d >= 0; --d) {
                if (strides[d] != expected_stride) return false;
                expected_stride *= mdw.dims()[d];
            }
            return true;
        }

        bool is_col_major(const memory_desc_wrapper &mdw) const {
            const int ndims = mdw.ndims();
            if (ndims < 2) return false;

            const auto &strides = mdw.blocking_desc().strides;
            const auto &dims = mdw.dims();

            if (strides[ndims - 2] != 1) return false;
            if (strides[ndims - 1] != dims[ndims - 2]) return false;

            dim_t expected_stride = dims[ndims - 2] * dims[ndims - 1];
            for (int d = ndims - 3; d >= 0; --d) {
                if (strides[d] != expected_stride) return false;
                expected_stride *= dims[d];
            }
            return true;
        }

        bool check_layouts(const memory_desc_wrapper &src_mdw,
                const memory_desc_wrapper &wei_mdw,
                const memory_desc_wrapper &dst_mdw) const {
            if (!is_row_major(src_mdw) || !is_row_major(dst_mdw)) return false;
            if (!is_row_major(wei_mdw) && !is_col_major(wei_mdw)) return false;
            return true;
        }

        bool check_bias(const memory_desc_wrapper &dst_mdw,
                const memory_desc_wrapper &bias_mdw) const {
            if (bias_mdw.is_zero()) return true;

            if (bias_mdw.data_type() != bias_d_type) return false;

            const int dst_ndims = dst_mdw.ndims();
            const int bias_ndims = bias_mdw.ndims();
            if (bias_ndims > dst_ndims) return false;

            const auto *dst_dims = dst_mdw.dims();
            const auto *bias_dims = bias_mdw.dims();

            for (int d = 1; d <= bias_ndims; ++d) {
                const dim_t bias_dim = bias_dims[bias_ndims - d];
                const dim_t dst_dim = dst_dims[dst_ndims - d];
                if (bias_dim != 1 && bias_dim != dst_dim) return false;
            }

            // The int8 JIT kernel only knows how to read a 1-D per-N bias
            if (is_int8_path_) {
                for (int d = 2; d <= bias_ndims; ++d) {
                    if (bias_dims[bias_ndims - d] != 1) return false;
                }
            }
            return true;
        }

        void init_gemm_conf(const memory_desc_wrapper &src_mdw,
                const memory_desc_wrapper &weights_mdw) {
            const int ndims = src_mdw.ndims();
            const int wei_ndims = weights_mdw.ndims();

            const dim_t *src_dims = src_mdw.dims();
            const dim_t *wei_dims = weights_mdw.dims();

            batch_ = 1;
            for (int i = 0; i < ndims - 2; ++i)
                batch_ *= src_dims[i];

            M_ = src_dims[ndims - 2];
            K_ = src_dims[ndims - 1];
            N_ = wei_dims[wei_ndims - 1];
            weights_col_major_ = is_col_major(weights_mdw);

            dim_t weights_batch_size = 1;
            for (int i = 0; i < wei_ndims - 2; ++i)
                weights_batch_size *= wei_dims[i];
            weights_are_broadcast_ = (weights_batch_size == 1 && batch_ > 1);
        }

        dim_t M_ = 0;
        dim_t N_ = 0;
        dim_t K_ = 0;
        dim_t batch_ = 0;
        bool weights_are_broadcast_ = false;
        bool weights_col_major_ = false;
        // Dispatch path selected in init(). is_f32_path_ keeps the historical
        // behavior; is_int8_path_ routes (s8|u8):(s8|u8):(s32|f32|s8|u8|f16|bf16)
        // through the int8 GEMM kernel and rejects non-default attrs except
        // bias (and scales/zero-points/post-ops); is_hp_path_ routes
        // f16:f16:f16 / bf16:bf16:bf16 through the half-precision GEMM kernel
        // and rejects all attrs including bias.
        bool is_f32_path_ = false;
        bool is_int8_path_ = false;
        bool is_hp_path_ = false;
        cpu_isa_t isa_ = isa_undef;

        int nthr_m_ = 1;
        int nthr_n_ = 1;
        int nthr_k_ = 1;
        dim_t MB_ = 0;
        dim_t NB_ = 0;
        dim_t KB_ = 0;
        dim_t m_unroll_ = 0;
        bool do_copy_ = false;
        // A-copy workspace booked in the scratchpad: number of per-thread
        // slices (0 = not booked) and the byte size of one slice. Covers both
        // the full GEMM partition and the parallel batch loop in execute().
        int ws_thr_slices_ = 0;
        size_t ws_slice_bytes_ = 0;

    private:
        void init_scratchpad();
    };

    rvv_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    // Per-row "bias + post-op chain" kernel (shared gemm post-op epilogue).
    std::shared_ptr<jit_uni_postops_kernel_t> postops_kernel_;
};

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_MATMUL_HPP
