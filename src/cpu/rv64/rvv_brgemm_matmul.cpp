/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#include <vector>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/binary_injector_utils.hpp"
#include "cpu/platform.hpp"
#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/rvv_brgemm_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace data_type;

// ---------------------------------------------------------------------------
// JIT kernel: pack_a_tile
// Copies valid_rows floats per column from col-major A (stride LDA_orig)
// into contiguous workspace (stride bd). Vectorized with LMUL=m4.
// ---------------------------------------------------------------------------
struct jit_pack_a_tile_t : public jit_generator_t {
    struct call_params_t {
        void *ws; // offset 0
        const void *A; // offset 8
        dim_t LDA_orig; // offset 16 — in elements (not bytes)
        dim_t bd; // offset 24 — in elements
        dim_t valid_rows; // offset 32
        dim_t K_inner; // offset 40
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_pack_a_tile_t)

    // input_typesize: 4 for f32, 2 for bf16/f16, 1 for int8 (widened to s32
    // during packing so the brgemm kernel can use plain e32 ops).
    explicit jit_pack_a_tile_t(int input_typesize)
        : jit_generator_t("jit_pack_a_tile"), input_typesize_(input_typesize) {
        assert(input_typesize == 1 || input_typesize == 2
                || input_typesize == 4);
        create_kernel();
    }

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
        using namespace Xbyak_riscv;

        const Reg reg_param = a0;
        const Reg reg_ws = a1;
        const Reg reg_A = a2;
        const Reg reg_LDA = t0;
        const Reg reg_bd = t1;
        const Reg reg_K = t2;
        const Reg reg_k = t3; // outer loop counter
        const Reg reg_src = t4;
        const Reg reg_dst = t5;
        const Reg reg_rows_remaining = t6;
        const Reg reg_vl = a3;
        const Reg reg_bytes = a4;
        const Reg reg_tmp = a5;

        const VReg v_tmp(0);

        const int elem_shift = (input_typesize_ == 4) ? 2
                : (input_typesize_ == 2)              ? 1
                                                      : 0;

        // Load parameters.
        ld(reg_ws, reg_param, 0);
        ld(reg_A, reg_param, 8);
        ld(reg_LDA, reg_param, 16);
        ld(reg_bd, reg_param, 24);
        ld(reg_K, reg_param, 40);

        slli(reg_LDA, reg_LDA, elem_shift);
        slli(reg_bd, reg_bd, elem_shift);

        const Reg reg_valid_rows = a6;
        ld(reg_valid_rows, reg_param, 32);

        if (input_typesize_ == 1) {
            const Reg reg_bidx = a7;
            xor_(reg_k, reg_k, reg_k);
            Label k_loop_i8, k_done_i8;
            L(k_loop_i8);
            beq(reg_k, reg_K, k_done_i8);

            mul(reg_tmp, reg_k, reg_LDA);
            add(reg_src, reg_A, reg_tmp);
            mul(reg_tmp, reg_k, reg_bd);
            slli(reg_tmp, reg_tmp, 2); // ×4 for s32 stride
            add(reg_dst, reg_ws, reg_tmp);

            xor_(reg_bidx, reg_bidx, reg_bidx);
            Label row_loop, row_done;
            L(row_loop);
            beq(reg_bidx, reg_valid_rows, row_done);
            lb(reg_tmp, reg_src, 0);
            sw(reg_tmp, reg_dst, 0);
            addi(reg_src, reg_src, 1);
            addi(reg_dst, reg_dst, 4);
            addi(reg_bidx, reg_bidx, 1);
            j_(row_loop);
            L(row_done);

            addi(reg_k, reg_k, 1);
            j_(k_loop_i8);
            L(k_done_i8);
            ret();
        }

        xor_(reg_k, reg_k, reg_k); // k = 0

        Label k_loop, k_done;
        L(k_loop);
        beq(reg_k, reg_K, k_done);

        mul(reg_tmp, reg_k, reg_LDA);
        add(reg_src, reg_A, reg_tmp);

        mul(reg_tmp, reg_k, reg_bd);
        add(reg_dst, reg_ws, reg_tmp);

        mv(reg_rows_remaining, reg_valid_rows);
        Label copy_loop, copy_done;
        L(copy_loop);
        beqz(reg_rows_remaining, copy_done);

        if (input_typesize_ == 4) {
            vsetvli(reg_vl, reg_rows_remaining, SEW::e32, LMUL::m4, VTA::ta,
                    VMA::ma);
            vle32_v(v_tmp, reg_src);
            vse32_v(v_tmp, reg_dst);
        } else {
            vsetvli(reg_vl, reg_rows_remaining, SEW::e16, LMUL::m4, VTA::ta,
                    VMA::ma);
            vle16_v(v_tmp, reg_src);
            vse16_v(v_tmp, reg_dst);
        }

        slli(reg_bytes, reg_vl, elem_shift);
        add(reg_src, reg_src, reg_bytes);
        add(reg_dst, reg_dst, reg_bytes);
        sub(reg_rows_remaining, reg_rows_remaining, reg_vl);
        j_(copy_loop);

        L(copy_done);

        addi(reg_k, reg_k, 1);
        j_(k_loop);

        L(k_done);
        ret();
#else
        ret();
#endif
    }

private:
    int input_typesize_;
};

status_t rvv_brgemm_matmul_t::pd_t::init(const engine_t *engine) {
    using smask_t = primitive_attr_t::skip_mask_t;

    VDISPATCH_MATMUL(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);

    VDISPATCH_MATMUL(dnnl_get_max_threads() <= 1, VERBOSE_IMPL_HEURISTIC_FAIL,
            "brgemm matmul is single-thread only");

    VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

    const memory_desc_wrapper src_mdw(src_md(0));
    const memory_desc_wrapper wei_mdw(weights_md(0));
    const memory_desc_wrapper dst_mdw(dst_md(0));
    const memory_desc_wrapper bias_mdw = bias_md_;

    VDISPATCH_MATMUL(!src_mdw.has_runtime_dims_or_strides()
                    && !wei_mdw.has_runtime_dims_or_strides()
                    && !dst_mdw.has_runtime_dims_or_strides()
                    && !bias_mdw.has_runtime_dims_or_strides(),
            VERBOSE_UNSUPPORTED_TAG);

    // Sparse memory descriptors (format_kind::sparse with a CSR/COO encoding)
    // carry no blocking descriptor; calling blocking_desc() on them triggers an
    // assertion in memory_desc_wrapper. The RV64 matmul kernel only handles
    // dense memory, so let the dispatch fall through to the reference matmul
    // (which does support sparse).
    VDISPATCH_MATMUL(!src_mdw.is_sparse_desc() && !wei_mdw.is_sparse_desc()
                    && !dst_mdw.is_sparse_desc(),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // Accepted: f32/f32/f32, bf16/bf16/f32 (Zvfbfwma), f16/f16/f32 (Zvfh),
    //           s8/s8/s32. u8 / mixed-sign are rejected at brgemm_desc_init.
    const auto src_dt = src_mdw.data_type();
    const auto wei_dt = wei_mdw.data_type();
    const bool same_in_dt = src_dt == wei_dt;
    // Derive the kernel ISA from the input dtype now, before the dtype/ISA gate
    // below, so a declined bf16/f16 PD reports brgemm:rvv_zvfbfwma / _zvfh in the
    // dispatch log instead of the default brgemm:rvv.
    isa_ = (src_dt == f16) ? zvfh : (src_dt == bf16) ? zvfbfwma : v;
    const bool in_dt_ok = same_in_dt
            && (src_dt == f32 || (src_dt == bf16 && mayiuse(zvfbfwma))
                    || (src_dt == f16 && mayiuse(zvfh)));
    const bool in_dt_ok_int8 = (src_dt == s8 && wei_dt == s8);
    const bool types_ok = (in_dt_ok || in_dt_ok_int8)
            && IMPLICATION(in_dt_ok,
                    dst_mdw.data_type() == f32
                            && desc()->accum_data_type == f32)
            && IMPLICATION(in_dt_ok_int8,
                    dst_mdw.data_type() == s32
                            && desc()->accum_data_type == s32)
            // int8 path rejects any bias explicitly below.
            && IMPLICATION(in_dt_ok && !bias_mdw.is_zero(),
                    bias_mdw.data_type() == f32);
    VDISPATCH_MATMUL(types_ok, VERBOSE_UNSUPPORTED_DT);

    input_typesize_ = static_cast<int>(types::data_type_size(src_dt));

    // int8 path supports no bias / post-ops / scales / zero-points yet.
    if (in_dt_ok_int8) {
        VDISPATCH_MATMUL(bias_mdw.is_zero(), VERBOSE_UNSUPPORTED_BIAS_CFG);
        VDISPATCH_MATMUL(
                attr()->has_default_values(smask_t::post_ops | smask_t::scales
                                | smask_t::zero_points,
                        s32),
                VERBOSE_UNSUPPORTED_ATTR);
    } else {
        VDISPATCH_MATMUL(attr()->has_default_values(smask_t::post_ops, f32),
                VERBOSE_UNSUPPORTED_ATTR);
    }

    // Resolve primary + post-op binary src1 formats before the post-op check:
    // a post-op binary src1 may be format_any and must be matched to dst before
    // binary_broadcast_ok() inspects its layout (matches x64).
    VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL(attr_.set_default_formats(dst_md(0)) == status::success,
            VERBOSE_UNSUPPORTED_POSTOP);

    // Post-ops applied per output row (length N) by the unified injector
    // kernel: supported eltwise chain + any number of scalar/per-N binaries.
    const dim_t N_po = dst_mdw.dims()[dst_mdw.ndims() - 1];
    VDISPATCH_MATMUL(jit_uni_postops_kernel_t::post_ops_supported(
                             attr()->post_ops_, N_po),
            VERBOSE_UNSUPPORTED_POSTOP);
    // A non-scalar binary rhs must be strict per-N (broadcast over M and batch);
    // per-M [M,1] (slips past nelems==N when M==N) must fall back.
    VDISPATCH_MATMUL(jit_uni_postops_kernel_t::binary_per_last_dim_ok(
                             attr()->post_ops_, N_po),
            VERBOSE_UNSUPPORTED_POSTOP);

    const int ndims = src_mdw.ndims();
    const int wei_ndims = wei_mdw.ndims();

    // Plain dense tensors
    VDISPATCH_MATMUL(src_mdw.blocking_desc().inner_nblks == 0
                    && wei_mdw.blocking_desc().inner_nblks == 0
                    && dst_mdw.blocking_desc().inner_nblks == 0,
            VERBOSE_UNSUPPORTED_TAG);

    // All tensors must be dense row-major (full stride chain verified).
    // Weights must be row-major; col-major would require transpose which
    // is not yet supported with copy_A packing.
    VDISPATCH_MATMUL(is_row_major(src_mdw), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL(is_row_major(dst_mdw), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL(is_row_major(wei_mdw), VERBOSE_UNSUPPORTED_TAG);

    // Check bias
    if (!bias_mdw.is_zero()) {
        VDISPATCH_MATMUL(bias_mdw.data_type() == f32, VERBOSE_UNSUPPORTED_DT);
        const int bias_ndims = bias_mdw.ndims();
        const auto *bias_dims = bias_mdw.dims();
        const auto *dst_dims = dst_mdw.dims();
        const int dst_ndims = dst_mdw.ndims();
        VDISPATCH_MATMUL(bias_ndims <= dst_ndims, VERBOSE_UNSUPPORTED_BIAS_CFG);
        for (int d = 1; d <= bias_ndims; ++d) {
            dim_t bias_dim = bias_dims[bias_ndims - d];
            dim_t dst_dim = dst_dims[dst_ndims - d];
            VDISPATCH_MATMUL(bias_dim == 1 || bias_dim == dst_dim,
                    VERBOSE_UNSUPPORTED_BIAS_CFG);
        }
    }

    // Check weight broadcast
    {
        bool bc_ok = true;
        for (int i = 0; i < wei_ndims - 2; ++i) {
            if (src_mdw.dims()[i] != wei_mdw.dims()[i]
                    && wei_mdw.dims()[i] != 1) {
                bc_ok = false;
                break;
            }
        }
        VDISPATCH_MATMUL(bc_ok, VERBOSE_UNSUPPORTED_TAG);
    }

    // Extract dimensions (same as rvv_matmul.cpp)
    const dim_t *src_dims = src_mdw.dims();
    const dim_t *wei_dims = wei_mdw.dims();

    batch_ = 1;
    for (int i = 0; i < ndims - 2; ++i)
        batch_ *= src_dims[i];

    M_ = src_dims[ndims - 2];
    K_ = src_dims[ndims - 1];
    N_ = wei_dims[wei_ndims - 1];

    dim_t weights_batch_size = 1;
    for (int i = 0; i < wei_ndims - 2; ++i)
        weights_batch_size *= wei_dims[i];
    weights_are_broadcast_ = (weights_batch_size == 1 && batch_ > 1);

    // Shape guards
    VDISPATCH_MATMUL(weights_are_broadcast_, VERBOSE_IMPL_HEURISTIC_FAIL,
            "weights are not broadcast across batch");
    VDISPATCH_MATMUL(
            N_ >= 16, VERBOSE_IMPL_HEURISTIC_FAIL, "N too small for brgemm");

    // f32 keeps the K/A_bytes thresholds; bf16/f16/int8 only require batch*M.
    const bool is_low_prec
            = (src_dt == data_type::bf16 || src_dt == data_type::f16);
    const bool is_int8 = (src_dt == s8);
    if (is_low_prec || is_int8) {
        VDISPATCH_MATMUL(K_ >= BRGEMM_BK && batch_ * M_ >= 128,
                VERBOSE_IMPL_HEURISTIC_FAIL,
                "shape too small for bf16/f16/int8 brgemm matmul");
    } else {
        const dim_t A_bytes = N_ * K_ * (dim_t)input_typesize_;
        const auto L2_bytes = platform::get_per_core_cache_size(3);
        VDISPATCH_MATMUL(K_ >= 4096 && A_bytes > L2_bytes && batch_ * M_ >= 128,
                VERBOSE_IMPL_HEURISTIC_FAIL,
                "shape not beneficial for f32 brgemm matmul");
    }

    // Compute blocking parameters
    const int vlen_f32 = get_platform_vlen() / 32;
    const int bd_block = vlen_f32 * 4; // LMUL=m4
    const dim_t M_brg = N_;
    const dim_t K_brg = K_;
    const dim_t LDA = bd_block; // packed stride (not N)
    const dim_t LDB = K_;
    const dim_t LDC = N_;
    const dim_t N_brg = batch_ * M_;

    cpu_isa_t brg_isa = src_dt == bf16 ? zvfbfwma : (src_dt == f16 ? zvfh : v);

    brgemm_desc_t brg_desc;
    CHECK(brgemm_desc_init(&brg_desc, brg_isa, brgemm_strd, src_dt, wei_dt,
            brgemm_col_major, 1.0f, 0.0f, LDA, LDB, LDC, M_brg, N_brg, K_brg));

    brgemm_kernel_t *kernel = nullptr;
    CHECK(brgemm_kernel_create(&kernel, brg_desc));
    brg_kernel_.reset(kernel);

    // pack_a_tile is dtype-aware; the bias+postops chain touches the dst and
    // is only built for the non-int8 path (int8 has no bias/postops yet).
    pack_kernel_.reset(new jit_pack_a_tile_t(input_typesize_));
    if (!in_dt_ok_int8) {
        const memory_desc_wrapper bias_d(desc()->bias_desc);
        jit_uni_postops_kernel_t::conf_t conf;
        conf.dst_dt = f32;
        conf.with_bias = !bias_d.is_zero();
        if (conf.with_bias) {
            const int bn = bias_d.ndims();
            conf.bias_per_element
                    = !(bias_d.nelems() == 1 || bias_d.dims()[bn - 1] == 1);
        }
        CHECK(jit_uni_postops_kernel_t::create(
                postops_kernel_, attr()->post_ops_, conf));
    }

    init_scratchpad();

    return status::success;
}

void rvv_brgemm_matmul_t::pd_t::init_scratchpad() {
    using namespace memory_tracking::names;
    const auto &brg = brg_kernel_->get_brg();
    auto scratchpad = scratchpad_registry().registrar();
    // int8 A is pre-widened to s32 during packing (4 bytes/elem).
    const int ws_typesize = (input_typesize_ == 1) ? 4 : input_typesize_;
    const size_t ws_bytes = (size_t)brg.bd_block * K_ * ws_typesize;
    scratchpad.template book<char>(key_brgemm_primitive_buffer_a, ws_bytes);
}

status_t rvv_brgemm_matmul_t::execute(const exec_ctx_t &ctx) const {
    // Byte arithmetic so one code path handles f32/bf16/f16/int8.
    auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    auto dst_char = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const dim_t M = pd()->M_;
    const dim_t N = pd()->N_;
    const dim_t K = pd()->K_;
    const dim_t batch = pd()->batch_;
    const int in_ts = pd()->input_typesize_;
    const bool is_int8 = pd()->brg_kernel_->get_brg().is_int8;

    const auto &brg = pd()->brg_kernel_->get_brg();
    const int bd = brg.bd_block;
    const int bdb = brg.bdb;
    const int bdb_tail = brg.bdb_tail;
    const dim_t total_N = batch * M;
    const dim_t LDA_orig = N; // row-major weights: original col-major LDA = N

    const auto *brg_kernel = pd()->brg_kernel_.get();
    const auto *pack_kernel = pd()->pack_kernel_.get();
    const auto *postops_kernel = pd()->postops_kernel_.get();

    // Packing workspace from scratchpad. Packed once per M-tile, then the
    // brgemm kernel is called per K-block with offset into the buffer.
    const dim_t BK = BRGEMM_BK;
    auto &grantor = ctx.get_scratchpad_grantor();
    char *ws = grantor.template get<char>(
            memory_tracking::names::key_brgemm_primitive_buffer_a);

    const int num_tiles = bdb + (bdb_tail > 0 ? 1 : 0);
    // int8 A is pre-widened to s32 inside the packing kernel (4 bytes/elem).
    const int ws_typesize = (in_ts == 1) ? 4 : in_ts;
    const int dst_typesize = brg.typesize_C; // 4 for both f32 and s32 dst
    for (int t = 0; t < num_tiles; t++) {
        const bool is_tail = (t == bdb);
        const int rows = is_tail ? bdb_tail : bd;
        // Tile start in the unpacked weights tensor (in_ts bytes/elem).
        const char *A_tile = weights + (dim_t)t * bd * in_ts;

        jit_pack_a_tile_t::call_params_t pack_p;
        pack_p.ws = ws;
        pack_p.A = A_tile;
        pack_p.LDA_orig = LDA_orig;
        pack_p.bd = bd;
        pack_p.valid_rows = rows;
        pack_p.K_inner = K;
        (*pack_kernel)(&pack_p);

        for (dim_t kb = 0; kb < K; kb += BK) {
            const dim_t K_inner = nstl::min(BK, K - kb);
            const float beta_kb = (kb == 0) ? 0.0f : 1.0f;

            brgemm_kernel_params_t p;
            p.ptr_A = ws + kb * bd * ws_typesize;
            p.ptr_B = src + kb * in_ts;
            p.ptr_C = dst_char + (dim_t)t * bd * dst_typesize;
            p.N = total_N;
            p.M = rows;
            p.K = K_inner;
            p.beta = beta_kb;
            p.ptr_bias = nullptr;
            (*brg_kernel)(&p);
        }
    }

    // int8 path: no bias/post-ops yet.
    if (is_int8) return status::success;

    // Beyond here dst is f32.
    auto dst = reinterpret_cast<float *>(dst_char);

    // Apply bias + post-ops using JIT kernel
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->desc()->bias_desc);
    const float *bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
    const post_ops_t &post_ops = pd()->attr()->post_ops_;

    if (!bias && post_ops.len() == 0) return status::success;

    const int dst_ndims = dst_d.ndims();
    const int bias_ndims = bias_d.ndims();
    const dim_t *bias_dims = bias_d.dims();
    const memory_desc_wrapper src_d(pd()->src_md());
    const dim_t *src_dims_ptr = src_d.dims();
    const dim_t dst_batch_stride = M * N;

    // Binary post-op src1 bases, one per binary in chain order (scalar or per-N;
    // each broadcasts over M/batch so the same array serves every row). Empty
    // when the chain has no binary entry. Each pointer starts at its memory
    // descriptor's logical origin; the kernel adds the in-row column offset.
    const std::vector<const void *> po_rhs
            = binary_injector_utils::prepare_binary_args(post_ops, ctx);
    const void *const *po_rhs_arr = po_rhs.empty() ? nullptr : po_rhs.data();

    parallel_nd(batch, [&](dim_t b) {
        float *dst_base = dst + b * dst_batch_stride;

        dim_t dst_idx_prefix[DNNL_MAX_NDIMS] = {};
        size_t bias_strides[DNNL_MAX_NDIMS] = {};

        if (bias && bias_ndims > 1) {
            bias_strides[bias_ndims - 1] = 1;
            for (int d = bias_ndims - 2; d >= 0; --d)
                bias_strides[d]
                        = bias_strides[d + 1] * (size_t)bias_dims[d + 1];
        }

        for (dim_t m = 0; m < M; ++m) {
            if (dst_ndims > 2) {
                utils::l_dims_by_l_offset(
                        dst_idx_prefix, b, src_dims_ptr, dst_ndims - 2);
            }
            dst_idx_prefix[dst_ndims - 2] = m;

            float *row_dst = dst_base + m * N;

            const float *bias_ptr = nullptr;
            if (bias) {
                if (bias_d.nelems() == 1) {
                    bias_ptr = bias;
                } else {
                    size_t base_bias_off = 0;
                    if (bias_ndims > 1) {
                        for (int d = 0; d < bias_ndims - 1; ++d) {
                            int dst_dim_idx = d + (dst_ndims - bias_ndims);
                            dim_t idx = (bias_dims[d] == 1)
                                    ? 0
                                    : dst_idx_prefix[dst_dim_idx];
                            base_bias_off += idx * bias_strides[d];
                        }
                    }
                    bias_ptr = bias + base_bias_off;
                }
            }

            // Fused bias + post-op chain over this output row (length N).
            jit_uni_postops_kernel_t::call_params_t cp;
            cp.dst = row_dst;
            cp.bias = bias_ptr;
            cp.rhs = po_rhs_arr;
            cp.off0 = 0; // per-N rhs starts at column 0 of every row
            cp.len = N;
            (*postops_kernel)(&cp);
        }
    });

    return status::success;
}

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
