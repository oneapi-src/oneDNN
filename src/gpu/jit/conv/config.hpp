/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifndef GPU_JIT_CONV_CONFIG_HPP
#define GPU_JIT_CONV_CONFIG_HPP

#include <iostream>
#include <sstream>
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/math_utils.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/compute/compute_engine.hpp"
#include "gpu/jit/conv/block_helper.hpp"
#include "gpu/jit/conv/fma_support.hpp"
#include "gpu/jit/conv/hw_config.hpp"
#include "gpu/jit/conv/tensor.hpp"
#include "gpu/jit/conv/tensor_config.hpp"
#include "gpu/jit/conv/utils.hpp"
#include "gpu/jit/jit_eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Description of the convolution problem.
class conv_problem_t {
public:
    conv_problem_t() = default;

    status_t init(convolution_pd_t *conv_pd) {
        if (conv_pd->has_zero_dim_memory()) return status::unimplemented;

        is_fwd = conv_pd->is_fwd();
        is_bwd_d = conv_pd->is_bwd_d();
        is_bwd_w = conv_pd->is_bwd_w();
        with_bias = conv_pd->with_bias();
        with_groups = conv_pd->with_groups();

        src_data_type = conv_pd->invariant_src_md()->data_type;
        wei_data_type = conv_pd->invariant_wei_md()->data_type;
        bia_data_type = conv_pd->invariant_bia_md()->data_type;
        dst_data_type = conv_pd->invariant_dst_md()->data_type;

        ndims = conv_pd->ndims();

        mb = conv_pd->MB();
        g = conv_pd->G();
        ic = ir_utils::safe_divide(conv_pd->IC(), g);
        oc = ir_utils::safe_divide(conv_pd->OC(), g);

        // Input spatial.
        id = conv_pd->ID();
        ih = conv_pd->IH();
        iw = conv_pd->IW();

        // Output spatial.
        od = conv_pd->OD();
        oh = conv_pd->OH();
        ow = conv_pd->OW();

        // Kernel sizes.
        kd = conv_pd->KD();
        kh = conv_pd->KH();
        kw = conv_pd->KW();

        // Strides.
        sd = conv_pd->KSD();
        sh = conv_pd->KSH();
        sw = conv_pd->KSW();

        // Padding.
        pd = conv_pd->padFront();
        ph = conv_pd->padT();
        pw = conv_pd->padL();

        // Dilation.
        dd = conv_pd->KDD();
        dh = conv_pd->KDH();
        dw = conv_pd->KDW();

        try_reduce_to_1d();

        is_dw = with_groups && (g > 1) && (oc == 1) && (ic == 1);
        osp = od * oh * ow;

        return status::success;
    }

    bool is_stride1() const { return sd == 1 && sh == 1 && sw == 1; }

    // Reduces dimensions for 1x1 kernel.
    void try_reduce_to_1d() {
        bool is_1x1 = (kd * kh * kw == 1);
        bool is_eq_oi = (od == id && oh == ih && ow == iw);
        bool is_iw_1 = iw == 1 && kw == 1 && pw == 0 && ow == 1;
        bool is_ih_1 = ih == 1 && kh == 1 && ph == 0 && oh == 1;
        reduced_dim = 0;
        auto shift_oh_to_ow = [&]() {
            ow = oh;
            iw = ih;
            ih = 1;
            oh = 1;
            kw = kh;
            kh = 1;
            pw = ph;
            ph = 0;
            sw = sh;
            sh = 1;
            dw = dh;
            dh = 0;
            reduced_dim += 1;
        };
        auto shift_od_to_oh = [&]() {
            oh = od;
            ih = id;
            id = 1;
            od = 1;
            kh = kd;
            kd = 1;
            ph = pd;
            pd = 0;
            sh = sd;
            sd = 1;
            dh = dd;
            dd = 0;
            reduced_dim += 1;
        };

        if (is_iw_1) { shift_oh_to_ow(); }
        if (is_ih_1 || is_iw_1) { shift_od_to_oh(); }
        if (is_iw_1 && is_ih_1) { shift_oh_to_ow(); }

        if (is_1x1 && is_stride1() && is_eq_oi) {
            ir_assert(pd == 0 && ph == 0 && pw == 0);
            ow = od * oh * ow;
            iw = id * ih * iw;
            od = id = kd = 1;
            oh = ih = kh = 1;
            reduced_dim = 3;
        }
    }

    std::string desc_str() const {
        std::ostringstream oss;
        oss << "mb" << mb;
        oss << "g" << g;
        oss << "ic" << ic;
        oss << "id" << id;
        oss << "ih" << ih;
        oss << "iw" << iw;
        oss << "oc" << oc;
        oss << "od" << od;
        oss << "oh" << oh;
        oss << "ow" << ow;
        oss << "kd" << kd;
        oss << "kh" << kh;
        oss << "kw" << kw;
        if (sd != 1) oss << "sd" << sd;
        if (sh != 1) oss << "sh" << sh;
        if (sw != 1) oss << "sw" << sw;
        if (dd != 0) oss << "dd" << dd;
        if (dh != 0) oss << "dh" << dh;
        if (dw != 0) oss << "dw" << dw;
        oss << "pd" << pd;
        oss << "ph" << ph;
        oss << "pw" << pw;
        return oss.str();
    }

    tensor_config_t tensor_config;

    data_type_t src_data_type;
    data_type_t wei_data_type;
    data_type_t dst_data_type;
    data_type_t bia_data_type;

    bool is_fwd;
    bool is_bwd_d;
    bool is_bwd_w;
    bool with_bias;
    bool with_groups;
    bool is_dw;

    int ndims;
    int mb; // Batch size.
    int g; // Groups.
    int ic, oc; // Input and output channels.
    int id, ih, iw; // Input spatial sizes.
    int od, oh, ow, osp; // Output spatial sizes.
    int kd, kh, kw; // Kernel sizes.
    int sd, sh, sw; // Strides.
    int pd, ph, pw; // Padding in the beginning.
    int dd, dh, dw; // Dilation.
    int reduced_dim; // Indicates which dims were shifted over or reduced.
};

class conv_config_t;

int estimate_register_count(const conv_config_t &cfg);

// Parameters for kernel generation.
class conv_config_t : public conv_problem_t {
public:
    conv_config_t() = default;

    status_t init(convolution_pd_t *conv_pd, primitive_attr_t *attr,
            engine_t *engine) {

        hw_config_t try_hw_cfg(engine);

        // Try large GRF mode first.
        int try_regs = try_hw_cfg.large_grf_support() ? 256 : 128;
        if (g == 1 && is_f32_conv()) try_regs = 128;
        try_hw_cfg.set_regs(try_regs);

        bool regs_overriden = false;
        bool max_tg_size_overriden = false;

#ifdef GEN_CONV_DEBUG
        int env_regs = getenv_int("regs", -1);
        if (env_regs != -1) {
            try_hw_cfg.set_regs(env_regs);
            regs_overriden = true;
        }
        int env_max_tg_size = getenv_int("max_tg_size", -1);
        if (env_max_tg_size != -1) {
            try_hw_cfg.set_max_tg_size(env_max_tg_size);
            max_tg_size_overriden = true;
        }
#endif

        // Use fixed iterations to avoid infinite loop.
        int max_iters = 10;
        bool ok = false;
        for (int i = 0; i < max_iters; i++) {
            if (i > 0) *this = conv_config_t();

            auto status = init_with_hw_config(conv_pd, attr, try_hw_cfg);

            // Reduce thread group size if SLM size is too large.
            if (!max_tg_size_overriden && status == status::runtime_error
                    && not_enough_slm) {
                try_hw_cfg.set_max_tg_size(hw_cfg.max_tg_size() / 2);
                continue;
            }
            CHECK(status);

            // If the kernel fits 128 registers, switch to the normal mode which is
            // expected to have better performance for such cases.
            int bound = (!is_dp_fma() ? 128 : 112);
            if (!regs_overriden && regs() == 256
                    && estimated_peak_grf_usage <= bound) {
                try_hw_cfg.set_regs(128);
                continue;
            }
            ok = true;
            break;
        }

        return ok ? status::success : status::runtime_error;
    }

    status_t init_with_hw_config(convolution_pd_t *conv_pd,
            primitive_attr_t *attr, const hw_config_t &_hw_cfg) {

        hw_cfg = _hw_cfg;

        // These functions have implicit dependencies between them. They cannot be
        // reordered with verifying these dependencies are satisfied.
        CHECK(conv_problem_t::init(conv_pd));
        CHECK(init_abc_data_types(attr));
        CHECK(init_acc_data_type());
        CHECK(init_fma_kind_and_simd_size());

        init_use_2d_send(conv_pd);
        CHECK(init_data_layouts(conv_pd));

        if (!data_types_ok()) return status::unimplemented;

        CHECK(init_common_config());

        init_fuse_spatial();
        init_hoist_masks_from_compute_loop();

        CHECK(init_common_blocking());

        const memory_desc_t *output_md = nullptr;
        if (is_fwd) {
            CHECK(init_fwd(conv_pd));
            output_md = conv_pd->dst_md();
        } else if (is_bwd_d) {
            CHECK(init_bwd_d(conv_pd));
            output_md = conv_pd->diff_src_md();
        } else if (is_bwd_w) {
            CHECK(init_bwd_w(conv_pd));
            output_md = conv_pd->diff_weights_md();
        } else {
            ir_error_not_expected();
        }

        estimated_peak_grf_usage = estimate_register_count(*this);

        CHECK(attr->set_default_formats(output_md));

        if (!zero_points_ok(conv_pd)) return status::unimplemented;
        if (!post_ops_ok(conv_pd)) return status::unimplemented;

        CHECK(init_extra_tensor_layouts(conv_pd));

        ir_trace() << "=== TRY config:\n" << *this << std::endl;

        return status::success;
    }

    status_t init_fwd(convolution_pd_t *conv_pd);
    status_t init_bwd_d(convolution_pd_t *conv_pd);
    status_t init_bwd_w(convolution_pd_t *conv_pd);

    status_t init_common_config() {
        using namespace ir_utils;

        use_preload = true;

        if (hw() <= ngen::HW::XeLP) use_preload = false;

        // No SLM buffering by default (will be enabled later).
        disable_slm_buffering();

        // No prefetch by default (will be enabled later).
        disable_prefetch();

        do_b_reduction = false;
        pad_slm = true;
        assign_sbids = is_dp_fma();
        do_pipeline_unroll = hw() > ngen::HW::XeLP;
        reduce_grf_usage = true;
        allow_grf_reorder = !matches_user_types();
        do_atomic_update = false;
        reuse_headers = hw() <= ngen::HW::XeLP;
        a_sub_tiles = 1;
        b_sub_tiles = 1;

        init_zero_points_default_config();

#ifdef GEN_CONV_DEBUG
        use_preload = getenv_bool("use_preload", use_preload);
        pad_slm = getenv_bool("pad_slm", pad_slm);
        assign_sbids = getenv_bool("assign_sbids", assign_sbids);
        do_pipeline_unroll
                = getenv_bool("do_pipeline_unroll", do_pipeline_unroll);
        reduce_grf_usage = getenv_bool("reduce_grf_usage", reduce_grf_usage);
        allow_grf_reorder = getenv_bool("allow_grf_reorder", allow_grf_reorder);
        reuse_headers = getenv_bool("reuse_headers", reuse_headers);
        a_sub_tiles = getenv_int("a_sub_tiles", a_sub_tiles);
        b_sub_tiles = getenv_int("b_sub_tiles", b_sub_tiles);
#endif

        return status::success;
    }

    status_t init_common_blocking();

    bool zero_points_ok(const convolution_pd_t *pd) const {
        auto *attr = pd->attr();

        // TODO: implement the rest of the cases and remove this 'if'
        bool ic_kdhw = (ic <= 8) && (kd * kh * kw > 1) && !is_dw;
        if (!attr->zero_points_.has_default_values(DNNL_ARG_SRC) && ic_kdhw)
            return false;

        using namespace data_type;
        const auto src_type = (is_fwd) ? pd->invariant_src_md()->data_type
                                       : pd->invariant_dst_md()->data_type;
        int mask_src = 0, mask_dst = 0;
        attr->zero_points_.get(DNNL_ARG_SRC, nullptr, &mask_src, nullptr);
        attr->zero_points_.get(DNNL_ARG_DST, nullptr, &mask_dst, nullptr);

        return IMPLICATION(!utils::one_of(src_type, s8, u8),
                       attr->zero_points_.has_default_values())
                && attr->zero_points_.has_default_values(DNNL_ARG_WEIGHTS)
                && (mask_src == 0 || mask_src == 1 << 1)
                && (mask_dst == 0 || mask_dst == 1 << 1);
    }

    bool post_ops_ok(const convolution_pd_t *pd) const {
        auto *attr = pd->attr();

        // No post-ops are supported for f64
        if (is_f64_conv() && !attr->has_default_values()) return false;

        if (is_fwd || is_bwd_d) {
            using sm = primitive_attr_t::skip_mask_t;
            auto attr_skip_mask = sm::post_ops | sm::oscale_runtime | sm::sum_dt
                    | sm::zero_points_runtime;
            if (!attr->has_default_values(attr_skip_mask)) return false;
        } else {
            if (!attr->has_default_values()) return false;
        }

        if (!attr->output_scales_.has_default_values()) {
            if (!is_s32_accumulator()) return false;
            // Only common and per_oc output scales were tested.
            if (!utils::one_of(attr->output_scales_.mask_, 0, (1 << 1)))
                return false;
        }
        for (int i = 0; i < attr->post_ops_.len(); i++) {
            auto &po = attr->post_ops_.entry_[i];
            if (po.is_eltwise()) {
                if (!jit_eltwise_injector_f32_is_supported(po.eltwise.alg))
                    return false;
            } else if (po.is_binary() || po.is_prelu()) {
                int mask = po.is_prelu()
                        ? po.prelu.mask
                        : utils::get_dims_mask(pd->invariant_dst_md()->dims,
                                po.binary.src1_desc.dims, ndims, true);
                // These cases don't have message-related limitations.
                if ((mask & (1 << 1)) == 0 || mask == (1 << 1)) continue;
                auto rhs_layout = po.is_prelu() ? layout_t(type_t::f32(), 0,
                                          get_prelu_weights_dims(po.prelu.mask,
                                                  *pd->invariant_dst_md()))
                                                : layout_t(po.binary.src1_desc);
                // No blocks means it's a scalar, can be always loaded.
                if (rhs_layout.blocks().empty()) return true;

                auto rhs0 = rhs_layout.blocks()[0];
                int block_bytes = rhs0.block * rhs_layout.type().size();
                // Innermost block must:
                // - be across output channels
                // - be dense
                // - aligned to 32 bytes (for HWord loads)
                if (rhs0.dim_idx != 1 || dim_t(rhs0.stride) != 1
                        || block_bytes % 32 != 0)
                    return false;
            }
        }
        return true;
    }

    bool data_types_ok() const {
        bool is_bf16 = utils::one_of(data_type::bf16, src_data_type,
                wei_data_type, dst_data_type, bia_data_type);
        if (!is_f64_conv()
                && utils::one_of(data_type::f64, src_data_type, wei_data_type,
                        dst_data_type, bia_data_type))
            return false;
        if (is_bf16 && hw() <= ngen::HW::XeLP) return false;
        if (is_f64_conv()
                && utils::one_of(hw(), ngen::HW::XeLP, ngen::HW::XeHPG))
            return false;
        if (is_fwd) return true;
        if (is_bwd_d) return true;
        if (is_bwd_w) {
            bool ok = true;
            ok &= (src_data_type == data_type::bf16
                    || src_data_type == data_type::f32);
            ok &= (dst_data_type == src_data_type);
            ok &= utils::one_of(wei_data_type, src_data_type, data_type::f32);

            if (with_bias) {
                ok &= utils::one_of(
                        bia_data_type, src_data_type, data_type::f32);
            }
            return ok;
        }
        return false;
    }

    int grid_dim(const std::string &name) const;
    int padded_dim(const std::string &name) const;
    const std::unordered_map<std::string, int> &padded_dims() const;
    const std::unordered_map<std::string, int> &dim_blocks() const;

    bool is_s32_accumulator() const { return acc_data_type == data_type::s32; }
    bool is_f32_conv() const {
        return utils::everyone_is(src_data_type, wei_data_type, data_type::f32);
    }
    bool is_f64_conv() const {
        return utils::everyone_is(src_data_type, wei_data_type, data_type::f64);
    }
    bool is_int8_dst() const {
        return utils::one_of(dst_data_type, data_type::s8, data_type::u8);
    }
    bool is_small_ic() const { return ic <= 8; }
    bool is_dw_large_mb() const { return is_dw && mb >= 16; }
    bool is_mixed_int8() const {
        return utils::one_of(a_data_type, dnnl_f16, dnnl_f32)
                && utils::one_of(c_data_type, dnnl_u8, dnnl_s8);
    }
    bool is_dp_fma() const {
        return utils::one_of(fma_kind, fma_kind_t::dpas, fma_kind_t::dpasw,
                fma_kind_t::dp4a);
    }
    bool is_dpas_or_dpasw_fma() const {
        return utils::one_of(fma_kind, fma_kind_t::dpas, fma_kind_t::dpasw);
    }

    ngen::HW hw() const { return hw_cfg.hw(); }
    int regs() const { return hw_cfg.regs(); }
    int simd_size() const { return hw_cfg.simd_size(); }
    int grf_size() const { return hw_cfg.grf_size(); }
    bool is_ge_xe_hpc() const { return (hw_cfg.hw() >= ngen::HW::XeHPC); }

    compute::nd_range_t nd_range() const {
        size_t gws[3];
        size_t lws[3];
        for (int i = 0; i < 3; i++) {
            lws[i] = tg_grid_dim[i] * (i == 0 ? simd_size() : 1);
            gws[i] = kernel_grid_dim[i] * lws[i];
        }
        return compute::nd_range_t(gws, lws);
    }

    const layout_t &a_layout() const {
        if (is_fwd) return src_layout;
        if (is_bwd_d) return dst_layout;
        return src_layout;
    }

    const layout_t &b_layout() const {
        if (is_fwd) return wei_layout;
        if (is_bwd_d) return wei_layout;
        return dst_layout;
    }

    const layout_t &c_layout() const {
        if (is_fwd) return dst_layout;
        if (is_bwd_d) return src_layout;
        return wei_layout;
    }

    std::string str() const;

    static bool matches_tag(const layout_t &layout, const std::string &tag) {
        auto tag_layout = make_layout(layout.type(), layout.dims(), tag);
        if (!layout.is_strictly_equal(tag_layout)) return false;
        return true;
    }

    bool is_compute_nhwc(const std::string &tag) const {
        auto &layout = tensor_config.compute_layout(tag);
        return matches_tag(layout, "axb");
    }

    bool not_enough_slm = false;

    data_type_t a_data_type;
    data_type_t b_data_type;
    data_type_t c_data_type;
    data_type_t acc_data_type;

    int a_data_type_size;
    int b_data_type_size;
    int c_data_type_size;
    int acc_data_type_size;

    layout_t src_layout;
    layout_t wei_layout;
    layout_t dst_layout;
    layout_t bia_layout;

    hw_config_t hw_cfg;

    // Thread group dimensions (thread group grid).
    std::array<int, 3> tg_grid_dim;

    // Number of thread groups across dimensions (kernel grid).
    std::array<int, 3> kernel_grid_dim;

    std::shared_ptr<block_helper_t> bh;

    // Block sizes per thread group.
    int g_tg_blk;
    int ic_tg_blk;
    int iw_tg_blk;
    int kw_tg_blk;
    int mb_tg_blk;
    int oc_tg_blk;
    int od_tg_blk;
    int oh_tg_blk;
    int ow_tg_blk;
    int osp_tg_blk;

    // Number of thread blocks across problem dimensions.
    int ic_thr_dim;
    int iw_thr_dim;
    int mb_thr_dim;
    int oc_thr_dim;
    int ow_thr_dim;
    int osp_thr_dim;

    // Block sizes per thread.
    int g_thr_blk;
    int ic_thr_blk;
    int iw_thr_blk;
    int mb_thr_blk;
    int oc_thr_blk;
    int od_thr_blk;
    int oh_thr_blk;
    int ow_thr_blk;
    int osp_thr_blk;

    // Block sizes per iteration.
    int ic_blk;
    int kw_blk;
    int mb_blk;
    int oc_blk;
    int ow_blk;

    // Block sizes in GEMM notation.
    int b_blk;
    int m_tg_blk;
    int n_tg_blk;
    int k_blk;

    // Unroll sizes.
    int mb_unroll;
    int ow_unroll;

    bool do_b_reduction;

    // Which instruction backend to use.
    fma_kind_t fma_kind = fma_kind_t::unknown;

    bool use_preload; // Whether to use SLM or prefetch.
    bool use_a_slm; // Whether to use SLM for A.
    bool use_b_slm; // Whether to use SLM for B.
    bool use_prefetch; // Whether to use prefetch for A and B.
    bool pad_slm; // Whether to pad SLM to avoid write conflicts.
    bool assign_sbids; // Whether to manually assign SBID tokens.
    int slm_bufs; // Number of SLM buffers to use.
    int gmem_bufs; // Number of GRF buffers to use for GMEM -> SLM copy.
    int prefetch_bufs; // Number of prefetch buffers for A and B.
    bool do_pipeline_unroll; // Whether to fully unroll inner loops for pipelining.
    bool reduce_grf_usage; // Whether to try to reduce GRF usage based on heuristics.
    bool allow_grf_reorder; // Whether to allow GRF reorders to FMA-friendly layouts.
    bool do_atomic_update; // Whether to use atomics during C update.
    bool reuse_headers; // Whether to reuse header messages to reduce GRF usage.
    bool bwd_d_optimize_strided; // Apply special optimization for strided BWD_D convolution.
    bool bwd_d_optimize_strided_iw; // Apply special optimization for strided BWD_D convolution (iw dim).
    bool use_ow_kw_grf_cache; // Whether to use GRF cache to reuse source for ow/kw pairs.
    bool fuse_spatial; // Apply blocking to fused spatial (otherwise only `w` is blocked).
    bool hoist_masks_from_compute_loop; // Whether to move send mask initialization out of compute loop.
    bool allow_slm_tg_slicing; // Whether to allow thread group split for SLM load/store.
    bool use_a_2d_send; // Whether to use 2D block messages for A.
    bool use_b_2d_send; // Whether to use 2D block messages for B.

    static const int max_slm_bufs = 3; // Maximum number of SLM buffers.

    // GRF usage for kernel arguments, local work IDs/sizes, signal header,
    // temporary expressions, etc.
    static const int reserved_regs = 16;

    // Specific to FWD int8
    struct zero_points_config_t {
        bool do_src_compensation;
        bool do_dst_compensation;
        bool is_runtime_src_zero_points;
        bool is_runtime_dst_zero_points;
        bool is_common_src_zero_point;
        bool is_common_dst_zero_point;
        int common_src_zero_point;
        int common_dst_zero_point;
    } zp_cfg;

    // Sub-tiles to split into for the inner A x B multiplication:
    // for i in range(0, a_sub_tiles):
    //     A_i = load(...)
    //     for j in range(0, b_sub_tiles):
    //         B_j = load(...)
    //         C_i_j += A_i * B_j
    //
    // GRF buffers for A_i and B_j are reused. Factors greater than one help to
    // reduce GRF usage.
    int a_sub_tiles;
    int b_sub_tiles;
    int estimated_peak_grf_usage = 0;

private:
    // Initializes A/B/C data types (GEMM notation: C += A * B) according to
    // the following convention:
    // FWD:        src -> A,      wei -> B,      dst -> C
    // BWD_D: diff_dst -> A,      wei -> B, diff_src -> C
    // BWD_W:      src -> A, diff_dst -> B, diff_wei -> C
    status_t init_abc_data_types(primitive_attr_t *attr) {
        if (is_fwd) {
            a_data_type = src_data_type;
            b_data_type = wei_data_type;
            c_data_type = dst_data_type;
        } else if (is_bwd_d) {
            a_data_type = dst_data_type;
            b_data_type = wei_data_type;
            c_data_type = src_data_type;
        } else if (is_bwd_w) {
            a_data_type = src_data_type;
            b_data_type = dst_data_type;
            // Always use f32 for accumulation/storing in the main kernel.
            c_data_type = data_type::f32;
        } else {
            ir_error_not_expected();
        }

        if (utils::everyone_is(
                    data_type::f32, a_data_type, b_data_type, c_data_type)) {

            // TODO: bf16 and f16 currently perform worse than tf32, this is
            // likely due to an extra reorder required on the b buffer.
            bool use_matching_fpmath = false;
#ifdef GEN_CONV_DEBUG
            use_matching_fpmath = ir_utils::getenv_bool(
                    "use_matching_fpmath", use_matching_fpmath);
#endif
            if (use_matching_fpmath
                    && attr->mayidownconvert(data_type::f32, data_type::bf16)
                    && fma_kind::get_supported_kind(hw(), data_type::bf16,
                               data_type::bf16, data_type::f32)
                            != fma_kind_t::unknown) {
                a_data_type = data_type::bf16;
                b_data_type = data_type::bf16;
            } else if (use_matching_fpmath
                    && attr->mayidownconvert(data_type::f32, data_type::f16)
                    && fma_kind::get_supported_kind(hw(), data_type::f16,
                               data_type::f16, data_type::f32)
                            != fma_kind_t::unknown) {
                a_data_type = data_type::f16;
                b_data_type = data_type::f16;
            } else if (attr->mayidownconvert(data_type::f32, data_type::tf32)
                    && fma_kind::get_supported_kind(hw(), data_type::tf32,
                               data_type::tf32, data_type::f32)
                            != fma_kind_t::unknown) {
                a_data_type = data_type::tf32;
                b_data_type = data_type::tf32;
            }
        }

        a_data_type_size = (int)types::data_type_size(a_data_type);
        b_data_type_size = (int)types::data_type_size(b_data_type);
        c_data_type_size = (int)types::data_type_size(c_data_type);
        return status::success;
    }

    bool matches_user_types() const {
        if (is_fwd) {
            return a_data_type == src_data_type && b_data_type == wei_data_type
                    && c_data_type == dst_data_type;
        } else if (is_bwd_d) {
            return a_data_type == dst_data_type && b_data_type == wei_data_type
                    && c_data_type == src_data_type;
        } else if (is_bwd_w) {
            return a_data_type == src_data_type && b_data_type == dst_data_type
                    && c_data_type == wei_data_type;
        } else {
            ir_error_not_expected();
            return false;
        }
    }

    status_t init_acc_data_type() {
        auto a = a_data_type;
        auto b = b_data_type;
        acc_data_type = data_type::undef;
        if (utils::one_of(a, data_type::s8, data_type::u8)
                && utils::one_of(b, data_type::s8, data_type::u8)) {
            acc_data_type = data_type::s32;
        } else if (utils::everyone_is(data_type::f16, a, b)
                || utils::everyone_is(data_type::bf16, a, b)) {
            acc_data_type = data_type::f32;
        } else if (utils::everyone_is(data_type::tf32, a, b)) {
            acc_data_type = data_type::f32;
        } else if (utils::everyone_is(data_type::f32, a, b)) {
            acc_data_type = data_type::f32;
        } else if (utils::everyone_is(data_type::f64, a, b)) {
            acc_data_type = data_type::f64;
        }
        if (acc_data_type == data_type::undef) return status::unimplemented;
        acc_data_type_size = (int)types::data_type_size(acc_data_type);
        return status::success;
    }

    status_t init_fma_kind_and_simd_size() {
        fma_kind = fma_kind::get_supported_kind(
                hw(), a_data_type, b_data_type, acc_data_type);

        // Force mad for some cases.
        if (is_dw) fma_kind = fma_kind_t::mad;

#ifdef GEN_CONV_DEBUG
        fma_kind = fma_kind::from_string(ir_utils::getenv_str(
                "fma_kind", fma_kind::to_string(fma_kind)));
#endif

        if (fma_kind == fma_kind_t::unknown) return status::unimplemented;

        init_simd_size();

        // Disable using mad instruction pre-XeHP until performance parity is
        // reached with OpenCL kernels.
        if (hw() < ngen::HW::XeHP) return status::unimplemented;

        return status::success;
    }

    void init_simd_size() {
        int simd_size = fma_kind::get_simd_size(
                hw(), fma_kind, a_data_type, b_data_type, acc_data_type);
        int vec_size = simd_size;
        if (fma_kind == fma_kind_t::mad) {
            int vec_dim = (is_fwd || is_bwd_w) ? oc : ic;
            if (vec_dim <= 8) vec_size = std::min(8, vec_size);
        }
        hw_cfg.set_simd_size(simd_size);
#ifdef GEN_CONV_DEBUG
        vec_size = getenv_int("vec_size", vec_size);
#endif
        hw_cfg.set_vec_size(vec_size);
    }

    void init_data_tags(bool allow_src_reorder, bool allow_wei_reorder,
            bool allow_dst_reorder, std::string &src_tag, std::string &wei_tag,
            std::string &dst_tag, std::string &user_wei_tag);
    status_t init_data_layouts(convolution_pd_t *conv_pd);

    int slm_size() const {
        if (prefer_prefetch()) return 0;

        int a_slm_size = m_tg_blk * k_blk * a_data_type_size;
        int b_slm_size = n_tg_blk * k_blk * b_data_type_size;

        int ret = 0;
        if (use_a_slm) ret += a_slm_size;
        if (use_b_slm) ret += b_slm_size;
        ret *= slm_bufs;

        return ret;
    }

    std::vector<dim_t> get_prelu_weights_dims(
            uint32_t mask, const memory_desc_t &md) const {
        std::vector<dim_t> dims(md.dims, md.dims + md.ndims);
        for (int i = 0; i < md.ndims; ++i)
            dims[i] = (mask & (1 << i)) ? dims[i] : 1;
        return dims;
    }

    status_t init_extra_tensor_layouts(const convolution_pd_t *conv_pd) {
        auto *attr = conv_pd->attr();
        if (zp_cfg.do_src_compensation && zp_cfg.is_runtime_src_zero_points) {
            int zp_ic = (zp_cfg.is_common_src_zero_point) ? 1 : ic;
            std::vector<dim_t> dims = {zp_ic};
            layout_t zp_layout(type_t::s32(), 0, dims);
            int arg_key = DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC;
            tensor_config.add_tensor("src_zero_points", arg_key,
                    /*is_input=*/true, /*is_output=*/false, zp_layout);
        }
        if (zp_cfg.do_dst_compensation && zp_cfg.is_runtime_dst_zero_points) {
            std::vector<dim_t> dims = {oc};
            layout_t zp_layout(type_t::s32(), 0, dims);
            int arg_key = DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST;
            tensor_config.add_tensor("dst_zero_points", arg_key,
                    /*is_input=*/true, /*is_output=*/false, zp_layout);
        }
        bool with_oscales = !attr->output_scales_.has_default_values();
        if (with_oscales) {
            std::vector<dim_t> dims = {attr->output_scales_.count_};
            layout_t oscales_layout(type_t::f32(), 0, dims);
            int arg_key = -1;
            if (!attr->output_scales_.defined())
                arg_key = DNNL_ARG_ATTR_OUTPUT_SCALES;
            tensor_config.add_tensor("oscales", arg_key, /*is_input=*/true,
                    /*is_output=*/false, oscales_layout);
        }
        for (int i = 0; i < attr->post_ops_.len(); i++) {
            auto &po = attr->post_ops_.entry_[i];
            if (po.is_eltwise()
                    || po.is_sum(/*require_scale_one=*/false,
                            /*require_zp_zero=*/false)) {
                // No extra tensors.
            } else if (po.is_binary()) {
                auto layout = make_layout(po.binary.src1_desc);
                int arg_key
                        = DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1;
                tensor_config.add_tensor("binary_rhs_" + std::to_string(i),
                        arg_key, /*is_input=*/true,
                        /*is_output=*/false, layout);
            } else if (po.is_prelu()) {
                layout_t layout(type_t::f32(), 0,
                        get_prelu_weights_dims(
                                po.prelu.mask, *conv_pd->invariant_dst_md()));
                int arg_key
                        = DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_WEIGHTS;
                tensor_config.add_tensor("prelu_rhs_" + std::to_string(i),
                        arg_key,
                        /*is_input=*/true, /*is_output=*/false, layout);
            } else {
                ir_error_not_expected();
            }
        }
        return status::success;
    }

    bool should_use_spatial_blocking(int d, int h, int w) const;
    void init_use_2d_send(const convolution_pd_t *conv_pd);
    void init_fuse_spatial();
    void init_hoist_masks_from_compute_loop();
    void init_bwd_d_optimize_strided(int iw_thr_blk);
    void init_use_ow_kw_grf_cache();
    void init_allow_slm_tg_slicing(
            int m_blk, int n_blk, int m_tg_dim, int n_tg_dim);

    void init_zero_points_default_config() {
        zp_cfg.do_src_compensation = false;
        zp_cfg.do_dst_compensation = false;
        zp_cfg.is_runtime_src_zero_points = false;
        zp_cfg.is_runtime_dst_zero_points = false;
        zp_cfg.is_common_src_zero_point = false;
        zp_cfg.is_common_dst_zero_point = false;
        zp_cfg.common_src_zero_point = 0;
        zp_cfg.common_dst_zero_point = 0;
    }

    status_t init_zero_points_config(convolution_pd_t *conv_pd);

    bool prefer_prefetch() const {
        bool ret = false;
        if (is_ge_xe_hpc()) ret = true;

#ifdef GEN_CONV_DEBUG
        ret = ir_utils::getenv_bool("prefer_prefetch", ret);
#endif
        return ret;
    }

    bool can_split_across_thread_group(int elems, int type_size) const {
        // Thread group grid is limited to powers of two. We can reliably split
        // only powers of two elements across such grids.
        if (!math::is_pow2(elems)) return false;

        // Check that the buffer can be uniformly distributed.
        int tg_size = (tg_grid_dim[0] * tg_grid_dim[1] * tg_grid_dim[2]);
        if (elems % tg_size != 0) return false;

        // Check that SLM can be stored with oword messages.
        int bytes_per_thr = (elems / tg_size) * type_size;
        if (bytes_per_thr % 16 != 0) return false;

        return true;
    }

    void enable_slm_buffering() {
        using namespace ir_utils;
        if (!use_ow_kw_grf_cache) {
            bool can_split_a = can_split_across_thread_group(
                    m_tg_blk * k_blk, a_data_type_size);
            use_a_slm = (tg_grid_dim[0] > 1) && can_split_a;
        }
        bool can_split_b = can_split_across_thread_group(
                n_tg_blk * k_blk, b_data_type_size);
        use_b_slm = (tg_grid_dim[1] > 1) && can_split_b;

        if (use_a_slm || use_b_slm) {
            bool is_small_tg = (tg_grid_dim[0] * tg_grid_dim[1] <= 8);
            int pref_slm_bufs
                    = ((is_small_tg || is_f32_conv()) && mb > 1 ? 2 : 3);
            if (do_pipeline_unroll) {
                slm_bufs = pref_slm_bufs;
                gmem_bufs = (is_dp_fma() ? 2 : 1);
            } else {
                // Double/triple SLM buffering is not supported when only one
                // matrix is SLM-buffered.
                slm_bufs = (use_a_slm == use_b_slm ? pref_slm_bufs : 1);
                gmem_bufs = 1;
            }
        } else {
            slm_bufs = 0;
            gmem_bufs = 0;
        }
#ifdef GEN_CONV_DEBUG
        use_a_slm = getenv_bool("use_a_slm", use_a_slm);
        use_b_slm = getenv_bool("use_b_slm", use_b_slm);
        slm_bufs = getenv_int("slm_bufs", slm_bufs);
        gmem_bufs = getenv_int("gmem_bufs", gmem_bufs);
#endif
    }

    void enable_prefetch() {
        using namespace ir_utils;

        bool can_split_a = (tg_grid_dim[0] == 1)
                || can_split_across_thread_group(
                        m_tg_blk * k_blk, a_data_type_size);
        bool can_split_b = (tg_grid_dim[1] == 1)
                || can_split_across_thread_group(
                        n_tg_blk * k_blk, b_data_type_size);

        use_prefetch = (can_split_a && can_split_b);
        prefetch_bufs = is_f32_conv() ? 2 : 3;
#ifdef GEN_CONV_DEBUG
        use_prefetch = getenv_bool("use_prefetch", use_prefetch);
        prefetch_bufs = getenv_int("prefetch_bufs", prefetch_bufs);
#endif
    }

    void disable_slm_buffering() {
        use_a_slm = false;
        use_b_slm = false;
        slm_bufs = 0;
        gmem_bufs = 0;
    }

    void disable_prefetch() {
        use_prefetch = false;
        prefetch_bufs = 0;
    }

    // Overwrites parameters that are implied by other parameters.
    status_t fixup_inference_consistency();

    bool try_reduce_grf_usage() {
        if (!reduce_grf_usage) return true;

        // TODO: improve estimate register count, it fails to account for tmp
        // values like mask_registers among other things.
        int max_regs = regs();
        int est_regs = estimate_register_count(*this);
        if (est_regs <= max_regs) return true;

        // Try to disable GRF buffering.
        if (gmem_bufs > 1) {
            gmem_bufs = 1;
            int est_regs = estimate_register_count(*this);
            if (est_regs <= max_regs) return true;
        }

        // Try to use sub-tiles for B.
        int n_thr_blk = utils::div_up(n_tg_blk, tg_grid_dim[0]);
        int max_b_sub_tiles
                = std::min((use_b_slm ? 4 : 2), n_thr_blk / simd_size());
        // XXX: avoid layout mismatch for B loads
        if (is_ge_xe_hpc() && is_bwd_w)
            max_b_sub_tiles = std::min(2, max_b_sub_tiles);
        while (b_sub_tiles < max_b_sub_tiles) {
            b_sub_tiles *= 2;
            int est_regs = estimate_register_count(*this);
            if (est_regs <= max_regs) return true;
        }

        // Try to use sub-tiles for A.
        int m_thr_blk = utils::div_up(m_tg_blk, tg_grid_dim[1]);
        int max_a_sub_tiles = std::min((use_a_slm ? 4 : 2), m_thr_blk / 8);
        if (b_sub_tiles > 1) max_a_sub_tiles = 1;
        while (a_sub_tiles < max_a_sub_tiles) {
            a_sub_tiles *= 2;
            int est_regs = estimate_register_count(*this);
            if (est_regs <= max_regs) return true;
        }

        // Try to use double SLM buffering.
        if (slm_bufs == 3) {
            slm_bufs = 2;
            int est_regs = estimate_register_count(*this);
            if (est_regs <= max_regs) return true;
        }

        // Try to use single SLM buffering.
        if (slm_bufs == 2) {
            slm_bufs = 1;
            int est_regs = estimate_register_count(*this);
            if (est_regs <= max_regs) return true;
        }

        // Last resort settings to reduce GRF usage.
        reuse_headers = true;
        do_pipeline_unroll = false;

        return estimate_register_count(*this) <= max_regs;
    }

    int src_arg_key() const {
        if (is_fwd) return DNNL_ARG_SRC;
        if (is_bwd_d) return DNNL_ARG_DIFF_SRC;
        if (is_bwd_w) return DNNL_ARG_SRC;
        ir_error_not_expected();
        return -1;
    }

    bool is_src_input() const { return is_fwd || is_bwd_w; }
    bool is_src_output() const { return is_bwd_d; }

    int wei_arg_key() const {
        if (is_fwd) return DNNL_ARG_WEIGHTS;
        if (is_bwd_d) return DNNL_ARG_WEIGHTS;
        if (is_bwd_w) return DNNL_ARG_DIFF_WEIGHTS;
        ir_error_not_expected();
        return -1;
    }

    bool is_wei_input() const { return is_fwd || is_bwd_d; }
    bool is_wei_output() const { return is_bwd_w; }

    int bia_arg_key() const {
        if (is_fwd) return DNNL_ARG_BIAS;
        if (is_bwd_d) return DNNL_ARG_BIAS;
        if (is_bwd_w) return DNNL_ARG_DIFF_BIAS;
        ir_error_not_expected();
        return -1;
    }

    bool is_bia_input() const { return is_fwd || is_bwd_d; }
    bool is_bia_output() const { return is_bwd_w; }

    int dst_arg_key() const {
        if (is_fwd) return DNNL_ARG_DST;
        if (is_bwd_d) return DNNL_ARG_DIFF_DST;
        if (is_bwd_w) return DNNL_ARG_DIFF_DST;
        ir_error_not_expected();
        return -1;
    }

    bool is_dst_input() const { return is_bwd_d || is_bwd_w; }
    bool is_dst_output() const { return is_fwd; }

    void set_allow_grf_reorder() {
        bool is_a_grf_blocked
                = (a_layout().innermost_block_layout().size() % grf_size()
                        == 0);
        if (is_fwd) {
            if (is_f64_conv()) {
                allow_grf_reorder = false;
            } else if (!is_dp_fma()
                    && !utils::everyone_is(
                            a_data_type, b_data_type, data_type::f32)) {
                allow_grf_reorder = true;
            } else if (is_small_ic() && is_dp_fma()) {
                allow_grf_reorder = true;
            } else if (!is_a_grf_blocked
                    && (ic_blk * a_data_type_size % grf_size() != 0
                            || ic != bh->padded_size("ic"))) {
                allow_grf_reorder = true;
            }
        } else if (is_bwd_d) {
            if (!is_dp_fma()
                    && !utils::everyone_is(
                            a_data_type, b_data_type, data_type::f32)) {
                allow_grf_reorder = true;
            } else if (!is_a_grf_blocked
                    && (oc_blk * a_data_type_size % grf_size() != 0
                            || oc != bh->padded_size("oc"))) {
                allow_grf_reorder = true;
            }
        } else if (is_bwd_w) {
            if (is_dw || is_dp_fma()) allow_grf_reorder = true;
        }
    }

    static std::string prepend_groups_to_tag(const std::string &tag) {
        auto ret = tag;
        for (auto &c : ret) {
            bool is_lower_dim = ('a' <= c && c < 'a' + DNNL_MAX_NDIMS);
            bool is_upper_dim = ('A' <= c && c < 'A' + DNNL_MAX_NDIMS);
            if (!is_lower_dim && !is_upper_dim) continue;
            c += 1;
        }
        return "a" + ret;
    }

    static void set_default_format(memory_desc_t &md, const std::string &tag) {
        if (md.format_kind != format_kind::any) return;
        md = make_layout(md, tag).to_dnnl(md.dims);
    }

    static layout_t init_layout(
            memory_desc_t &user_md, const std::string &optimal_tag) {
        auto optimal = make_layout(user_md, optimal_tag);
        if (user_md.format_kind != format_kind::any) {
            auto user = make_layout(user_md);
            // If layouts are physically different return the layout passed by
            // the user and return unimplemented later.
            if (user != optimal) return user;
        } else {
            user_md = optimal.to_dnnl(user_md.dims);
        }
        return optimal;
    }

    static layout_t make_layout(const memory_desc_t &md) {
        return layout_t(md, /*do_normalize=*/false);
    }

    static layout_t make_layout(
            const memory_desc_t &md, const std::string &tag) {
        return layout_t(md, tag, /*do_normalize=*/false);
    }

    static layout_t make_layout(const type_t &type,
            const std::vector<dim_t> &dims, const std::string &tag) {
        return layout_t(type, 0, tag, dims, /*do_normalize=*/false);
    }

    static bool with_sum_post_op(const convolution_pd_t *pd) {
        auto &post_ops = pd->attr()->post_ops_;
        return post_ops.find(primitive_kind::sum) != -1;
    }

    static bool matches_tag(const memory_desc_t &md, const std::string &tag) {
        if (md.format_kind == format_kind::any) return false;
        return matches_tag(make_layout(md), tag);
    }

    // Returns true if 1) md has nhwc layout and 2) it can't be treated as
    // blocking layout, and false otherwise.
    static bool is_pure_nhwc(
            const memory_desc_t &md, const std::string &blocking_tag) {
        if (!matches_tag(md, "axb")) return false;
        if (make_layout(md) == make_layout(md, blocking_tag)) return false;
        return true;
    }

    int get_thread_groups() const {
        return kernel_grid_dim[0] * kernel_grid_dim[1] * kernel_grid_dim[2];
    }
    int get_thread_group_size() const {
        return tg_grid_dim[0] * tg_grid_dim[1] * tg_grid_dim[2];
    }
    int get_thread_count() const {
        return get_thread_groups() * get_thread_group_size();
    }

    // Return thread utilization as a percentage. If this value is low,
    // parallelism is a fundamental limitation to the current work scheduling.
    float get_thread_utilization() const {
        auto arch = convert_ngen_arch_to_dnnl(hw());
        int slice_eu_count = compute::device_info_t::max_eus_per_wg(arch);
        int slice_count = hw_cfg.eu_count() / slice_eu_count;

        int min_wg_per_slice_wave
                = std::max(slice_eu_count / get_thread_group_size(), 1);
        int min_wg_per_wave = slice_count * min_wg_per_slice_wave;

        int wg = get_thread_groups();

        return ((float)wg / utils::rnd_up(wg, min_wg_per_wave)) * 100;
    }

    // Return wave utilization as a percentage. If this value is low, memory
    // latency may be an issue due to limited use of SMT to hide the latency.
    float get_wave_utilization() const {
        auto arch = convert_ngen_arch_to_dnnl(hw());
        int threads_per_eu = compute::device_info_t::threads_per_eu(
                arch, hw_cfg.large_grf_support());
        int slice_eu_count = compute::device_info_t::max_eus_per_wg(arch);
        int slice_count = hw_cfg.eu_count() / slice_eu_count;

        int max_wg_per_slice_wave
                = slice_eu_count * threads_per_eu / get_thread_group_size();
        int max_wg_per_wave = slice_count * max_wg_per_slice_wave;

        int wg = get_thread_groups();

        return ((float)wg / utils::rnd_up(wg, max_wg_per_wave)) * 100;
    }
};

inline std::ostream &operator<<(std::ostream &out, const conv_config_t &cfg) {
    out << cfg.str();
    return out;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
