/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "gpu/intel/jit/conv/tiler.hpp"

#include <cmath>
#include <mutex>
#include <unordered_set>

#include "xpu/stream_profiler.hpp"

#include "gpu/intel/jit/conv/config.hpp"
#include "gpu/intel/jit/conv/lookup_table.hpp"
#include "gpu/intel/jit/conv/model_bridge.hpp"
#include "gpu/intel/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

namespace {

int conv_tune_level() {
    return gpu_utils::dev_getenv("gpu_conv_tune", 0);
}

std::vector<tensor_kind_t> input_tensors(const conv_problem_t &prb) {
    std::vector<tensor_kind_t> ret;
    if (prb.is_fwd || prb.is_bwd_w) ret.push_back(tensor_kind_t::src);
    if (prb.is_fwd || prb.is_bwd_d) ret.push_back(tensor_kind_t::wei);
    if (prb.is_bwd_d || prb.is_bwd_w) ret.push_back(tensor_kind_t::dst);
    return ret;
}

bool is_reduction_dim(const prb_dim_t &d, const conv_problem_t &prb) {
    return to_gemm(d, prb) == prb_dims::k;
}

bool is_vectorized_dim(const prb_dim_t &d, const conv_problem_t &prb) {
    if (prb.is_dw) return d == prb_dims::g;
    return to_gemm(d, prb) == prb_dims::n;
}

int tensor_conv_dim_index(const prb_dim_t &d, tensor_kind_t t) {
    using namespace prb_dims;
    std::vector<prb_dim_t> src_dims = {mb, g, ic, id, ih, iw};
    std::vector<prb_dim_t> wei_dims = {g, oc, ic, kd, kh, kw};
    std::vector<prb_dim_t> dst_dims = {mb, g, oc, od, oh, ow};
    std::vector<prb_dim_t> *prb_dims = nullptr;
    switch (t) {
        case tensor_kind_t::src: prb_dims = &src_dims; break;
        case tensor_kind_t::wei: prb_dims = &wei_dims; break;
        case tensor_kind_t::dst: prb_dims = &dst_dims; break;
        default: ir_error_not_expected();
    }
    auto it = std::find(prb_dims->begin(), prb_dims->end(), d);
    if (it == prb_dims->end()) return -1;
    return (int)(it - prb_dims->begin());
}

// Used for fused reduction dimensions with dpas only.
struct x2_tile_info_t {
    x2_tile_info_t() = default;
    x2_tile_info_t(const prb_dim_t &dim0, const prb_dim_t &dim1)
        : dim0(dim0), dim1(dim1) {
        ir_assert(dim0 != dim1);
    }
    void add(tile_flags_t f) { flags = flags | f; }
    void set_iter_unit(int unit) { d.set_iter_unit(unit); }
    void set_iter_unit0(int unit) { d0.set_iter_unit(unit); }
    void set_iter_unit1(int unit) { d1.set_iter_unit(unit); }

    std::vector<std::pair<int, int>> iter_blocks(int size0, int size1) const {
        if (!any(flags & tile_flags_t::iter)) return {std::make_pair(1, 1)};

        std::vector<std::pair<int, int>> ret;
        int lo = std::min(
                size0 * size1, (int)tile_info_t::default_min_iter_blk);
        int hi = tile_info_t::default_max_iter_blk;
        for (int eff = 100; eff > 0; eff--) {
            for (int ij = lo; ij <= hi; ij++) {
                if (!d.is_iter_ok(ij)) continue;
                auto factors = tile_info_t::get_factors(ij);
                if (!tile_info_t::block_ok(size0 * size1, ij, eff)) continue;
                for (int i : factors) {
                    int j = ij / i;
                    if (d0.is_iter_ok(i) && d1.is_iter_ok(j)) {
                        ret.push_back(std::make_pair(i, j));
                    }
                }
            }
            if (!ret.empty()) return ret;
        }
        ir_error_not_expected();
        return ret;
    }

    std::vector<std::pair<int, int>> thread_group_blocks(
            int size0, int size1) const {
        if (any(flags & tile_flags_t::thread_group)) ir_error_not_expected();
        return {std::make_pair(1, 1)};
    }

    std::vector<std::pair<int, int>> loop_blocks(int size0, int size1) const {
        if (!any(flags & tile_flags_t::loop)) return {std::make_pair(1, 1)};
        if (!any(flags & tile_flags_t::loop_span)) ir_error_not_expected();
        return {std::make_pair(size0, size1)};
    }

    prb_dim_t dim0;
    prb_dim_t dim1;
    tile_flags_t flags = tile_flags_t::undef;
    div_info_t d;
    div_info_t d0;
    div_info_t d1;
};

const layout_t &compute_layout(const conv_config_t &cfg, tensor_kind_t kind) {
    switch (kind) {
        case tensor_kind_t::src: return cfg.src_layout().compute();
        case tensor_kind_t::wei: return cfg.wei_layout().compute();
        case tensor_kind_t::dst: return cfg.dst_layout().compute();
        default: ir_error_not_expected();
    }
    return cfg.src_layout().compute();
}

int get_layout_unit(const conv_config_t &cfg, const layout_t &layout,
        tensor_kind_t tensor_kind, const prb_dim_t &d) {
    auto &prb = cfg.prb();
    if (!is_reduction_dim(d, prb)) return 1;
    int dim_idx = tensor_conv_dim_index(d, tensor_kind);
    if (dim_idx == -1) return 1;

    std::vector<int> blocks;
    for (auto &b : layout.blocks()) {
        if (b.dim_idx == dim_idx) blocks.push_back(b.block);
    }
    if (blocks.size() <= 1) return 1;
    blocks.resize(blocks.size() - 1);

    int ret = 1;
    for (int b : blocks)
        ret *= b;
    return ret;
}

int get_layout_unit(const conv_config_t &cfg, const prb_dim_t &d) {
    int ret = 1;
    for (auto t :
            {tensor_kind_t::src, tensor_kind_t::wei, tensor_kind_t::dst}) {
        auto &l = compute_layout(cfg, t);
        ret = math::lcm(ret, get_layout_unit(cfg, l, t, d));
    }
    return ret;
}

bool is_mad_x8_non_dw(const conv_config_t &cfg) {
    auto &prb = cfg.prb();
    return (cfg.fma_kind() == fma_kind_t::mad) && !prb.is_dw
            && (prb.a_data_type_size == 1) && (prb.b_data_type_size == 1);
}

void get_level_tiles(int size0, int size1, const x2_tile_info_t &info,
        std::vector<level_tile_t> &ret0, std::vector<level_tile_t> &ret1) {
    ret0.clear();
    ret1.clear();
    auto tg_blocks = info.thread_group_blocks(size0, size1);
    for (auto &tg : tg_blocks) {
        int iter_size1 = utils::div_up(size0, tg.first);
        int iter_size0 = utils::div_up(size1, tg.second);
        auto iter_blocks = info.iter_blocks(iter_size0, iter_size1);
        for (auto &iter : iter_blocks) {
            int loop_size0 = utils::div_up(size0, tg.first * iter.first);
            int loop_size1 = utils::div_up(size1, tg.second * iter.second);
            auto loop_blocks = info.loop_blocks(loop_size0, loop_size1);
            for (auto &loop : loop_blocks) {
                level_tile_t t0;
                level_tile_t t1;
                if (any(info.flags & tile_flags_t::loop)) {
                    t0[levels::loop] = loop.first;
                    t1[levels::loop] = loop.second;
                }
                if (any(info.flags & tile_flags_t::thread_group)) {
                    t0[levels::thread_group] = tg.first;
                    t1[levels::thread_group] = tg.second;
                }
                if (any(info.flags & tile_flags_t::iter)) {
                    t0[levels::iter] = iter.first;
                    t1[levels::iter] = iter.second;
                }
                ret0.push_back(t0);
                ret1.push_back(t1);
            }
        }
    }
}

// Blocking scheme describing recipes to generate blockings.
class conv_blocking_scheme_t : public blocking_scheme_t {
public:
    conv_blocking_scheme_t() = default;
    conv_blocking_scheme_t(const std::string &s) : blocking_scheme_t(s) {}

    const std::vector<x2_tile_info_t> &x2_tile_infos() const {
        return x2_tile_infos_;
    }

    void finalize(const conv_config_t &cfg) {
        finalize_iter_units(cfg);
        finalize_iter_min_blocks(cfg);
        finalize_fused_reduction(cfg);
        finalize_loop_dims(cfg);
    }

    level_tile_set_t make_level_tile_set(
            const prb_tile_t &padded_shape) const override {
        const auto all_dims = dims();
        const int ndims = (int)all_dims.size();
        std::vector<int> deps(ndims, -1);
        std::vector<std::vector<level_tile_t>> tiles(ndims);

        auto to_idx = [&](const prb_dim_t &d) {
            for (int i = 0; i < ndims; i++)
                if (all_dims[i] == d) return i;
            ir_error_not_expected();
            return -1;
        };

        std::vector<bool> seen(ndims);
        for (auto &info : x2_tile_infos()) {
            int idx0 = to_idx(info.dim0);
            int idx1 = to_idx(info.dim1);
            get_level_tiles(padded_shape[info.dim0], padded_shape[info.dim1],
                    info, tiles[idx0], tiles[idx1]);
            ir_assert(!seen[idx0] && !seen[idx1]);
            seen[idx0] = seen[idx1] = true;
            deps[std::max(idx0, idx1)] = std::min(idx0, idx1);
        }
        for (int i = 0; i < ndims; i++) {
            if (seen[i]) continue;
            auto &d = all_dims[i];
            get_level_tiles(padded_shape[d], tile_info(d), tiles[i]);
        }
        return level_tile_set_t(tiles, deps, all_dims);
    }

private:
    void finalize_iter_units(const conv_config_t &cfg) {
        auto &prb = cfg.prb();
        bool is_dpas = cfg.is_dp_fma();
        int rdims = 0;
        for (auto &d : iter_) {
            if (is_reduction_dim(d, prb)) rdims++;
        }
        bool is_fused_reduction = (rdims > 1);
        for (auto &d : iter_) {
            auto &info = tile_info(d);
            int unit = 1;
            if (is_vectorized_dim(d, prb)) unit = cfg.vec_size();
            if (is_reduction_dim(d, prb)) {
                // This is to ensure that reduction-related address shifts are
                // constant. For example with a_blk = 8 and Ax16a layout there are two
                // kinds of "a" shifts: inside the innermost block and outer shift.
                int dpas_unit = (is_dpas ? 32 / prb.a_data_type_size : 1);
                int layout_unit = get_layout_unit(cfg, d);
                if (is_fused_reduction) {
                    // dpas unit is handled by finalize_fused_reduction().
                    unit = math::lcm(unit, layout_unit);
                } else if (layout_unit > dpas_unit
                        && (layout_unit % dpas_unit == 0)
                        && any(info.flags & tile_flags_t::loop_iter_unroll)) {
                    // Case of BWD_W with 32n16c layouts. 32n block is too
                    // large for iteration blocking so require loop unrolling
                    // via unroll unit.
                    info.set_unroll_unit(layout_unit);
                    unit = math::lcm(unit, dpas_unit);
                } else {
                    unit = math::lcm(unit, dpas_unit);
                    unit = math::lcm(unit, layout_unit);
                }
            }
            info.set_iter_unit(unit);
        }
    }

    void finalize_iter_min_blocks(const conv_config_t &cfg) {
        if (is_mad_x8_non_dw(cfg)) {
            auto &prb = cfg.prb();
            // Reduce min block size for mad/x8 as it requires a lot of
            // additional space for x8 -> s16 reorder.
            int min_m_iter_block_hint = 2;
            for (auto &d : iter_) {
                if (to_gemm(d, prb) != prb_dims::m) continue;
                auto &info = tile_info(d);
                int blk = std::min(info.min_iter_blk, min_m_iter_block_hint);
                int pow2_blk = utils::rnd_up_pow2(blk);
                info.set_min_iter_block(blk, pow2_blk);
            }
        }
    }

    void finalize_fused_reduction(const conv_config_t &cfg) {
        if (!cfg.is_dp_fma()) return;
        auto &prb = cfg.prb();
        int rdims = 0;
        prb_dim_t d0;
        prb_dim_t d1;
        for (auto &d : iter_) {
            if (!is_reduction_dim(d, prb)) continue;
            rdims++;
            (d0.is_undef() ? d0 : d1) = d;
        }
        if (rdims == 1) return;
        ir_assert(rdims == 2) << "Can't fuse more than two dimensions.";
        auto &info0 = tile_info(d0);
        auto &info1 = tile_info(d1);
        tile_flags_t flags = tile_flags_t::iter | tile_flags_t::loop
                | tile_flags_t::loop_span;
        ir_assert(info0.flags == flags);
        ir_assert(info1.flags == flags);
        ir_assert(info0.min_iter_blk == tile_info_t::default_min_iter_blk);
        ir_assert(info1.min_iter_blk == tile_info_t::default_min_iter_blk);

        int unit = 32 / prb.a_data_type_size;
        x2_tile_info_t x2_info(d0, d1);
        x2_info.add(flags);
        x2_info.set_iter_unit(unit);
        x2_info.d0 = info0.div_info;
        x2_info.d1 = info1.div_info;
        x2_tile_infos_.push_back(x2_info);
    }

    void finalize_loop_dims(const conv_config_t &cfg) {
        auto &prb = cfg.prb();
        if (prb.is_bwd_w) {
            struct loop_dim_t {
                prb_dim_t dim;
                int size = 0;

                static loop_dim_t *find(
                        const prb_dim_t &dim, std::vector<loop_dim_t> &dims) {
                    for (auto &d : dims)
                        if (d.dim == dim) return &d;
                    return nullptr;
                }
            };
            auto shape = cfg.shape(/*pad=*/true);
            std::vector<loop_dim_t> loop_dims;
            const int iter_dim_hint = 16;
            for (auto &d : loop_) {
                if (any(tile_info(d).flags & tile_flags_t::loop_iter_unroll))
                    continue;
                loop_dim_t ld;
                ld.dim = d;
                ld.size = shape.get(d, 1);
                if (iter_.has(d))
                    ld.size = utils::div_up(ld.size, iter_dim_hint);
                loop_dims.push_back(ld);
            }
            std::sort(loop_dims.begin(), loop_dims.end(),
                    [&](const loop_dim_t &a, const loop_dim_t &b) {
                        return a.size > b.size;
                    });

            // Deterministic mode doesn't allow reduction splitting between threadgroups.
            if (!prb.deterministic) {
                // For XeHPG and earlier hardware use only linear loops with SLM
                // pipelining to avoid overflowing icache. Prefetch pipeline can
                // handle nested loops without fully unrolling them.
                int max_loop_ndims = (cfg.hw() <= ngen::HW::XeHPG ? 1 : 2);
                for (int i = max_loop_ndims; i < (int)loop_dims.size(); i++)
                    loop_dims[i].dim = prb_dim_t();
            }

            for (auto &d : loop_) {
                auto &info = tile_info(d);
                if (any(info.flags & tile_flags_t::loop_iter_unroll)) continue;
                if (!loop_dim_t::find(d, loop_dims))
                    info.remove(tile_flags_t::loop);
            }
        }
    }

protected:
    std::vector<x2_tile_info_t> x2_tile_infos_;
};

int inner_block(const conv_config_t &cfg, tensor_kind_t tensor_kind,
        const prb_dim_t &dim) {
    int dim_idx = tensor_conv_dim_index(dim, tensor_kind);
    ir_assert(dim_idx != -1);
    auto &layout = compute_layout(cfg, tensor_kind);
    return layout.inner_block(
            dim_idx, /*skip_outer=*/true, /*inner_only=*/false);
}

int inner_block(const conv_config_t &cfg, const prb_dim_t &dim) {
    int ret = 0;
    for (auto t : input_tensors(cfg.prb())) {
        if (tensor_conv_dim_index(dim, t) == -1) continue;
        int blk = inner_block(cfg, t, dim);
        ret = (ret == 0 ? blk : math::gcd(ret, blk));
    }
    return ret == 0 ? 1 : ret;
}

dim_t inner_stride(const conv_config_t &cfg, tensor_kind_t tensor_kind,
        const prb_dim_t &dim) {
    int dim_idx = tensor_conv_dim_index(dim, tensor_kind);
    ir_assert(dim_idx != -1);
    auto &layout = compute_layout(cfg, tensor_kind);
    for (auto &b : layout.blocks()) {
        if (b.dim_idx == dim_idx) return (dim_t)b.stride;
    }
    return 0;
}

bool is_inner_non_blocked(const conv_config_t &cfg, const prb_dim_t &dim) {
    for (auto t : input_tensors(cfg.prb())) {
        if (tensor_conv_dim_index(dim, t) == -1) continue;
        if (inner_block(cfg, dim) != 1) continue;
        if (inner_stride(cfg, t, dim) == 1) return true;
    }
    return false;
}

int grf_usage_bytes(fma_kind_t fma, int b_iter, int m_iter, int n_iter,
        int k_iter, int a_type_size, int b_type_size, int c_type_size) {
    int a_elems = b_iter * m_iter * k_iter;
    int b_elems = b_iter * k_iter * n_iter;
    int c_elems = m_iter * n_iter;
    int a_size = a_elems * a_type_size;
    int a_reorder_size = 0;
    int b_size = b_elems * b_type_size;
    int b_reorder_size = 0;
    int c_size = c_elems * c_type_size;
    int dword_size = 4;
    if (fma == fma_kind_t::mad) {
        // mad/x8 case always requires x8 -> dword-strided s16 reorder.
        if (a_type_size == 1) a_reorder_size += a_elems * dword_size;
        if (b_type_size == 1) b_reorder_size += b_elems * dword_size;
    }

    int abc_size = 0;
    abc_size += a_size + a_reorder_size;
    abc_size += b_size + b_reorder_size;
    abc_size += c_size;
    return abc_size;
}

int slm_usage_bytes(const conv_config_t &cfg, int b_tg, int m_tg, int n_tg,
        int k_tg, int b_iter, int m_iter, int n_iter, int k_iter) {
    if (cfg.hw() >= ngen::HW::XeHPC) return 0;

    auto &prb = cfg.prb();
    bool slm_a = (n_tg != 1);
    bool slm_b = (m_tg != 1);
    int max_slm_bufs = 0;
    for (auto do_unroll : {false, true}) {
        int bufs = slm_bufs_hint(prb, m_tg, n_tg,
                cfg.zp_cfg().do_src_compensation, slm_a, slm_b, do_unroll);
        max_slm_bufs = std::max(max_slm_bufs, bufs);
    }
    int a_slm_elems = n_tg * b_iter * m_iter * k_iter;
    int b_slm_elems = m_tg * b_iter * n_iter * k_iter;
    int a_slm_size = a_slm_elems * prb.a_data_type_size;
    int b_slm_size = b_slm_elems * prb.b_data_type_size;
    int ab_slm_size = 0;
    if (slm_a) ab_slm_size += a_slm_size;
    if (slm_b) ab_slm_size += b_slm_size;
    int slm_size = max_slm_bufs * ab_slm_size;
    return slm_size;
}

int slm_usage_bytes_for_params(
        const conv_config_t &cfg, const blocking_params_t &params) {
    auto &prb = cfg.prb();
    auto tg = to_gemm(params.blocking().thread_group(), prb);
    auto iter = to_gemm(params.blocking().iter(), prb);
    int b_tg = tg.get(prb_dims::b, 1);
    int m_tg = tg.get(prb_dims::m, 1);
    int n_tg = tg.get(prb_dims::n, 1);
    int k_tg = tg.get(prb_dims::k, 1);
    int b_iter = iter.get(prb_dims::b, 1);
    int m_iter = iter.get(prb_dims::m, 1);
    int n_iter = iter.get(prb_dims::n, 1);
    int k_iter = iter.get(prb_dims::k, 1);
    return slm_usage_bytes(
            cfg, b_tg, m_tg, n_tg, k_tg, b_iter, m_iter, n_iter, k_iter);
}

class conv_blocking_checker_t : public blocking_checker_t {
public:
    conv_blocking_checker_t(const conv_config_t &cfg)
        : cfg_(cfg)
        , padded_shape_(cfg.shape(/*pad=*/true))
        , padded_gemm_shape_(to_gemm(padded_shape_, cfg.prb()))
        , max_tg_size_(
                  cfg.hw().max_tg_size(cfg.exec_cfg().regs(), cfg.simd())) {
        reset_checks();
    }

    void reset_checks() override {
        ir_assert((int)check_kind_t::_max < 64);

        check_mask_ = 0;
        optional_check_mask_ = 0;
        set_check(optional_check_mask_, check_kind_t::limit_k_iter);
        if (cfg_.prb().deterministic) {
            set_check(check_kind_t::check_deterministic);
        } else {
            set_check(optional_check_mask_,
                    check_kind_t::check_k_slicing_utilization);
            set_check(check_kind_t::check_k_slicing_utilization);
        }
        set_check(check_kind_t::check_vec);
        set_check(check_kind_t::check_tg_size);
        set_check(check_kind_t::check_dpas);
        set_check(check_kind_t::check_grf_usage);
        set_check(check_kind_t::check_slm_usage);
        set_check(check_kind_t::check_bwd_d_optimize);
        set_check(check_kind_t::check_layouts);
        set_check(check_kind_t::limit_m_iter);
        set_check(check_kind_t::limit_n_iter);
        set_check(check_kind_t::limit_k_iter);
    }

    bool relax_checks() override {
        for (int i = 0; i < (int)check_kind_t::_max; i++) {
            auto check = static_cast<check_kind_t>(i);
            if (!is_optional(check)) continue;
            if (!is_enabled(check)) continue;
            set_check(check, false);
            return true;
        }
        return false;
    }

    bool is_ok(const blocking_t &blk) const override {
        context_t ctx(blk, cfg_);
        if (!check_vec_ok(ctx)) return false;
        if (!check_tg_size_ok(ctx)) return false;
        if (!check_dpas_ok(ctx)) return false;
        if (!check_grf_usage_ok(ctx)) return false;
        if (!check_slm_usage_ok(ctx)) return false;
        if (!check_bwd_d_optimize_ok(ctx)) return false;
        if (!check_layouts_ok(ctx)) return false;
        if (!check_k_slicing_utilization_ok(ctx)) return false;
        if (!check_deterministic_ok(ctx)) return false;
        if (!limit_m_iter_ok(ctx)) return false;
        if (!limit_n_iter_ok(ctx)) return false;
        if (!limit_k_iter_ok(ctx)) return false;
        return true;
    }

private:
    struct context_t {
        context_t(const blocking_t &blk, const conv_config_t &cfg) : blk(blk) {
            auto &prb = cfg.prb();
            auto gemm_iter = to_gemm(blk.iter(), prb);
            auto gemm_loop = to_gemm(blk.loop(), prb);
            auto gemm_tg = to_gemm(blk.thread_group(), prb);
            b_iter = gemm_iter.get(prb_dims::b, 1);
            m_iter = gemm_iter.get(prb_dims::m, 1);
            n_iter = gemm_iter.get(prb_dims::n, 1);
            k_iter = gemm_iter.get(prb_dims::k, 1);
            k_loop = gemm_loop.get(prb_dims::k, 1);
            b_tg = gemm_tg.get(prb_dims::b, 1);
            m_tg = gemm_tg.get(prb_dims::m, 1);
            n_tg = gemm_tg.get(prb_dims::n, 1);
            k_tg = gemm_tg.get(prb_dims::k, 1);
            dpas_2x_depth = get_dpas_2x_depth(blk, cfg);
        }

        bool get_dpas_2x_depth(
                const blocking_t &blk, const conv_config_t &cfg) const {
            if (!cfg.is_dp_fma() || cfg.regs() <= 128) return false;

            // Use 2x reduction when the reduction dimension is dense to avoid
            // partial cache line loads.
            for (auto &d : blk.iter())
                if (is_reduction_dim(d, cfg.prb()))
                    if (is_inner_non_blocked(cfg, d)) return true;

            // Use larger reduction when M/N are small.
            int mn = m_iter * n_iter;
            if (mn <= 128) return true;

            return false;
        }

        blocking_t blk;
        int b_iter;
        int m_iter;
        int n_iter;
        int k_iter;
        int k_loop;
        int b_tg;
        int m_tg;
        int n_tg;
        int k_tg;

        bool dpas_2x_depth = false;
    };

    enum class check_kind_t : int {
        check_vec,
        check_tg_size,
        check_dpas,
        check_grf_usage,
        check_slm_usage,
        check_bwd_d_optimize,
        check_layouts,
        check_k_slicing_utilization,
        check_deterministic,
        limit_m_iter,
        limit_n_iter,
        limit_k_iter,
        _max
    };

    // Constant values.
    static const int dpas_reduce_bytes_ = 32;

    // Hint values.
    static const int max_mad_reduce_bytes_ = 64;
    static const int max_tg_dim_ = 8;
    static const int max_dpas_m_iter_ = 32;
    static const int max_dpas_n_iter_ = 64;
    static const int max_mad_m_iter_ = 16;
    static const int max_mad_n_iter_ = 32;
    static const int min_m_iter_ = 16;
    static const int min_mad_x8_non_dw_m_iter_ = 2;

    void set_check(check_kind_t check, bool value = true) {
        set_check(check_mask_, check, value);
    }

    void set_check(uint64_t &mask, check_kind_t check, bool do_set = true) {
        if (do_set) {
            mask |= (1ULL << (int)check);
        } else {
            mask &= ~(1ULL << (int)check);
        }
    }

    bool is_enabled(check_kind_t check) const {
        return (check_mask_ & (1ULL << (int)check)) != 0;
    }

    bool is_optional(check_kind_t check) const {
        return (optional_check_mask_ & (1ULL << (int)check)) != 0;
    }

    bool check_vec_ok(const context_t &ctx) const {
        if (!is_enabled(check_kind_t::check_vec)) return true;

        int vec_ndims = 0;
        for (auto &d : ctx.blk.iter()) {
            if (is_vectorized_dim(d, cfg_.prb())) vec_ndims++;
        }
        return vec_ndims == 1;
    }

    bool check_tg_size_ok(const context_t &ctx) const {
        if (!is_enabled(check_kind_t::check_tg_size)) return true;

        auto &tg = ctx.blk.thread_group();
        int tg_size = 1;
        int max_tg = 1;
        for (auto &d : tg) {
            tg_size *= tg[d];
            max_tg = std::max(tg[d], max_tg);
        }
        if (max_tg > max_tg_dim_) return false;
        if (tg_size > max_tg_size_) return false;
        return true;
    }

    bool check_dpas_ok(const context_t &ctx) const {
        if (!is_enabled(check_kind_t::check_dpas)) return true;

        if (!cfg_.is_dp_fma()) return true;
        int ab_size = cfg_.prb().a_data_type_size;
        int dpas_k = dpas_reduce_bytes_ / ab_size;
        return cfg_.prb().ab_swap_transpose ? ctx.n_iter % cfg_.simd() == 0
                                            : ctx.k_iter % dpas_k == 0;
    }

    bool check_grf_usage_ok(const context_t &ctx) const {
        if (!is_enabled(check_kind_t::check_grf_usage)) return true;

        auto &prb = cfg_.prb();
        int abc_size = grf_usage_bytes(cfg_.fma_kind(), ctx.b_iter, ctx.m_iter,
                ctx.n_iter, ctx.k_iter, prb.a_data_type_size,
                prb.b_data_type_size, prb.acc_data_type_size);
        auto &exec_cfg = cfg_.exec_cfg();
        int usage_limit = exec_cfg.grf_size()
                * (exec_cfg.regs() - cfg_.reserved_regs());
        return abc_size <= usage_limit;
    }

    bool check_slm_usage_ok(const context_t &ctx) const {
        if (!is_enabled(check_kind_t::check_slm_usage)) return true;

        int slm_size = slm_usage_bytes(cfg_, ctx.b_tg, ctx.m_tg, ctx.n_tg,
                ctx.k_tg, ctx.b_iter, ctx.m_iter, ctx.n_iter, ctx.k_iter);
        if (slm_size == 0) return true;

        auto &exec_cfg = cfg_.exec_cfg();
        int tg_size = ctx.b_tg * ctx.m_tg * ctx.n_tg * ctx.k_tg;
        int max_slm_size = compute::device_info_t::max_slm_size_per_tg(
                convert_ngen_arch_to_dnnl(cfg_.hw().to_ngen()), tg_size,
                exec_cfg.regs() > 128);
        if (slm_size > max_slm_size) return false;

        return true;
    }

    bool check_bwd_d_optimize_ok(const context_t &ctx) const {
        if (!is_enabled(check_kind_t::check_bwd_d_optimize)) return true;

        auto &prb = cfg_.prb();
        switch (cfg_.bwd_d_optimize_kind()) {
            case bwd_d_optimize_kind_t::none: return true;
            case bwd_d_optimize_kind_t::skip_out_of_bound_w: {
                int iw_iter = ctx.blk.iter().get(prb_dims::iw, 1);
                int iw_tg = ctx.blk.thread_group().get(prb_dims::iw, 1);
                if (iw_iter != 1 || iw_tg != 1) return false;
                return true;
            }
            case bwd_d_optimize_kind_t::skip_strided_dh: return true;
            case bwd_d_optimize_kind_t::skip_strided_dhw: {
                int iw_iter = ctx.blk.iter().get(prb_dims::iw, 1);
                if (iw_iter > 1) return false;
                int iw_tg = ctx.blk.thread_group().get(prb_dims::iw, 1);
                if (!math::is_pow2(iw_tg)) return false;
                if ((prb.iw / prb.sw) % iw_tg != 0) return false;
                return true;
            }
            default: ir_error_not_expected();
        }
        return false;
    }

    // Checks that the layout can be split based as required by level_blocks.
    static bool layout_dim_ok(prop_kind_t prop, tensor_kind_t tensor_kind,
            const layout_t &layout, const prb_dim_t &d,
            std::vector<std::pair<level_kind_t, int>> level_blocks) {
        if (level_blocks.empty()) return true;
        int dim_idx = tensor_conv_dim_index(d, tensor_kind);
        if (dim_idx == -1) return true;
        std::vector<int> blocks;
        for (auto &b : layout.blocks()) {
            if (b.dim_idx == dim_idx) blocks.push_back(b.block);
        }
        if (blocks.size() <= 1) return true;
        blocks.resize(blocks.size() - 1);
        auto step = [&](std::pair<level_kind_t, int> &kv) {
            int &block = kv.second;
            for (auto &b : blocks) {
                if (b == 1) continue;
                if (b % block == 0) {
                    b /= block;
                    block = 1;
                    return true;
                }
                if (block % b == 0) {
                    block /= b;
                    b = 1;
                    return true;
                }
                return false;
            }
            block = 1;
            return true;
        };
        for (auto &kv : level_blocks) {
            while (kv.second != 1)
                if (!step(kv)) return false;
        }
        return true;
    }

    static bool layout_ok(const blocking_t &blk, const layout_t &layout,
            prop_kind_t prop, tensor_kind_t tensor_kind) {
        for (int i = 0; i < prb_dim_t::max_id(); i++) {
            auto d = prb_dim_t::from_id(i);
            std::vector<std::pair<level_kind_t, int>> blocks;
            if (blk.iter().has(d))
                blocks.emplace_back(level_kind_t::iter, blk.iter_dim(d));
            if (blk.thread_group().has(d))
                blocks.emplace_back(
                        level_kind_t::thread_group, blk.thread_group_dim(d));
            if (!layout_dim_ok(prop, tensor_kind, layout, d, blocks))
                return false;
        }
        return true;
    }

    bool check_layouts_ok(const context_t &ctx) const {
        if (!is_enabled(check_kind_t::check_layouts)) return true;

        if (!layout_ok(ctx.blk, cfg_.src_layout().compute(),
                    cfg_.prb().prop_kind(), tensor_kind_t::src))
            return false;
        if (!layout_ok(ctx.blk, cfg_.wei_layout().compute(),
                    cfg_.prb().prop_kind(), tensor_kind_t::wei))
            return false;
        if (!layout_ok(ctx.blk, cfg_.dst_layout().compute(),
                    cfg_.prb().prop_kind(), tensor_kind_t::dst))
            return false;
        return true;
    }

    bool check_k_slicing_utilization_ok(const context_t &ctx) const {
        if (!is_enabled(check_kind_t::check_k_slicing_utilization)) return true;

        int b = padded_gemm_shape_.get(prb_dims::b, 1);
        int m = padded_gemm_shape_.get(prb_dims::m, 1);
        int n = padded_gemm_shape_.get(prb_dims::n, 1);
        int k = padded_gemm_shape_.get(prb_dims::k, 1);

        int64_t nthr = 1;
        nthr *= utils::div_up(b, ctx.b_iter);
        nthr *= utils::div_up(m, ctx.m_iter);
        nthr *= utils::div_up(n, ctx.n_iter);
        nthr *= utils::div_up(k, ctx.k_iter * ctx.k_loop);
        if (nthr < 16 && ctx.k_loop > 512) return false;

        return true;
    }

    bool check_deterministic_ok(const context_t &ctx) const {
        if (!is_enabled(check_kind_t::check_deterministic)) return true;
        int k = padded_gemm_shape_.get(prb_dims::k, 1);
        return ctx.k_loop * ctx.k_iter >= k;
    }

    int hint_min_m_iter() const {
        if (is_mad_x8_non_dw(cfg_)) return min_mad_x8_non_dw_m_iter_;
        return min_m_iter_;
    }

    int min_m_iter(const context_t &ctx) const {
        auto &prb = cfg_.prb();
        int max_blk = 1;
        for (auto &d : ctx.blk.iter()) {
            if (to_gemm(d, prb) == prb_dims::m) {
                int d_blk = inner_block(cfg_, d);
                max_blk = std::max(max_blk, d_blk);
            }
        }
        return std::min(hint_min_m_iter(), max_blk);
    }

    bool limit_m_iter_ok(const context_t &ctx) const {
        if (!is_enabled(check_kind_t::limit_m_iter)) return true;

        if (cfg_.is_dp_fma()) {
            if (ctx.m_iter > max_dpas_m_iter_) return false;
        } else {
            if (ctx.m_iter > max_mad_m_iter_) return false;
        }
        if (ctx.m_iter < min_m_iter(ctx)) return false;
        return true;
    }

    bool limit_n_iter_ok(const context_t &ctx) const {
        if (!is_enabled(check_kind_t::limit_n_iter)) return true;

        if (cfg_.is_dp_fma()) {
            if (ctx.n_iter > max_dpas_n_iter_) return false;
        } else {
            if (ctx.n_iter > max_mad_n_iter_) return false;
        }
        return true;
    }

    int min_k_iter(const context_t &ctx) const {
        if (!cfg_.is_dp_fma()) return 1;
        int type_size = cfg_.prb().a_data_type_size;
        int k_iter = dpas_reduce_bytes_ / type_size;
        if (ctx.dpas_2x_depth) k_iter *= 2;
        return k_iter;
    }

    int max_k_iter(const context_t &ctx) const {
        int type_size = cfg_.prb().a_data_type_size;
        if (!cfg_.is_dp_fma()) return max_mad_reduce_bytes_ / type_size;
        int k_iter = dpas_reduce_bytes_ / type_size;
        if (ctx.dpas_2x_depth) k_iter *= 2;
        return k_iter;
    }

    bool limit_k_iter_ok(const context_t &ctx) const {
        if (!is_enabled(check_kind_t::limit_k_iter)) return true;

        if (ctx.k_iter < min_k_iter(ctx)) return false;
        if (ctx.k_iter > max_k_iter(ctx)) return false;
        return true;
    }

    const conv_config_t &cfg_;
    const prb_tile_t padded_shape_;
    const prb_tile_t padded_gemm_shape_;
    const int max_tg_size_ = 0;

    uint64_t check_mask_ = 0;
    uint64_t optional_check_mask_ = 0;
};

// clang-format off
namespace conv_schemes {
// Conventions:
//   l    - loop dimension
//   T    - thread group dimension
//   i    - iteration dimension
//   ls   - loop dimension with span (loop block fully matches the remaining
//          dimension)
//   li   - loop dimension with unroll
//   #dim - remove minimum block restriction (minimum is 1)
conv_blocking_scheme_t fwd_T_wo_I_noi("ls:[ic,kd,kh,kw],T:[oc,ow],i:[mb,oc,ic]");
conv_blocking_scheme_t fwd_T_no_I_noi("ls:[ic,kd,kh,kw],T:[oc,mb],i:[mb,oc,ic]");
conv_blocking_scheme_t fwd_T_wn_I_wnoi("ls:[ic,kd,kh,kw],T:[ow,mb],i:[ow,mb,oc,ic]");
conv_blocking_scheme_t fwd_T_i_I_noi("ls:[ic,kd,kh,kw],T:[ic],i:[mb,oc,ic]");
conv_blocking_scheme_t fwd_T_iw_I_wnoi("ls:[ic,kd,kh,kw],T:[ic,ow],i:[ow,mb,oc,ic]");
conv_blocking_scheme_t fwd_T_wo_I_woi("ls:[ic,kd,kh,kw],T:[oc,ow],i:[ow,oc,ic]");
conv_blocking_scheme_t fwd_T_i_I_woi("ls:[ic,kd,kh,kw],T:[ic],i:[ow,oc,ic]");
conv_blocking_scheme_t fwd_T_wo_I_woki("ls:[ic,kd,kh,kw],T:[oc,ow],i:[ow,oc,kw,ic]");
conv_blocking_scheme_t fwd_T_w_I_woki("ls:[ic,kd,kh,kw],T:[ow],i:[ow,oc,kw,ic]");
conv_blocking_scheme_t fwd_T_w_I_noki("ls:[ic,kd,kh,kw],T:[ow],i:[mb,ow,oc,kw,ic]");
conv_blocking_scheme_t fwd_T_wo_I_noki("ls:[ic,kd,kh,kw],T:[oc,ow],i:[mb,oc,kw,ic]");
conv_blocking_scheme_t fwd_dw_T_w_I_wgk("ls:[kd,kh,kw],T:[ow],i:[ow,g,#kw]");
conv_blocking_scheme_t fwd_dw_T_w_I_ngk("ls:[kd,kh,kw],T:[ow],i:[mb,g,#kw]");
conv_blocking_scheme_t bwd_d_T_wi_I_nio("ls:[oc,kd,kh,kw],T:[ic,iw],i:[mb,ic,oc]");
conv_blocking_scheme_t bwd_d_T_ni_I_nio("ls:[oc,kd,kh,kw],T:[ic,mb],i:[mb,ic,oc]");
conv_blocking_scheme_t bwd_d_T_o_I_nio("ls:[oc,kd,kh,kw],T:[oc],i:[mb,ic,oc]");
conv_blocking_scheme_t bwd_d_T_w_I_on("ls:[oc,kd,kh,kw],T:[iw],i:[oc,mb]");
conv_blocking_scheme_t bwd_d_T_wi_I_wio("ls:[oc,kd,kh,kw],T:[ic,iw],i:[iw,ic,oc]");
conv_blocking_scheme_t bwd_d_T_o_I_wio("ls:[oc,kd,kh,kw],T:[oc],i:[iw,ic,oc]");
conv_blocking_scheme_t bwd_d_dw_T_w_I_wg("ls:[kd,kh,kw],T:[iw],i:[iw,g]");
conv_blocking_scheme_t bwd_d_dw_T_w_I_ng("ls:[kd,kh,kw],T:[iw],i:[mb,g]");
conv_blocking_scheme_t bwd_w_T_io_I_ion("l:[oh,ow],li:[mb],T:[oc,ic],i:[ic,oc,mb]");
conv_blocking_scheme_t bwd_w_T_io_I_ion_d("ls:[mb,od,oh,ow],T:[oc,ic],i:[ic,oc,mb]");
conv_blocking_scheme_t bwd_w_T_io_I_kon("l:[oh,ow],li:[mb],T:[oc,ic],i:[kw,oc,mb]");
conv_blocking_scheme_t bwd_w_T_io_I_ikon("l:[oh,ow],li:[mb],T:[oc,ic],i:[ic,kw,oc,mb]");
conv_blocking_scheme_t bwd_w_dw_I_gw("l:[mb,oh,ow],i:[g,ow]");
conv_blocking_scheme_t bwd_w_dw_I_gw_d("ls:[mb,od,oh,ow],i:[g,ow]");
conv_blocking_scheme_t bwd_w_dw_I_gn("l:[mb,oh,ow],i:[g,mb]");
conv_blocking_scheme_t bwd_w_dw_I_gn_d("ls:[mb,od,oh,ow],i:[g,mb]");
conv_blocking_scheme_t bwd_w_T_io_I_iow("l:[mb,oh,ow],T:[oc,ic],i:[ic,oc,ow]");
conv_blocking_scheme_t bwd_w_T_io_I_iow_d("ls:[mb,od,oh,ow],T:[oc,ic],i:[ic,oc,ow]");
conv_blocking_scheme_t bwd_w_T_io_I_ikow("l:[mb,oh,ow],T:[oc,ic],i:[ic,kw,oc,ow]");
} // namespace conv_schemes
// clang-format on

double get_iter_dim_score(
        const prb_dim_t &dim, const conv_config_t &cfg, int dim_size) {
    auto &prb = cfg.prb();
    if (utils::one_of(dim, prb_dims::ow, prb_dims::iw)) {
        if (prb.ksp > 1 || dim_size % 16 != 0) return 16 - 1;
        return dim_size;
    } else if (dim == prb_dims::mb) {
        return dim_size;
    } else {
        ir_error_not_expected() << "Unknown dimension: " << dim;
    }
    return 0;
}

prb_dim_t select_non_blocked_iter_dim(
        const conv_config_t &cfg, const std::vector<prb_dim_t> &dims) {
    const auto shape = cfg.shape(/*pad=*/false);
    std::vector<double> scores;
    for (auto &d : dims)
        scores.push_back(get_iter_dim_score(d, cfg, shape[d]));
    auto max_it = std::max_element(scores.begin(), scores.end());
    return dims[max_it - scores.begin()];
}

prb_dim_t select_iter_dim(
        const conv_config_t &cfg, const std::vector<prb_dim_t> &_dims) {
    bool is_bwd_d_w_opt = utils::one_of(cfg.bwd_d_optimize_kind(),
            bwd_d_optimize_kind_t::skip_strided_dhw,
            bwd_d_optimize_kind_t::skip_out_of_bound_w);
    std::vector<prb_dim_t> dims;
    for (auto &d : _dims) {
        if (is_bwd_d_w_opt && d == prb_dims::iw) continue;
        dims.push_back(d);
    }
    ir_assert(!dims.empty());
    if (dims.size() == 1) return dims[0];

    std::vector<int> dim_blocks;
    for (auto &d : dims) {
        dim_blocks.push_back(inner_block(cfg, d));
    }
    int max_block = *std::max_element(dim_blocks.begin(), dim_blocks.end());
    if (max_block == 1) return select_non_blocked_iter_dim(cfg, dims);
    for (int i = 0; i < (int)dims.size(); i++) {
        if (dim_blocks[i] == max_block) return dims[i];
    }
    ir_error_not_expected();
    return *dims.begin();
}

using conv_blocking_scheme_list_t
        = blocking_scheme_list_impl_t<conv_blocking_scheme_t>;

conv_blocking_scheme_list_t get_blocking_schemes_fwd_dw(
        const conv_config_t &cfg) {
    conv_blocking_scheme_list_t ret(conv_tune_level());
    auto m_iter_dim = select_iter_dim(cfg, {prb_dims::mb, prb_dims::ow});
    bool m_is_mb = (m_iter_dim == prb_dims::mb);
    bool m_is_ow = (m_iter_dim == prb_dims::ow);
    ret.add(m_is_mb, conv_schemes::fwd_dw_T_w_I_ngk);
    ret.add(m_is_ow, conv_schemes::fwd_dw_T_w_I_wgk);
    return ret;
}

conv_blocking_scheme_list_t get_blocking_schemes_bwd_d_dw(
        const conv_config_t &cfg) {
    conv_blocking_scheme_list_t ret(conv_tune_level());
    auto m_iter_dim = select_iter_dim(cfg, {prb_dims::mb, prb_dims::iw});
    bool m_is_mb = (m_iter_dim == prb_dims::mb);
    bool m_is_iw = (m_iter_dim == prb_dims::iw);
    ret.add(m_is_mb, conv_schemes::bwd_d_dw_T_w_I_ng);
    ret.add(m_is_iw, conv_schemes::bwd_d_dw_T_w_I_wg);
    return ret;
}

conv_blocking_scheme_list_t get_blocking_schemes_bwd_w_dw(
        const conv_config_t &cfg) {
    conv_blocking_scheme_list_t ret(conv_tune_level());
    auto k_iter_dim = select_iter_dim(cfg, {prb_dims::mb, prb_dims::ow});
    bool k_is_mb = (k_iter_dim == prb_dims::mb);
    bool k_is_ow = (k_iter_dim == prb_dims::ow);
    ret.add(k_is_mb, conv_schemes::bwd_w_dw_I_gn);
    ret.add(k_is_ow, conv_schemes::bwd_w_dw_I_gw);
    ret.add(k_is_mb && cfg.prb().deterministic, conv_schemes::bwd_w_dw_I_gn_d);
    ret.add(k_is_ow && cfg.prb().deterministic, conv_schemes::bwd_w_dw_I_gw_d);
    return ret;
}

conv_blocking_scheme_list_t get_blocking_schemes_fwd(const conv_config_t &cfg) {
    conv_blocking_scheme_list_t ret(conv_tune_level());
    auto m_iter_dim = cfg.prb().ab_swap_transpose
            ? prb_dims::oc
            : select_iter_dim(cfg, {prb_dims::mb, prb_dims::ow});
    bool m_is_mb = (m_iter_dim == prb_dims::mb);
    bool m_is_ow = (m_iter_dim == prb_dims::ow);
    bool m_is_oc = (m_iter_dim == prb_dims::oc);
    bool ge_xelp = (cfg.hw() >= ngen::HW::XeLP);
    bool small_ic = (is_small_ic(cfg.prb()) && cfg.prb().kw > 1);
    ret.add(m_is_mb, conv_schemes::fwd_T_wo_I_noi);
    ret.add(m_is_mb, conv_schemes::fwd_T_no_I_noi);
    ret.add(m_is_mb && ge_xelp, conv_schemes::fwd_T_i_I_noi);
    ret.add(m_is_oc, conv_schemes::fwd_T_wn_I_wnoi);
    ret.add(m_is_oc && ge_xelp, conv_schemes::fwd_T_i_I_noi);
    ret.add(m_is_oc && ge_xelp, conv_schemes::fwd_T_iw_I_wnoi);
    ret.add(m_is_ow, conv_schemes::fwd_T_wo_I_woi);
    ret.add(m_is_ow && ge_xelp, conv_schemes::fwd_T_i_I_woi);
    ret.add(m_is_mb && small_ic, conv_schemes::fwd_T_wo_I_noki);
    ret.add(m_is_oc && small_ic, conv_schemes::fwd_T_w_I_woki);
    ret.add(m_is_oc && small_ic, conv_schemes::fwd_T_w_I_noki);
    ret.add(m_is_ow && small_ic, conv_schemes::fwd_T_wo_I_woki);
    return ret;
}

conv_blocking_scheme_list_t get_blocking_schemes_bwd_d(
        const conv_config_t &cfg) {
    conv_blocking_scheme_list_t ret(conv_tune_level());
    auto m_iter_dim = cfg.prb().ab_swap_transpose
            ? prb_dims::ic
            : select_iter_dim(cfg, {prb_dims::mb, prb_dims::iw});
    bool m_is_mb = (m_iter_dim == prb_dims::mb);
    bool m_is_iw = (m_iter_dim == prb_dims::iw);
    bool m_is_ic = (m_iter_dim == prb_dims::ic);
    bool ge_xelp = (cfg.hw() >= ngen::HW::XeLP);
    ret.add(m_is_mb, conv_schemes::bwd_d_T_ni_I_nio);
    ret.add(m_is_mb, conv_schemes::bwd_d_T_wi_I_nio);
    ret.add(m_is_mb && ge_xelp, conv_schemes::bwd_d_T_o_I_nio);
    ret.add(m_is_iw, conv_schemes::bwd_d_T_wi_I_wio);
    ret.add(m_is_iw && ge_xelp, conv_schemes::bwd_d_T_o_I_wio);
    ret.add(m_is_ic, conv_schemes::bwd_d_T_w_I_on);
    return ret;
}

conv_blocking_scheme_list_t get_blocking_schemes_bwd_w(
        const conv_config_t &cfg) {
    conv_blocking_scheme_list_t ret(conv_tune_level());
    auto k_iter_dim = select_iter_dim(cfg, {prb_dims::mb, prb_dims::ow});
    bool k_is_mb = (k_iter_dim == prb_dims::mb);
    bool k_is_ow = (k_iter_dim == prb_dims::ow);
    bool small_ic = is_small_ic(cfg.prb());
    ret.add(k_is_mb, conv_schemes::bwd_w_T_io_I_ion);
    ret.add(k_is_ow, conv_schemes::bwd_w_T_io_I_iow);
    ret.add(k_is_mb && small_ic, conv_schemes::bwd_w_T_io_I_kon);
    ret.add(k_is_mb && small_ic, conv_schemes::bwd_w_T_io_I_ikon);
    ret.add(k_is_ow && small_ic, conv_schemes::bwd_w_T_io_I_ikow);
    ret.add(cfg.prb().deterministic, conv_schemes::bwd_w_T_io_I_ion_d);
    ret.add(cfg.prb().deterministic, conv_schemes::bwd_w_T_io_I_iow_d);
    return ret;
}

conv_blocking_scheme_list_t get_blocking_schemes_dw_impl(
        const conv_config_t &cfg) {
    auto &prb = cfg.prb();
    if (prb.is_fwd) return get_blocking_schemes_fwd_dw(cfg);
    if (prb.is_bwd_d) return get_blocking_schemes_bwd_d_dw(cfg);
    if (prb.is_bwd_w) return get_blocking_schemes_bwd_w_dw(cfg);
    ir_error_not_expected();
    return conv_blocking_scheme_list_t();
}

conv_blocking_scheme_list_t get_blocking_schemes_impl(
        const conv_config_t &cfg) {
    auto &prb = cfg.prb();
    if (prb.is_dw) return get_blocking_schemes_dw_impl(cfg);
    if (prb.is_fwd) return get_blocking_schemes_fwd(cfg);
    if (prb.is_bwd_d) return get_blocking_schemes_bwd_d(cfg);
    if (prb.is_bwd_w) return get_blocking_schemes_bwd_w(cfg);
    ir_error_not_expected();
    return conv_blocking_scheme_list_t();
}

std::vector<conv_blocking_scheme_t> get_blocking_schemes(
        const conv_config_t &cfg) {
    auto ret = get_blocking_schemes_impl(cfg).get();
    for (auto &s : ret)
        s.finalize(cfg);
    return ret;
}

} // namespace

int grf_usage_bytes(const conv_config_t &cfg, const blocking_params_t &params) {
    auto &prb = cfg.prb();
    auto iter = to_gemm(params.blocking().iter(), prb);
    int b_iter = iter.get(prb_dims::b, 1);
    int m_iter = iter.get(prb_dims::m, 1);
    int n_iter = iter.get(prb_dims::n, 1);
    int k_iter = iter.get(prb_dims::k, 1);
    int abc_size = grf_usage_bytes(cfg.fma_kind(), b_iter, m_iter, n_iter,
            k_iter, prb.a_data_type_size, prb.b_data_type_size,
            prb.acc_data_type_size);
    return abc_size;
}

void sort_by_model_scores(params_generator_t &params_gen,
        const conv_config_t &cfg, tiler_mode_t mode) {
    std::unordered_map<int, float> eff_scores;
    for (int i = 0; i < params_gen.configs(); i++) {
        auto &p = params_gen.at(i);
        float score = model::get_score(cfg, p);
        float eff = p.blocking().get_efficiency(cfg.shape(/*pad=*/true));
        eff_scores.emplace(p.id(), score * eff);
    }
    if (mode == tiler_mode_t::lookup) {
        // Give the lookup entry the highest score.
        eff_scores[params_gen.at(0).id()] = 1.0f;
    }
    params_gen.sort(0, params_gen.configs(),
            [&](const blocking_params_t &p) { return -eff_scores.at(p.id()); });
#ifdef DNNL_DEV_MODE
    using namespace ir_utils;
    std::vector<std::string> headers
            = {"Config", "Score", "Eff", "Regs", "SLM size"};
    table_t table("List of configs", headers);
    for (auto &p : params_gen.params_vec()) {
        float score = model::get_score(cfg, p);
        float eff = p.blocking().get_efficiency(cfg.shape(/*pad=*/true));
        int regs = utils::div_up(grf_usage_bytes(cfg, p), cfg.grf_size());
        int slm_size = slm_usage_bytes_for_params(cfg, p);
        table << p.str() << (int)(score * 1000) / 1000.0 << eff << regs
              << slm_size << std::endl;
    }
    ir_trace() << table.str() << std::endl;
#endif
    MAYBE_UNUSED(&slm_usage_bytes_for_params);
}

// Tuner class.
class conv_tuner_t {
public:
    struct primitive_info_t {
        const conv_tiler_t *impl = nullptr;
        conv_key_t key;
        blocking_params_t params;
    };

    int configs() const { return 0; }

    void set_params(prim_config_t &cfg) { params_gen_.set_params(cfg); }

    void notify_create(
            const conv_config_t &cfg, const impl::primitive_t *primitive) {
        std::lock_guard<std::mutex> lock(mutex_);
        created_configs_++;
        auto &info = primitive_infos_[primitive];
        info.key = cfg.key();
        const auto undef = blocking_params_t::bufs_hint_undef;
        info.params = cfg.params((!cfg.slm() && !cfg.prefetch()) ? 0 : undef);
    }

    void set_profile_info(uint64_t stamp, const blocking_params_t &params) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto &info = stamp_infos_[stamp];
        info.tuner = this;
        info.params = params;
    }

    void add_time(const blocking_params_t &params, uint64_t cur_nsec) {
        tune_data_.add_time(params.id(), cur_nsec);
    }

    void finalize(const blocking_params_t &params) {
        bool is_best = (tune_data_.best_id() == params.id());
        if (is_best) {
            conv_lookup_table().set(key_.to_filter(), params);
            best_params_dbg_ = params;
        }
        uint64_t nsec = tune_data_.nsec(params.id());
        double gops_sec = ops_ / nsec;
        maybe_print_header();
        std::ostringstream oss;
        oss << "perf,conv,";
        oss << key_.str(/*csv=*/true) << ",";
        oss << params.str(/*csv=*/true) << ",";
        oss << nsec << ",";
        oss << gops_sec << std::endl;
        std::cout << oss.str();
        if (is_best) std::cout << "onednn_tune_best,conv," << oss.str();
        maybe_rescore();
    }

    bool can_move_next() const { return params_gen_.can_move_next(); }

    void move_next() {
        ir_assert(can_move_next());
        params_gen_.move_next();
    }

    void print_all() const { params_gen_.print_all(); }

    static const primitive_info_t &get_primitive_info(
            const impl::primitive_t *primitive) {
        std::lock_guard<std::mutex> lock(mutex_);
        return primitive_infos_.at(primitive);
    }

    static conv_tuner_t *get_tuner(const conv_key_t &key, bool do_lock = true) {
        std::unique_lock<std::mutex> lock(mutex_, std::defer_lock_t());
        if (do_lock) lock.lock();
        auto it = conv2tuner_.find(key);
        return it != conv2tuner_.end() ? &it->second : nullptr;
    }

    static conv_tuner_t *get_tuner(int tune_level, int simd_size,
            blocking_checker_t &chk,
            const std::vector<level_tile_set_t> &level_tile_sets,
            const conv_key_t &key, double ops,
            const std::function<prb_tile_t(const prb_tile_t &)> &convert,
            bool create_if_not_found = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto *tuner = get_tuner(key, /*do_lock=*/false);
        if (tuner) return tuner;
        if (!create_if_not_found) return nullptr;

        params_generator_t params_gen(
                tune_level, simd_size, chk, level_tile_sets);
        params_distance_t dist(params_gen.params_vec(), convert);
        auto ret = conv2tuner_.emplace(
                key, conv_tuner_t(key, ops, params_gen, dist));
        return &ret.first->second;
    }

    static void profile_callback(uint64_t stamp, uint64_t nsec) {
        bool is_complete = (nsec == std::numeric_limits<uint64_t>::max());
        if (is_complete) {
            std::unordered_set<int> seen;
            for (auto &kv : stamp_infos_) {
                auto &info = kv.second;
                int id = info.params.id();
                if (seen.count(id) > 0) continue;
                info.tuner->finalize(info.params);
                seen.insert(id);
            }
            stamp_infos_.clear();
            return;
        }
        auto it = stamp_infos_.find(stamp);
        if (it == stamp_infos_.end()) return;
        auto &info = it->second;
        info.tuner->add_time(info.params, nsec);
    }

private:
    conv_tuner_t(const conv_key_t &key, double ops,
            params_generator_t params_gen, params_distance_t params_dist)
        : key_(key)
        , params_gen_(std::move(params_gen))
        , params_dist_(std::move(params_dist))
        , ops_(ops) {
        params_gen_.shuffle(conv_key_hash_t()(key_));
    }

    struct stamp_info_t {
        conv_tuner_t *tuner;
        blocking_params_t params;
    };

    static void maybe_print_header() {
        static bool printed = false;
        if (printed) return;
        std::cout << ir_utils::make_seq_print_helper(tune_csv_keys(), ",")
                  << std::endl;
        printed = true;
    }

    static std::vector<std::string> tune_csv_keys() {
        std::vector<std::string> ret;
        ret.emplace_back("perf");
        ret.emplace_back("conv");
        for (auto &k : conv_key_t::csv_keys())
            ret.push_back(k);
        for (auto &k : blocking_params_t::csv_keys())
            ret.push_back(k);
        ret.emplace_back("nsec");
        ret.emplace_back("gops_sec");
        return ret;
    }

    void maybe_rescore() {
        if (tune_data_.reported_points() != created_configs_) return;
        if (created_configs_ < 8) return;

        int beg = params_gen_.cur_index() + 1;
        int end = params_gen_.configs();
        if (beg == end) return;

        const int nbest = 5;
        auto best_ids = tune_data_.best_ids(nbest);
        std::unordered_map<int, float> dists;
        ir_trace() << "[Tuning] Rescoring: " << (end - beg) << " configs left"
                   << std::endl;
        ir_trace() << "  Best config: " << best_params_dbg_ << std::endl;
        for (int i = beg; i < end; i++) {
            auto &p = params_gen_.at(i);
            dists[p.id()] = std::numeric_limits<float>::max();
            for (int id : best_ids) {
                float d = params_dist_.dist(id, p.id());
                dists[p.id()] = std::min(dists[p.id()], d);
            }
        }
        params_gen_.sort(beg, end,
                [&](const blocking_params_t &p) { return dists.at(p.id()); });

        for (int i = beg; i < end; i++) {
            auto &p = params_gen_.at(i);
            ir_trace() << "  " << p << " [dist:" << dists[p.id()] << "]"
                       << std::endl;
        }
    }

    conv_key_t key_;
    params_generator_t params_gen_;
    const params_distance_t params_dist_;
    tune_data_t tune_data_;
    blocking_params_t best_params_dbg_;

    int created_configs_ = 0;
    double ops_ = 0;

    static std::unordered_map<conv_key_t, conv_tuner_t, conv_key_hash_t>
            conv2tuner_;
    static std::unordered_map<uint64_t, stamp_info_t> stamp_infos_;
    static std::unordered_map<const impl::primitive_t *,
            conv_tuner_t::primitive_info_t>
            primitive_infos_;
    static std::mutex mutex_;
};

std::unordered_map<conv_key_t, conv_tuner_t, conv_key_hash_t>
        conv_tuner_t::conv2tuner_;
std::unordered_map<uint64_t, conv_tuner_t::stamp_info_t>
        conv_tuner_t::stamp_infos_;
std::unordered_map<const impl::primitive_t *, conv_tuner_t::primitive_info_t>
        conv_tuner_t::primitive_infos_;

std::mutex conv_tuner_t::mutex_;

class conv_tiler_impl_t {
public:
    conv_tiler_impl_t() = default;
    conv_tiler_impl_t(const conv_config_t &cfg) {
        double init_time_ms = 0;
#ifdef DNNL_DEV_MODE
        init_time_ms = get_msec();
#endif
        init(cfg);
#ifdef DNNL_DEV_MODE
        init_time_ms = get_msec() - init_time_ms;
#endif
        print_info(init_time_ms);
    }

    int configs() const {
        if (is_tuning_mode()) return tuner_->configs();
        return params_gen_.configs();
    }

    bool is_tuning_mode() const { return tuner_; }

    bool can_move_next() const {
        if (is_tuning_mode()) return tuner_->can_move_next();
        return params_gen_.can_move_next();
    }

    void set_params(conv_config_t &cfg) {
        init_regs(cfg);
        if (is_tuning_mode()) {
            tuner_->move_next();
            tuner_->set_params(cfg);
        } else {
            if (!try_small_grf_) params_gen_.move_next();
            params_gen_.set_params(cfg);
            maybe_try_small_grf(cfg);
        }
    }

    void notify_out_of_registers(const conv_config_t &cfg) {
        if (is_tuning_mode() || cfg.regs() != default_regs(cfg)) return;
        grf_usage_limit_ = estimate_register_count(cfg) * cfg.grf_size();
    }

    bool is_grf_limit_ok(const conv_config_t &cfg) const {
        if (is_tuning_mode() || grf_usage_limit_ == 0) return true;
        int cur_usage_bytes = estimate_register_count(cfg) * cfg.grf_size();
        return cur_usage_bytes < grf_usage_limit_;
    }

    void print_all() const {
        if (is_tuning_mode()) {
            tuner_->print_all();
        } else {
            params_gen_.print_all();
        }
    }

private:
    void init(const conv_config_t &cfg) {
        if (cfg.loop_dims().is_overridden()
                || cfg.thread_group_dims().is_overridden()
                || cfg.iter_dims().is_overridden()) {
            mode_ = tiler_mode_t::env_config;
        } else {
            mode_ = tiler_params().mode;
        }

        auto padded_shape = cfg.shape(/*pad=*/true);
        std::vector<level_tile_set_t> level_tile_sets;
        for (auto &s : get_blocking_schemes(cfg))
            level_tile_sets.emplace_back(s.make_level_tile_set(padded_shape));
        auto try_cfg = cfg;
        init_regs(try_cfg);
        conv_blocking_checker_t chk(try_cfg);
        const int simd_size = cfg.vec_size();
        const int tune_level = conv_tune_level();

        switch (mode_) {
            case tiler_mode_t::env_config:
                params_gen_ = params_generator_t(
                        tune_level, simd_size, chk, level_tile_sets);
                break;
            case tiler_mode_t::env_tiler:
                params_gen_ = params_generator_t(tune_level, simd_size, chk,
                        level_tile_sets, tiler_params().env_params_idx);
                break;
                break;
            case tiler_mode_t::lookup: {
                const auto params = const_conv_lookup_table().find(cfg.key());
                if (!params.is_empty() && chk.is_ok(params.blocking())) {
                    params_gen_ = params_generator_t(tune_level, simd_size, chk,
                            level_tile_sets, params);
                } else {
                    mode_ = tiler_mode_t::model;
                    params_gen_ = params_generator_t(
                            tune_level, simd_size, chk, level_tile_sets);
                }
                break;
            }
            case tiler_mode_t::tune: {
                auto convert = [&](const prb_tile_t &tile) {
                    return to_gemm(tile, cfg.prb());
                };
                tuner_ = conv_tuner_t::get_tuner(tune_level, simd_size, chk,
                        level_tile_sets, cfg.key(), cfg.prb().ops(), convert,
                        /*create_if_not_found=*/true);
                break;
            }
            default:
                params_gen_ = params_generator_t(
                        tune_level, simd_size, chk, level_tile_sets);
                break;
        }
        if (!is_tuning_mode()) {
            ir_assert(!params_gen_.is_empty()) << "No configurations found.";
            sort_by_model_scores(params_gen_, cfg, mode_);
        }
        if (tiler_params().do_list) print_all();
    }

    void maybe_try_small_grf(conv_config_t &cfg) {
        auto try_cfg = cfg;
        init_walk_order(try_cfg);
        init_kernel_grid(try_cfg);
        init_thread_group_grid(try_cfg);
        int kg_elems = try_cfg.kernel_grid().elems(),
            tg_elems = try_cfg.thread_group_grid().elems();
        try_cfg.set_regs(128);
        int new_wave_util = conv_config_t::get_wave_utilization(
                try_cfg.exec_cfg(), kg_elems, tg_elems);
        int wave_util = conv_config_t::get_wave_utilization(
                cfg.exec_cfg(), kg_elems, tg_elems);
        if (wave_util > 90 && new_wave_util >= wave_util && !try_small_grf_
                && cfg.regs() > 128
                && !cfg.exec_cfg_param().is_overridden("regs")) {
            cfg.set_regs(128);
            try_small_grf_ = true;
        } else {
            try_small_grf_ = false;
        }
    }

    void print_info(double init_time_ms) {
        ir_info() << "Convolution tiler:" << std::endl;
        ir_info() << "  Mode:              " << to_string(mode_) << std::endl;
        ir_info() << "  Filtered configs:  " << configs() << std::endl;
        ir_info() << "  Init time (ms):    " << init_time_ms << std::endl;
    }

    tiler_mode_t mode_ = tiler_mode_t::undef;
    params_generator_t params_gen_;
    conv_tuner_t *tuner_ = nullptr;
    int grf_usage_limit_ = 0;
    bool try_small_grf_ = false;
};

conv_tiler_t::conv_tiler_t(const conv_config_t &cfg)
    : impl_(std::make_shared<conv_tiler_impl_t>(cfg)) {}

int conv_tiler_t::configs() const {
    return impl_->configs();
}

bool conv_tiler_t::is_tuning_mode() const {
    return impl_->is_tuning_mode();
}

bool conv_tiler_t::can_move_next() const {
    return impl_->can_move_next();
}

void conv_tiler_t::set_params(conv_config_t &cfg) {
    impl_->set_params(cfg);
}

void conv_tiler_t::notify_out_of_registers(const conv_config_t &cfg) {
    impl_->notify_out_of_registers(cfg);
}

bool conv_tiler_t::is_grf_limit_ok(const conv_config_t &cfg) const {
    return impl_->is_grf_limit_ok(cfg);
}
void conv_tiler_t::after_create_hook(
        const conv_config_t &cfg, const impl::primitive_t *primitive) {
    if (!cfg.tiler().is_tuning_mode()) return;
    auto *tuner = conv_tuner_t::get_tuner(cfg.key());
    tuner->notify_create(cfg, primitive);
}

void conv_tiler_t::before_exec_hook(
        const impl::primitive_t *primitive, impl::stream_t *stream) {
    if (tiler_params().mode != tiler_mode_t::tune) return;
    if (!stream->is_profiling_enabled()) return;
    auto &info = conv_tuner_t::get_primitive_info(primitive);
    auto *tuner = conv_tuner_t::get_tuner(info.key);
    auto *compute_stream = utils::downcast<compute::compute_stream_t *>(stream);
    auto &profiler = compute_stream->profiler();
    tuner->set_profile_info(profiler.stamp(), info.params);
    profiler.set_callback(conv_tuner_t::profile_callback);
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
