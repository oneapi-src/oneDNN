/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gpu/jit/conv/tiler.hpp"

#include <cmath>
#include <mutex>
#include <random>
#include <unordered_set>

#include "gpu/compute/stream_profiler.hpp"
#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/lookup_table.hpp"
#include "gpu/jit/conv/model_bridge.hpp"
#include "gpu/jit/conv/params.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Tile level kinds.
enum class level_kind_t {
    undef = 0,
    loop,
    thread_group,
    iter,
    _max,
};

std::string to_string(level_kind_t kind) {
    std::ostringstream oss;
    switch (kind) {
#define CASE(name, value) \
    case level_kind_t::name: return value
        CASE(loop, "l");
        CASE(thread_group, "T");
        CASE(iter, "i");
#undef CASE
        default: ir_error_not_expected();
    }
    return oss.str();
}

namespace {

using level_t = tile_key_t<level_kind_t>;
using level_tile_t = tile_generic_t<level_t>;

namespace levels {
level_t loop(level_kind_t::loop);
level_t thread_group(level_kind_t::thread_group);
level_t iter(level_kind_t::iter);
}; // namespace levels

enum class tensor_kind_t {
    undef,
    src,
    wei,
    dst,
};

std::vector<tensor_kind_t> input_tensors(const conv_config_t &cfg) {
    std::vector<tensor_kind_t> ret;
    auto &prb = cfg.prb();
    if (prb.is_fwd || prb.is_bwd_w) ret.push_back(tensor_kind_t::src);
    if (prb.is_fwd || prb.is_bwd_d) ret.push_back(tensor_kind_t::wei);
    if (prb.is_bwd_d || prb.is_bwd_w) ret.push_back(tensor_kind_t::dst);
    return ret;
}

// Flags specifying blocking restrictions for a convolution dimension.
enum class tile_flags_t : uint32_t {
    undef = 0,
    // Dimension participates in loop blocking.
    loop = (1 << 0),
    // Dimension participates in thread group blocking.
    thread_group = (1 << 1),
    // Dimension participates in iteration blocking.
    iter = (1 << 2),
    // Loop block spans the remaining dimension.
    loop_span = (1 << 3),
    // Loop block is fully unrolled.
    loop_iter_unroll = (1 << 4),
};

tile_flags_t operator&(tile_flags_t a, tile_flags_t b) {
    auto _a = static_cast<uint32_t>(a);
    auto _b = static_cast<uint32_t>(b);
    return static_cast<tile_flags_t>(_a & _b);
}

tile_flags_t operator|(tile_flags_t a, tile_flags_t b) {
    auto _a = static_cast<uint32_t>(a);
    auto _b = static_cast<uint32_t>(b);
    return static_cast<tile_flags_t>(_a | _b);
}

tile_flags_t operator~(tile_flags_t a) {
    auto _a = static_cast<uint32_t>(a);
    return static_cast<tile_flags_t>(~_a);
}

bool any(tile_flags_t a) {
    return a != tile_flags_t::undef;
}

bool is_reduction_dim(
        const conv_dim_t &d, prop_kind_t prop, bool is_transpose) {
    return to_gemm(d, prop, is_transpose) == gemm_dims::k;
}

bool is_vectorized_dim(const conv_dim_t &d, const conv_problem_t &prb) {
    if (prb.is_dw) return d == conv_dims::g;
    bool transpose = prb.ab_swap_transpose;
    switch (prb.prop_kind()) {
        case prop_kind::forward:
            return (transpose ? d == conv_dims::mb : d == conv_dims::oc);
        case prop_kind::backward_data:
            return (transpose ? d == conv_dims::mb : d == conv_dims::ic);
        case prop_kind::backward_weights:
            return (transpose ? d == conv_dims::ic : d == conv_dims::oc);
        default: ir_error_not_expected();
    }
    return false;
}

int tensor_conv_dim_index(const conv_dim_t &d, tensor_kind_t t) {
    using namespace conv_dims;
    std::vector<conv_dim_t> src_dims = {mb, g, ic, id, ih, iw};
    std::vector<conv_dim_t> wei_dims = {g, oc, ic, kd, kh, kw};
    std::vector<conv_dim_t> dst_dims = {mb, g, oc, od, oh, ow};
    std::vector<conv_dim_t> *conv_dims = nullptr;
    switch (t) {
        case tensor_kind_t::src: conv_dims = &src_dims; break;
        case tensor_kind_t::wei: conv_dims = &wei_dims; break;
        case tensor_kind_t::dst: conv_dims = &dst_dims; break;
        default: ir_error_not_expected();
    }
    auto it = std::find(conv_dims->begin(), conv_dims->end(), d);
    if (it == conv_dims->end()) return -1;
    return (int)(it - conv_dims->begin());
}

std::vector<conv_dim_t> all_conv_dims() {
    std::vector<conv_dim_t> ret;
    for (int i = 0; i < conv_dim_t::max_id(); i++) {
        auto d = conv_dim_t::from_id(i);
        if (d.is_undef()) continue;
        ret.push_back(d);
    }
    return ret;
}

std::vector<int> get_factors(int n) {
    std::vector<int> ret;
    int n_sqrt = (int)std::sqrt(n);
    for (int i = 1; i <= n_sqrt; i++) {
        if (n % i == 0) ret.push_back(i);
    }
    int lo = n_sqrt;
    if (n_sqrt * n_sqrt == n) lo--;
    for (int i = lo; i >= 1; i--) {
        if (n % i == 0) ret.push_back(n / i);
    }
    return ret;
}

std::vector<int> get_loop_blocks(int n) {
    const int step = 4;
    int steps = (int)(std::log((float)n) / std::log((float)step));
    auto factors = get_factors(n);
    if (factors.size() >= (size_t)steps) return factors;

    std::vector<int> ret;
    ret.reserve(steps);
    for (int i = 1; i <= n; i *= step) {
        int a = i;
        int b = i * step;
        bool found = false;
        for (int j : factors) {
            if (a <= j && j < b) {
                found = true;
                ret.push_back(j);
                break;
            }
        }
        if (!found) ret.push_back(i);
    }
    return ret;
}

bool block_ok(int size, int blk, int target_eff) {
    int size_padded = utils::rnd_up(size, blk);
    double eff = size / (double)size_padded;
    return eff * 100 >= target_eff;
}

// Divisibility restrictions for a convlution dimension.
struct div_info_t {
    // Iteration block must be divisible by this value.
    int iter_unit = 1;
    // (Iteration block) x (loop unroll) must be divisible by this value.
    int unroll_unit = 1;

    void set_iter_unit(int new_unit) {
        iter_unit = math::lcm(iter_unit, new_unit);
    }
    void set_unroll_unit(int new_unit) {
        unroll_unit = math::lcm(unroll_unit, new_unit);
    }

    bool is_iter_ok(int blk) const {
        if (iter_unit != 1 && blk % iter_unit != 0) return false;
        if (iter_unit != 1 && !math::is_pow2(blk)) return false;
        return true;
    }
};

// Specifies blocking restrictions for a convolution dimension.
struct tile_info_t {
    tile_info_t() = default;
    tile_info_t(const conv_dim_t &dim) : dim(dim) {}
    void add(tile_flags_t f) { flags = flags | f; }
    void remove(tile_flags_t f) { flags = flags & ~f; }
    void set_iter_unit(int unit) { div_info.set_iter_unit(unit); }
    void set_unroll_unit(int unit) { div_info.set_unroll_unit(unit); }
    void set_min_iter_block(int block, int pow2_block = 0) {
        min_iter_blk = block;
        if (pow2_block != 0) min_iter_pow2_blk = pow2_block;
    }

    std::vector<int> iter_blocks(int size) const {
        if (!any(flags & tile_flags_t::iter)) return {1};
        std::vector<int> ret;
        int lo = std::min(size, (int)min_iter_blk);
        int hi = max_iter_blk;
        uint32_t pow2_seen = 0;
        // Step 1. Check the divisors.
        for (int i = lo; i <= hi; i++) {
            if (!div_info.is_iter_ok(i)) continue;
            if (size % i == 0) {
                if (math::is_pow2(i)) pow2_seen |= (1U << math::ilog2q(i));
                ret.push_back(i);
            }
        }
        // Step 2. Add at least one power-of-two block.
        int pow2_count = 0;
        int min_pow2 = min_iter_pow2_blk;
        int max_pow2 = utils::rnd_down_pow2(max_iter_blk);
        for (int eff = 75; eff >= 0 && pow2_count == 0; eff--) {
            for (int i = min_pow2; i <= max_pow2; i *= 2) {
                if (!div_info.is_iter_ok(i)) continue;
                if (!block_ok(size, i, eff)) continue;
                if ((pow2_seen & (uint32_t)i) == 0) ret.push_back(i);
                pow2_count++;
            }
        }
        return ret;
    }

    std::vector<int> thread_group_blocks(int size) const {
        std::vector<int> ret;
        int bound = any(flags & tile_flags_t::thread_group)
                ? max_thread_group_blk
                : 1;
        for (int i = 1; i <= bound; i *= 2) {
            int size_padded = utils::rnd_up(size, i);
            double eff = (double)size / size_padded;
            if (eff >= 0.75) ret.push_back(i);
        }
        return ret;
    }

    std::vector<int> loop_blocks(int size, int iter_blk) const {
        if (!any(flags & tile_flags_t::loop)) return {1};
        if (any(flags & tile_flags_t::loop_span)) return {size};
        if (any(flags & tile_flags_t::loop_iter_unroll)) {
            int blk = math::lcm(div_info.unroll_unit, iter_blk);
            return {blk / iter_blk};
        }
        return get_loop_blocks(size);
    }

    conv_dim_t dim;
    tile_flags_t flags = tile_flags_t::undef;
    div_info_t div_info;

    int min_iter_blk = default_min_iter_blk;
    int min_iter_pow2_blk = default_min_iter_pow2_blk;
    int max_iter_blk = default_max_iter_blk;
    int max_thread_group_blk = default_max_thread_group_blk;

    static const int default_min_iter_blk = 6;
    static const int default_min_iter_pow2_blk = 8;
    static const int default_max_iter_blk = 64;
    static const int default_max_thread_group_blk = 16;
};

// Used for fused reduction dimensions with dpas only.
struct x2_tile_info_t {
    x2_tile_info_t() = default;
    x2_tile_info_t(const conv_dim_t &dim0, const conv_dim_t &dim1)
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
                auto factors = get_factors(ij);
                if (!block_ok(size0 * size1, ij, eff)) continue;
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

    conv_dim_t dim0;
    conv_dim_t dim1;
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
        tensor_kind_t tensor_kind, const conv_dim_t &d) {
    auto &prb = cfg.prb();
    if (!is_reduction_dim(d, prb.prop_kind(), prb.ab_swap_transpose)) return 1;
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

int get_layout_unit(const conv_config_t &cfg, const conv_dim_t &d) {
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

// Blocking scheme describing recipes to generate blockings.
class blocking_scheme_t {
public:
    blocking_scheme_t() = default;
    blocking_scheme_t(const std::string &s) {
        ir_assert(s[s.length() - 1] == ']');
        auto parts = ir_utils::split(s.substr(0, s.length() - 1), "],");
        for (auto &p : parts) {
            auto p_parts = ir_utils::split(p, ":");
            auto &key = p_parts[0];
            auto &vec = p_parts[1];
            ir_assert(vec[0] == '[');
            auto s_dims = ir_utils::split(vec.substr(1, vec.length() - 1), ",");
            for (auto &s : s_dims)
                set(key, s);
        }
    }

    tile_info_t &tile_info(const conv_dim_t &d) {
        auto it = tile_infos_.find(d.id());
        if (it != tile_infos_.end()) return it->second;
        auto &info = tile_infos_[d.id()];
        info = tile_info_t(d);
        return info;
    }

    const tile_info_t &tile_info(const conv_dim_t &d) const {
        return tile_infos_.at(d.id());
    }

    const std::vector<x2_tile_info_t> &x2_tile_infos() const {
        return x2_tile_infos_;
    }

    std::vector<conv_dim_t> dims() const {
        std::vector<conv_dim_t> ret;
        for (auto &d : all_conv_dims()) {
            if (loop_.has(d) || thread_group_.has(d) || iter_.has(d)) {
                ret.push_back(d);
            }
        }
        return ret;
    }

    void finalize(const conv_config_t &cfg) {
        finalize_iter_units(cfg);
        finalize_iter_min_blocks(cfg);
        finalize_fused_reduction(cfg);
        finalize_loop_dims(cfg);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "l:" << loop_;
        oss << " T:" << thread_group_;
        oss << " i:" << iter_;
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    void set(const std::string &s_tile, const std::string &_s_dim) {
        ir_assert(!_s_dim.empty());
        bool no_min_check = (_s_dim[0] == '#');
        auto s_dim = no_min_check ? _s_dim.substr(1) : _s_dim;
        auto d = conv_dim_t::from_name(s_dim);
        if (no_min_check) ir_assert(s_tile == "i");
        if (s_tile == "i") {
            add_iter_dim(d);
            if (no_min_check) tile_info(d).set_min_iter_block(1);
        } else if (s_tile == "T") {
            add_thread_group_dim(d);
        } else if (s_tile == "l") {
            add_loop_dim(d);
        } else if (s_tile == "ls") {
            add_loop_dim_with_span(d);
        } else if (s_tile == "li") {
            add_loop_dim_with_iter_unroll(d);
        } else {
            ir_error_not_expected() << s_tile;
        }
    }

    void add_loop_dim(const conv_dim_t &d) {
        loop_[d] = 1;
        auto &info = tile_info(d);
        info.add(tile_flags_t::loop);
    }

    void add_loop_dim_with_span(const conv_dim_t &d) {
        add_loop_dim(d);
        tile_info(d).add(tile_flags_t::loop_span);
    }

    void add_loop_dim_with_iter_unroll(const conv_dim_t &d) {
        add_loop_dim(d);
        tile_info(d).add(tile_flags_t::loop_iter_unroll);
    }

    void add_thread_group_dim(const conv_dim_t &d) {
        thread_group_[d] = 1;
        auto &info = tile_info(d);
        info.add(tile_flags_t::thread_group);
    }

    void add_iter_dim(const conv_dim_t &d) {
        iter_[d] = 1;
        auto &info = tile_info(d);
        info.add(tile_flags_t::iter);
    }

    void finalize_iter_units(const conv_config_t &cfg) {
        auto &prb = cfg.prb();
        bool is_dpas = cfg.is_dp_fma();
        int rdims = 0;
        for (auto d : iter_) {
            if (is_reduction_dim(d, prb.prop_kind(), prb.ab_swap_transpose))
                rdims++;
        }
        bool is_fused_reduction = (rdims > 1);
        for (auto d : iter_) {
            auto &info = tile_info(d);
            int unit = 1;
            if (is_vectorized_dim(d, prb)) unit = cfg.vec_size();
            if (is_reduction_dim(d, prb.prop_kind(), prb.ab_swap_transpose)) {
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
        auto &prb = cfg.prb();
        if (is_mad_x8_non_dw(cfg)) {
            // Reduce min block size for mad/x8 as it requires a lot of
            // additional space for x8 -> s16 reorder.
            int min_m_iter_block_hint = 2;
            for (auto d : iter_) {
                if (to_gemm(d, prb.prop_kind(), prb.ab_swap_transpose)
                        != gemm_dims::m)
                    continue;
                auto &info = tile_info(d);
                int blk = std::min(info.min_iter_blk, min_m_iter_block_hint);
                int pow2_blk = utils::rnd_up_pow2(blk);
                info.set_min_iter_block(blk, pow2_blk);
            }
        }
    }

    void finalize_fused_reduction(const conv_config_t &cfg) {
        if (!cfg.is_dp_fma()) return;
        int rdims = 0;
        conv_dim_t d0;
        conv_dim_t d1;
        for (auto d : iter_) {
            if (!is_reduction_dim(
                        d, cfg.prb().prop_kind(), cfg.prb().ab_swap_transpose))
                continue;
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

        int unit = 32 / cfg.prb().a_data_type_size;
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
                conv_dim_t dim;
                int size = 0;

                static loop_dim_t *find(
                        const conv_dim_t &dim, std::vector<loop_dim_t> &dims) {
                    for (auto &d : dims)
                        if (d.dim == dim) return &d;
                    return nullptr;
                }
            };
            auto shape = get_conv_shape(cfg, /*pad=*/true);
            std::vector<loop_dim_t> loop_dims;
            const int iter_dim_hint = 16;
            for (auto d : loop_) {
                if (any(tile_info(d).flags & tile_flags_t::loop_iter_unroll))
                    continue;
                loop_dim_t ld;
                ld.dim = d;
                ld.size = shape.at(d, 1);
                if (iter_.has(d))
                    ld.size = utils::div_up(ld.size, iter_dim_hint);
                loop_dims.push_back(ld);
            }
            std::sort(loop_dims.begin(), loop_dims.end(),
                    [&](const loop_dim_t &a, const loop_dim_t &b) {
                        return a.size > b.size;
                    });
            // For XeHPG and earlier hardware use only linear loops with SLM
            // pipelining to avoid overflowing icache. Prefetch pipeline can
            // handle nested loops without fully unrolling them.
            int max_loop_ndims = (cfg.hw() <= ngen::HW::XeHPG ? 1 : 2);
            for (int i = max_loop_ndims; i < (int)loop_dims.size(); i++)
                loop_dims[i].dim = conv_dim_t();

            for (auto d : loop_) {
                auto &info = tile_info(d);
                if (any(info.flags & tile_flags_t::loop_iter_unroll)) continue;
                if (!loop_dim_t::find(d, loop_dims))
                    info.remove(tile_flags_t::loop);
            }
        }
    }

    conv_tile_t loop_;
    conv_tile_t thread_group_;
    conv_tile_t iter_;
    std::unordered_map<int, tile_info_t> tile_infos_;
    std::vector<x2_tile_info_t> x2_tile_infos_;
};

int get_default_max_tg_size(const hw_config_t &hw_cfg, int regs, int simd) {
    const compute::gpu_arch_t arch = convert_ngen_arch_to_dnnl(hw_cfg.hw());
    const int max_eus_per_wg = compute::device_info_t::max_eus_per_wg(arch);
    const int threads_per_eu
            = compute::device_info_t::threads_per_eu(arch, regs > 128);
    // When threads_per_eu is reduced by cfg (large_grf_mode) wg_per_thread
    // is increased proportionally.
    const int wg_per_thr = simd * compute::device_info_t::threads_per_eu(arch)
            / threads_per_eu;

    // Optimal thread group size may differ from hardware thread count due
    // to simd_size used in computation.
    return std::min(max_eus_per_wg * utils::rnd_down_pow2(threads_per_eu),
            static_cast<int>(hw_cfg.max_wg_size() / wg_per_thr));
}

void get_level_tiles(
        int size, const tile_info_t &info, std::vector<level_tile_t> &ret) {
    ret.clear();
    auto iter_blocks = info.iter_blocks(size);
    for (int iter : iter_blocks) {
        int tg_size = utils::div_up(size, iter);
        auto tg_blocks = info.thread_group_blocks(tg_size);
        for (int tg : tg_blocks) {
            int loop_size = utils::div_up(size, tg * iter);
            auto loop_blocks = info.loop_blocks(loop_size, iter);
            for (int loop : loop_blocks) {
                level_tile_t t;
                if (any(info.flags & tile_flags_t::loop))
                    t[levels::loop] = loop;
                if (any(info.flags & tile_flags_t::thread_group))
                    t[levels::thread_group] = tg;
                if (any(info.flags & tile_flags_t::iter))
                    t[levels::iter] = iter;
                ret.push_back(t);
            }
        }
    }
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

int inner_block(const conv_config_t &cfg, tensor_kind_t tensor_kind,
        const conv_dim_t &dim) {
    int dim_idx = tensor_conv_dim_index(dim, tensor_kind);
    ir_assert(dim_idx != -1);
    auto &layout = compute_layout(cfg, tensor_kind);
    return layout.inner_block(
            dim_idx, /*skip_outer=*/true, /*inner_only=*/false);
}

int inner_block(const conv_config_t &cfg, const conv_dim_t &dim) {
    int ret = 0;
    for (auto t : input_tensors(cfg)) {
        if (tensor_conv_dim_index(dim, t) == -1) continue;
        int blk = inner_block(cfg, t, dim);
        ret = (ret == 0 ? blk : math::gcd(ret, blk));
    }
    return ret == 0 ? 1 : ret;
}

dim_t inner_stride(const conv_config_t &cfg, tensor_kind_t tensor_kind,
        const conv_dim_t &dim) {
    int dim_idx = tensor_conv_dim_index(dim, tensor_kind);
    ir_assert(dim_idx != -1);
    auto &layout = compute_layout(cfg, tensor_kind);
    for (auto &b : layout.blocks()) {
        if (b.dim_idx == dim_idx) return (dim_t)b.stride;
    }
    return 0;
}

bool is_inner_non_blocked(const conv_config_t &cfg, const conv_dim_t &dim) {
    for (auto t : input_tensors(cfg)) {
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
        const conv_config_t &cfg, const conv_params_t &params) {
    auto &prb = cfg.prb();
    auto tg = to_gemm(params.blocking().thread_group(), prb.prop_kind(),
            prb.ab_swap_transpose);
    auto iter = to_gemm(
            params.blocking().iter(), prb.prop_kind(), prb.ab_swap_transpose);
    int b_tg = tg.at(gemm_dims::b, 1);
    int m_tg = tg.at(gemm_dims::m, 1);
    int n_tg = tg.at(gemm_dims::n, 1);
    int k_tg = tg.at(gemm_dims::k, 1);
    int b_iter = iter.at(gemm_dims::b, 1);
    int m_iter = iter.at(gemm_dims::m, 1);
    int n_iter = iter.at(gemm_dims::n, 1);
    int k_iter = iter.at(gemm_dims::k, 1);
    return slm_usage_bytes(
            cfg, b_tg, m_tg, n_tg, k_tg, b_iter, m_iter, n_iter, k_iter);
}

class blocking_checker_t {
public:
    blocking_checker_t(const conv_config_t &cfg) : cfg_(cfg) {
        init_checks();
        padded_shape_ = get_conv_shape(cfg, /*pad=*/true);
        padded_gemm_shape_ = to_gemm(padded_shape_, cfg.prb().prop_kind(),
                cfg.prb().ab_swap_transpose);
        max_tg_size_ = get_default_max_tg_size(
                cfg.hw_cfg(), cfg.exec_cfg().regs(), cfg.simd());
    }

    bool relax_checks() {
        for (int i = 0; i < (int)check_kind_t::_max; i++) {
            auto check = static_cast<check_kind_t>(i);
            if (!is_optional(check)) continue;
            if (!is_enabled(check)) continue;
            set_check(check, false);
            return true;
        }
        return false;
    }

    bool is_ok(const blocking_t &blk) const {
        context_t ctx(blk, cfg_);
        if (!check_tg_size_ok(ctx)) return false;
        if (!check_dpas_ok(ctx)) return false;
        if (!check_grf_usage_ok(ctx)) return false;
        if (!check_slm_usage_ok(ctx)) return false;
        if (!check_bwd_d_optimize_ok(ctx)) return false;
        if (!check_layouts_ok(ctx)) return false;
        if (!check_k_slicing_utilization_ok(ctx)) return false;
        if (!limit_m_iter_ok(ctx)) return false;
        if (!limit_n_iter_ok(ctx)) return false;
        if (!limit_k_iter_ok(ctx)) return false;
        return true;
    }

private:
    struct context_t {
        context_t(const blocking_t &blk, const conv_config_t &cfg) : blk(blk) {
            auto &prb = cfg.prb();
            auto gemm_iter = to_gemm(
                    blk.iter(), prb.prop_kind(), prb.ab_swap_transpose);
            auto gemm_loop = to_gemm(
                    blk.loop(), prb.prop_kind(), prb.ab_swap_transpose);
            auto gemm_tg = to_gemm(
                    blk.thread_group(), prb.prop_kind(), prb.ab_swap_transpose);
            b_iter = gemm_iter.at(gemm_dims::b, 1);
            m_iter = gemm_iter.at(gemm_dims::m, 1);
            n_iter = gemm_iter.at(gemm_dims::n, 1);
            k_iter = gemm_iter.at(gemm_dims::k, 1);
            k_loop = gemm_loop.at(gemm_dims::k, 1);
            b_tg = gemm_tg.at(gemm_dims::b, 1);
            m_tg = gemm_tg.at(gemm_dims::m, 1);
            n_tg = gemm_tg.at(gemm_dims::n, 1);
            k_tg = gemm_tg.at(gemm_dims::k, 1);
            dpas_2x_depth = get_dpas_2x_depth(blk, cfg);
        }

        bool get_dpas_2x_depth(
                const blocking_t &blk, const conv_config_t &cfg) const {
            if (!cfg.is_dp_fma() || cfg.regs() <= 128) return false;

            // Use 2x reduction when the reduction dimension is dense to avoid
            // partial cache line loads.
            for (auto d : blk.iter()) {
                if (to_gemm(d, cfg.prb().prop_kind(),
                            cfg.prb().ab_swap_transpose)
                        == gemm_dims::k) {
                    if (is_inner_non_blocked(cfg, d)) return true;
                }
            }

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
        check_tg_size,
        check_dpas,
        check_grf_usage,
        check_slm_usage,
        check_bwd_d_optimize,
        check_layouts,
        check_k_slicing_utilization,
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

    void init_checks() {
        ir_assert((int)check_kind_t::_max < 64);

        set_check(optional_check_mask_, check_kind_t::limit_k_iter);
        set_check(optional_check_mask_,
                check_kind_t::check_k_slicing_utilization);
        set_check(check_kind_t::check_tg_size);
        set_check(check_kind_t::check_dpas);
        set_check(check_kind_t::check_grf_usage);
        set_check(check_kind_t::check_slm_usage);
        set_check(check_kind_t::check_bwd_d_optimize);
        set_check(check_kind_t::check_layouts);
        set_check(check_kind_t::check_k_slicing_utilization);
        set_check(check_kind_t::limit_m_iter);
        set_check(check_kind_t::limit_n_iter);
        set_check(check_kind_t::limit_k_iter);
    }

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

    bool check_tg_size_ok(const context_t &ctx) const {
        if (!is_enabled(check_kind_t::check_tg_size)) return true;

        auto &tg = ctx.blk.thread_group();
        int tg_size = 1;
        int max_tg = 1;
        for (auto d : tg) {
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
                convert_ngen_arch_to_dnnl(cfg_.hw()), tg_size,
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
                int iw_iter = ctx.blk.iter().at(conv_dims::iw, 1);
                int iw_tg = ctx.blk.thread_group().at(conv_dims::iw, 1);
                if (iw_iter != 1 || iw_tg != 1) return false;
                return true;
            }
            case bwd_d_optimize_kind_t::skip_strided_dh: return true;
            case bwd_d_optimize_kind_t::skip_strided_dhw: {
                int iw_iter = ctx.blk.iter().at(conv_dims::iw, 1);
                if (iw_iter > 1) return false;
                int iw_tg = ctx.blk.thread_group().at(conv_dims::iw, 1);
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
            const layout_t &layout, const conv_dim_t &d,
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
        for (int i = 0; i < conv_dim_t::max_id(); i++) {
            auto d = conv_dim_t::from_id(i);
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

        int b = padded_gemm_shape_.at(gemm_dims::b, 1);
        int m = padded_gemm_shape_.at(gemm_dims::m, 1);
        int n = padded_gemm_shape_.at(gemm_dims::n, 1);
        int k = padded_gemm_shape_.at(gemm_dims::k, 1);

        int64_t nthr = 1;
        nthr *= utils::div_up(b, ctx.b_iter);
        nthr *= utils::div_up(m, ctx.m_iter);
        nthr *= utils::div_up(n, ctx.n_iter);
        nthr *= utils::div_up(k, ctx.k_iter * ctx.k_loop);
        if (nthr < 16 && ctx.k_loop > 512) return false;

        return true;
    }

    int hint_min_m_iter() const {
        if (is_mad_x8_non_dw(cfg_)) return min_mad_x8_non_dw_m_iter_;
        return min_m_iter_;
    }

    int min_m_iter(const context_t &ctx) const {
        auto &prb = cfg_.prb();
        int max_blk = 1;
        for (auto d : ctx.blk.iter()) {
            if (to_gemm(d, prb.prop_kind(), prb.ab_swap_transpose)
                    == gemm_dims::m) {
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
    uint64_t check_mask_ = 0;
    uint64_t optional_check_mask_ = 0;
    conv_tile_t padded_shape_;
    gemm_tile_t padded_gemm_shape_;
    int max_tg_size_ = 0;
};

// Returns the ratio of all operations (with padding) to "useful"
// operations.
float get_efficiency(const blocking_t &blk, const conv_config_t &cfg) {
    float ret = 1;
    auto shape = get_conv_shape(cfg, /*pad=*/true);
    for (auto d : shape) {
        int loop = blk.loop().at(d, 1);
        int tg = blk.thread_group().at(d, 1);
        int iter = blk.iter().at(d, 1);
        int size = shape[d];
        int size_padded = utils::rnd_up(size, loop * tg * iter);
        if (size_padded == size) continue;
        ret *= (float)size / size_padded;
    }
    return ret;
}

void set(blocking_t &blk, const conv_dim_t &dim, const level_tile_t &tile) {
    if (tile.has(levels::loop)) blk.set_loop(dim, tile[levels::loop]);
    if (tile.has(levels::thread_group))
        blk.set_thread_group(dim, tile[levels::thread_group]);
    if (tile.has(levels::iter)) blk.set_iter(dim, tile[levels::iter]);
}

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
blocking_scheme_t fwd_T_wo_I_noi("ls:[ic,kd,kh,kw],T:[oc,ow],i:[mb,oc,ic]");
blocking_scheme_t fwd_T_no_I_noi("ls:[ic,kd,kh,kw],T:[oc,mb],i:[mb,oc,ic]");
blocking_scheme_t fwd_T_wn_I_wnoi("ls:[ic,kd,kh,kw],T:[ow,mb],i:[ow,mb,oc,ic]");
blocking_scheme_t fwd_T_i_I_noi("ls:[ic,kd,kh,kw],T:[ic],i:[mb,oc,ic]");
blocking_scheme_t fwd_T_iw_I_wnoi("ls:[ic,kd,kh,kw],T:[ic,ow],i:[ow,mb,oc,ic]");
blocking_scheme_t fwd_T_wo_I_woi("ls:[ic,kd,kh,kw],T:[oc,ow],i:[ow,oc,ic]");
blocking_scheme_t fwd_T_i_I_woi("ls:[ic,kd,kh,kw],T:[ic],i:[ow,oc,ic]");
blocking_scheme_t fwd_T_wo_I_woki("ls:[ic,kd,kh,kw],T:[oc,ow],i:[ow,oc,kw,ic]");
blocking_scheme_t fwd_T_w_I_woki("ls:[ic,kd,kh,kw],T:[ow],i:[ow,oc,kw,ic]");
blocking_scheme_t fwd_T_w_I_noki("ls:[ic,kd,kh,kw],T:[ow],i:[mb,ow,oc,kw,ic]");
blocking_scheme_t fwd_T_wo_I_noki("ls:[ic,kd,kh,kw],T:[oc,ow],i:[mb,oc,kw,ic]");
blocking_scheme_t fwd_dw_T_w_I_wgk("ls:[kd,kh,kw],T:[ow],i:[ow,g,#kw]");
blocking_scheme_t fwd_dw_T_w_I_ngk("ls:[kd,kh,kw],T:[ow],i:[mb,g,#kw]");
blocking_scheme_t bwd_d_T_wi_I_nio("ls:[oc,kd,kh,kw],T:[ic,iw],i:[mb,ic,oc]");
blocking_scheme_t bwd_d_T_ni_I_nio("ls:[oc,kd,kh,kw],T:[ic,mb],i:[mb,ic,oc]");
blocking_scheme_t bwd_d_T_o_I_nio("ls:[oc,kd,kh,kw],T:[oc],i:[mb,ic,oc]");
blocking_scheme_t bwd_d_T_w_I_on("ls:[oc,kd,kh,kw],T:[iw],i:[oc,mb]");
blocking_scheme_t bwd_d_T_wi_I_wio("ls:[oc,kd,kh,kw],T:[ic,iw],i:[iw,ic,oc]");
blocking_scheme_t bwd_d_T_o_I_wio("ls:[oc,kd,kh,kw],T:[oc],i:[iw,ic,oc]");
blocking_scheme_t bwd_d_dw_T_w_I_wgk("ls:[kd,kh,kw],T:[iw],i:[iw,g,kw]");
blocking_scheme_t bwd_d_dw_T_w_I_ngk("ls:[kd,kh,kw],T:[iw],i:[mb,g,kw]");
blocking_scheme_t bwd_w_T_io_I_ion("l:[oh,ow],li:[mb],T:[oc,ic],i:[ic,oc,mb]");
blocking_scheme_t bwd_w_T_io_I_kon("l:[oh,ow],li:[mb],T:[oc,ic],i:[kw,oc,mb]");
blocking_scheme_t bwd_w_T_io_I_ikon("l:[oh,ow],li:[mb],T:[oc,ic],i:[ic,kw,oc,mb]");
blocking_scheme_t bwd_w_dw_I_gw("l:[mb,oh,ow],i:[g,ow]");
blocking_scheme_t bwd_w_dw_I_gn("l:[mb,oh,ow],i:[g,mb]");
blocking_scheme_t bwd_w_T_io_I_iow("l:[mb,oh,ow],T:[oc,ic],i:[ic,oc,ow]");
blocking_scheme_t bwd_w_T_io_I_ikow("l:[mb,oh,ow],T:[oc,ic],i:[ic,kw,oc,ow]");
} // namespace conv_schemes
// clang-format on

double get_iter_dim_score(
        const conv_dim_t &dim, const conv_config_t &cfg, int dim_size) {
    auto &prb = cfg.prb();
    if (utils::one_of(dim, conv_dims::ow, conv_dims::iw)) {
        if (prb.ksp > 1 || dim_size % 16 != 0) return 16 - 1;
        return dim_size;
    } else if (dim == conv_dims::mb) {
        return dim_size;
    } else {
        ir_error_not_expected() << "Unknown dimension: " << dim;
    }
    return 0;
}

conv_dim_t select_non_blocked_iter_dim(
        const conv_config_t &cfg, const std::vector<conv_dim_t> &dims) {
    const auto shape = get_conv_shape(cfg, /*pad=*/false);
    std::vector<double> scores;
    for (auto d : dims)
        scores.push_back(get_iter_dim_score(d, cfg, shape[d]));
    auto max_it = std::max_element(scores.begin(), scores.end());
    return dims[max_it - scores.begin()];
}

conv_dim_t select_iter_dim(
        const conv_config_t &cfg, const std::vector<conv_dim_t> &_dims) {
    bool is_bwd_d_w_opt = utils::one_of(cfg.bwd_d_optimize_kind(),
            bwd_d_optimize_kind_t::skip_strided_dhw,
            bwd_d_optimize_kind_t::skip_out_of_bound_w);
    std::vector<conv_dim_t> dims;
    for (auto d : _dims) {
        if (is_bwd_d_w_opt && d == conv_dims::iw) continue;
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

std::vector<blocking_scheme_t> get_blocking_schemes_fwd_dw(
        const conv_config_t &cfg) {
    std::vector<blocking_scheme_t> ret;
    auto m_iter_dim = select_iter_dim(cfg, {conv_dims::mb, conv_dims::ow});
    if (m_iter_dim == conv_dims::mb) {
        ret.push_back(conv_schemes::fwd_dw_T_w_I_ngk);
    } else {
        ret.push_back(conv_schemes::fwd_dw_T_w_I_wgk);
    }
    return ret;
}

std::vector<blocking_scheme_t> get_blocking_schemes_bwd_d_dw(
        const conv_config_t &cfg) {
    std::vector<blocking_scheme_t> ret;
    auto m_iter_dim = select_iter_dim(cfg, {conv_dims::mb, conv_dims::iw});
    if (m_iter_dim == conv_dims::mb) {
        ret.push_back(conv_schemes::bwd_d_dw_T_w_I_ngk);
    } else {
        ret.push_back(conv_schemes::bwd_d_dw_T_w_I_wgk);
    }
    return ret;
}

std::vector<blocking_scheme_t> get_blocking_schemes_bwd_w_dw(
        const conv_config_t &cfg) {
    std::vector<blocking_scheme_t> ret;
    auto k_iter_dim = select_iter_dim(cfg, {conv_dims::mb, conv_dims::ow});
    if (k_iter_dim == conv_dims::mb) {
        ret.push_back(conv_schemes::bwd_w_dw_I_gn);
    } else {
        ret.push_back(conv_schemes::bwd_w_dw_I_gw);
    }
    return ret;
}

std::vector<blocking_scheme_t> get_blocking_schemes_fwd(
        const conv_config_t &cfg) {
    std::vector<blocking_scheme_t> ret;
    auto m_iter_dim = cfg.prb().ab_swap_transpose
            ? conv_dims::oc
            : select_iter_dim(cfg, {conv_dims::mb, conv_dims::ow});
    if (m_iter_dim == conv_dims::mb) {
        ret.push_back(conv_schemes::fwd_T_wo_I_noi);
        ret.push_back(conv_schemes::fwd_T_no_I_noi);
        if (cfg.hw() >= ngen::HW::XeLP)
            ret.push_back(conv_schemes::fwd_T_i_I_noi);
    } else if (m_iter_dim == conv_dims::oc) {
        ret.push_back(conv_schemes::fwd_T_wn_I_wnoi);
        if (cfg.hw() >= ngen::HW::XeLP) {
            ret.push_back(conv_schemes::fwd_T_i_I_noi);
            ret.push_back(conv_schemes::fwd_T_iw_I_wnoi);
        }
    } else {
        ret.push_back(conv_schemes::fwd_T_wo_I_woi);
        if (cfg.hw() >= ngen::HW::XeLP)
            ret.push_back(conv_schemes::fwd_T_i_I_woi);
    }
    if (is_small_ic(cfg.prb()) && cfg.prb().kw > 1) {
        if (m_iter_dim == conv_dims::mb) {
            ret.push_back(conv_schemes::fwd_T_wo_I_noki);
        } else if (m_iter_dim == conv_dims::oc) {
            ret.push_back(conv_schemes::fwd_T_w_I_woki);
            ret.push_back(conv_schemes::fwd_T_w_I_noki);
        } else {
            ret.push_back(conv_schemes::fwd_T_wo_I_woki);
        }
    }
    return ret;
}

std::vector<blocking_scheme_t> get_blocking_schemes_bwd_d(
        const conv_config_t &cfg) {
    std::vector<blocking_scheme_t> ret;
    auto m_iter_dim = cfg.prb().ab_swap_transpose
            ? conv_dims::ic
            : select_iter_dim(cfg, {conv_dims::mb, conv_dims::iw});
    if (m_iter_dim == conv_dims::mb) {
        ret.push_back(conv_schemes::bwd_d_T_ni_I_nio);
        ret.push_back(conv_schemes::bwd_d_T_wi_I_nio);
        if (cfg.hw() >= ngen::HW::XeLP)
            ret.push_back(conv_schemes::bwd_d_T_o_I_nio);
    } else {
        ret.push_back(conv_schemes::bwd_d_T_wi_I_wio);
        if (cfg.hw() >= ngen::HW::XeLP)
            ret.push_back(conv_schemes::bwd_d_T_o_I_wio);
        if (m_iter_dim == conv_dims::ic)
            ret.push_back(conv_schemes::bwd_d_T_w_I_on);
    }
    return ret;
}

std::vector<blocking_scheme_t> get_blocking_schemes_bwd_w(
        const conv_config_t &cfg) {
    std::vector<blocking_scheme_t> ret;
    auto k_iter_dim = select_iter_dim(cfg, {conv_dims::mb, conv_dims::ow});
    if (k_iter_dim == conv_dims::mb) {
        ret.push_back(conv_schemes::bwd_w_T_io_I_ion);
    } else {
        ret.push_back(conv_schemes::bwd_w_T_io_I_iow);
    }
    if (is_small_ic(cfg.prb())) {
        if (k_iter_dim == conv_dims::mb) {
            ret.push_back(conv_schemes::bwd_w_T_io_I_kon);
            ret.push_back(conv_schemes::bwd_w_T_io_I_ikon);
        } else {
            ret.push_back(conv_schemes::bwd_w_T_io_I_ikow);
        }
    }
    return ret;
}

std::vector<blocking_scheme_t> get_blocking_schemes_dw_impl(
        const conv_config_t &cfg) {
    auto &prb = cfg.prb();
    if (prb.is_fwd) return get_blocking_schemes_fwd_dw(cfg);
    if (prb.is_bwd_d) return get_blocking_schemes_bwd_d_dw(cfg);
    if (prb.is_bwd_w) return get_blocking_schemes_bwd_w_dw(cfg);
    ir_error_not_expected();
    return std::vector<blocking_scheme_t>();
}

std::vector<blocking_scheme_t> get_blocking_schemes_impl(
        const conv_config_t &cfg) {
    auto &prb = cfg.prb();
    if (prb.is_dw) return get_blocking_schemes_dw_impl(cfg);
    if (prb.is_fwd) return get_blocking_schemes_fwd(cfg);
    if (prb.is_bwd_d) return get_blocking_schemes_bwd_d(cfg);
    if (prb.is_bwd_w) return get_blocking_schemes_bwd_w(cfg);
    ir_error_not_expected();
    return std::vector<blocking_scheme_t>();
}

std::vector<blocking_scheme_t> get_blocking_schemes(const conv_config_t &cfg) {
    auto ret = get_blocking_schemes_impl(cfg);
    for (auto &s : ret)
        s.finalize(cfg);
    return ret;
}

} // namespace

int grf_usage_bytes(const conv_config_t &cfg, const conv_params_t &params) {
    auto &prb = cfg.prb();
    auto iter = to_gemm(
            params.blocking().iter(), prb.prop_kind(), prb.ab_swap_transpose);
    int b_iter = iter.at(gemm_dims::b, 1);
    int m_iter = iter.at(gemm_dims::m, 1);
    int n_iter = iter.at(gemm_dims::n, 1);
    int k_iter = iter.at(gemm_dims::k, 1);
    int abc_size = grf_usage_bytes(cfg.fma_kind(), b_iter, m_iter, n_iter,
            k_iter, prb.a_data_type_size, prb.b_data_type_size,
            prb.acc_data_type_size);
    return abc_size;
}

enum class tiler_mode_t {
    undef,
    env_config,
    env_tiler,
    lookup,
    model,
    tune,
    default_mode = lookup
};

struct conv_tiler_params_t {
    tiler_mode_t mode = tiler_mode_t::default_mode;
    bool do_list = false;
    int tune_iters = 0;
    int env_params_idx = -1;
};

conv_tiler_params_t &tiler_params() {
    static conv_tiler_params_t params = []() {
        conv_tiler_params_t ret;
        auto s_opts = gpu_utils::dev_getenv("tiler", std::string());
        if (s_opts.empty()) return ret;
        auto opts = ir_utils::split(s_opts, ",");
        for (auto &opt : opts) {
            if (opt.empty()) continue;
            if (opt == "list") {
                ret.do_list = true;
                continue;
            }
            if (opt == "lookup") {
                ret.mode = tiler_mode_t::lookup;
                continue;
            }
            if (opt == "model") {
                ret.mode = tiler_mode_t::model;
                continue;
            }
            if (opt == "tune") {
                ret.mode = tiler_mode_t::tune;
                continue;
            }
            auto sub_opts = ir_utils::split(opt, ":");
            ir_assert((int)sub_opts.size() == 2);
            auto &key = sub_opts[0];
            auto &value = sub_opts[1];
            if (key == "tune_iters") {
                ret.tune_iters = std::stoi(value);
            } else if (key == "params") {
                ret.mode = tiler_mode_t::env_tiler;
                ret.env_params_idx = std::stoi(value);
            } else {
                ir_error_not_expected();
            }
        }
        bool do_tune = (ret.mode == tiler_mode_t::tune);
        ir_assert(do_tune == (ret.tune_iters != 0));
        return ret;
    }();
    return params;
}

std::string to_string(tiler_mode_t mode) {
    switch (mode) {
#define CASE(name) \
    case tiler_mode_t::name: return #name
        CASE(undef);
        CASE(env_config);
        CASE(env_tiler);
        CASE(lookup);
        CASE(model);
        CASE(tune);
#undef CASE
    }
    ir_error_not_expected();
    return "(unknown)";
}

class level_tile_set_t {
public:
    level_tile_set_t(
            const blocking_scheme_t &scheme, const conv_tile_t &padded_shape) {
        dims_ = scheme.dims();
        int ndims = (int)dims_.size();
        tiles_.resize(ndims);
        deps_.resize(ndims, -1);

        auto to_idx = [&](const conv_dim_t &d) {
            for (int i = 0; i < ndims; i++)
                if (dims_[i] == d) return i;
            ir_error_not_expected();
            return -1;
        };

        std::vector<bool> seen(ndims);
        for (auto &info : scheme.x2_tile_infos()) {
            int idx0 = to_idx(info.dim0);
            int idx1 = to_idx(info.dim1);
            get_level_tiles(padded_shape[info.dim0], padded_shape[info.dim1],
                    info, tiles_[idx0], tiles_[idx1]);
            ir_assert(!seen[idx0] && !seen[idx1]);
            seen[idx0] = seen[idx1] = true;
            deps_[std::max(idx0, idx1)] = std::min(idx0, idx1);
        }
        for (int i = 0; i < ndims; i++) {
            if (seen[i]) continue;
            auto &d = dims_[i];
            get_level_tiles(padded_shape[d], scheme.tile_info(d), tiles_[i]);
        }
    }

    int count() const {
        int ret = 1;
        int ntiles = (int)tiles_.size();
        for (int i = 0; i < ntiles; i++) {
            if (deps_[i] != -1) continue;
            ret *= (int)tiles_[i].size();
        }
        return ret;
    }

    std::vector<blocking_t> product(int simd) const {
        std::vector<blocking_t> ret;
        blocking_t blk;
        blk.set_simd(simd);
        std::vector<int> cur_idxs(dims_.size());
        product_impl(0, cur_idxs, blk, ret);
        return ret;
    }

    std::vector<blocking_t> sample(int target,
            const blocking_checker_t &blocking_checker, int simd,
            int tries_mult_bound = 5) const {
        std::vector<blocking_t> ret;
        ir_utils::fast_random_t r;
        int max_tries = target * tries_mult_bound;
        for (int tries = 0; tries < max_tries; tries++) {
            auto try_tiles = sample(r);
            blocking_t blk;
            blk.set_simd(simd);
            for (int i = 0; i < (int)dims_.size(); i++) {
                set(blk, dims_[i], try_tiles[i]);
            }
            if (!blocking_checker.is_ok(blk)) continue;
            ret.push_back(blk);
            if ((int)ret.size() >= target) break;
        }
        return ret;
    }

private:
    void product_impl(int idx, std::vector<int> &cur_idxs, blocking_t &blk,
            std::vector<blocking_t> &ret) const {
        if (idx == (int)dims_.size()) {
            ret.push_back(blk);
            return;
        }
        auto &v = tiles_[idx];
        if (deps_[idx] != -1) {
            cur_idxs[idx] = cur_idxs[deps_[idx]];
            set(blk, dims_[idx], v[cur_idxs[idx]]);
            product_impl(idx + 1, cur_idxs, blk, ret);
            return;
        }
        for (int i = 0; i < (int)v.size(); i++) {
            cur_idxs[idx] = i;
            set(blk, dims_[idx], v[i]);
            product_impl(idx + 1, cur_idxs, blk, ret);
            blk.unset(dims_[idx]);
        }
    }

    std::vector<level_tile_t> sample(ir_utils::fast_random_t &r) const {
        int ndims = (int)dims_.size();
        std::vector<int> cur_idxs(ndims);
        std::vector<level_tile_t> ret;
        ret.reserve(ndims);
        for (int i = 0; i < ndims; i++) {
            cur_idxs[i] = (deps_[i] == -1) ? r.rand_index(tiles_[i])
                                           : cur_idxs[deps_[i]];
            ret.push_back(tiles_[i][cur_idxs[i]]);
        }
        return ret;
    }

    std::vector<std::vector<level_tile_t>> tiles_;
    std::vector<int> deps_;
    std::vector<conv_dim_t> dims_;
};

class blocking_generator_t {
public:
    blocking_generator_t(int target_blockings, const conv_tile_t &padded_shape,
            const std::vector<blocking_scheme_t> &schemes, int *total_blockings)
        : schemes_(schemes) {
        UNUSED(target_blockings);
        for (auto &s : schemes_)
            level_tile_sets_.emplace_back(s, padded_shape);
    }

    void generate(const conv_config_t &cfg) {
        int nschemes = (int)schemes_.size();
        for (int s_idx = 0; s_idx < nschemes; s_idx++) {
            add(s_idx, cfg);
        }
    }

    std::vector<blocking_t> blockings(const conv_config_t &cfg) const {
        return std::vector<blocking_t>(blockings_.begin(), blockings_.end());
    }

private:
    void add(int idx, const conv_config_t &cfg) { generate_all(idx, cfg); }

    void generate_all(int idx, const conv_config_t &cfg) {
        blocking_checker_t blocking_checker(cfg);
        auto s_dims = schemes_[idx].dims();
        auto s_blockings
                = level_tile_sets_[idx].product(cfg.exec_cfg().vec_size());
        for (;;) {
            bool added = false;
            for (auto &b : s_blockings) {
                if (!blocking_checker.is_ok(b)) continue;
                added = true;
                blockings_.insert(b);
            }
            if (!added && blocking_checker.relax_checks()) continue;
            break;
        }
    }

    // TODO: Remove.
    void generate_sample(int idx, const conv_config_t &cfg) {
        ir_assert(false);
        blocking_checker_t blocking_checker(cfg);
        std::vector<int> target_sizes;
        auto s_blockings = level_tile_sets_[idx].sample(
                target_sizes[idx], blocking_checker, cfg.vec_size());
        blockings_.insert(s_blockings.begin(), s_blockings.end());
    }

    const std::vector<blocking_scheme_t> &schemes_;
    std::vector<level_tile_set_t> level_tile_sets_;

    std::unordered_set<blocking_t, blocking_hash_t> blockings_;
};

class params_generator_t {
public:
    params_generator_t() = default;

    params_generator_t(const conv_params_t &params) {
        params_vec_.push_back(params);
        assign_ids();
    }

    params_generator_t(const conv_config_t &cfg, const conv_params_t &params) {
        params_vec_ = generate_params_vec(cfg);
        params_vec_.insert(params_vec_.begin(), params);
        assign_ids();
    }

    params_generator_t(const conv_config_t &cfg, int idx = -1) {
        params_vec_ = generate_params_vec(cfg);
        if (idx != -1) {
            ir_assert(idx >= 0 && idx < configs());
            params_vec_ = std::vector<conv_params_t>({params_vec_[idx]});
        }
        assign_ids();
    }

    const std::vector<conv_params_t> &params_vec() const { return params_vec_; }

    bool is_empty() const { return params_vec_.empty(); }

    bool can_move_next() const { return cur_idx_ + 1 < configs(); }

    void move_next() {
        ir_assert(can_move_next());
        cur_idx_++;
    }

    int cur_index() const { return cur_idx_; }

    const conv_params_t &cur_params() const { return at(cur_idx_); }

    const conv_params_t &at(int idx) const {
        ir_assert(idx >= 0 && idx < configs());
        return params_vec_[idx];
    }

    void set_params(conv_config_t &cfg) {
        auto &params = params_vec_[cur_idx_];
        ir_trace() << "set params #" << cur_idx_ << ": " << params << std::endl;
        params.apply_to(cfg);
    }

    int configs() const { return (int)params_vec_.size(); }

    template <typename KeyFuncT>
    void sort(int beg, int end, const KeyFuncT &key_func) {
        ir_assert(beg >= 0 && beg < configs());
        ir_assert(end >= beg && end <= configs());
        std::sort(params_vec_.begin() + beg, params_vec_.begin() + end,
                [&](const conv_params_t &a, const conv_params_t &b) {
                    return key_func(a) < key_func(b);
                });
    }

    template <typename PredicateFuncT>
    void remove_if(const PredicateFuncT &func) {
        ir_assert(cur_idx_ == -1);
        params_vec_.erase(
                std::remove_if(params_vec_.begin(), params_vec_.end(), func),
                params_vec_.end());
    }

    void shuffle(size_t seed) {
        std::minstd_rand g(static_cast<uint32_t>(seed));
        std::shuffle(params_vec_.begin(), params_vec_.end(), g);
    }

    void print_all() const {
        using namespace ir_utils;
        std::vector<std::string> headers = {};
        table_t table("List of configs", headers);
        for (int i = 0; i < configs(); i++) {
            auto &params = params_vec_[i];
            ir_trace() << "params #" << i << ": " << params << std::endl;
        }
    }

private:
    void assign_ids() {
        for (int i = 0; i < configs(); i++)
            params_vec_[i].set_id(i);
    }

    std::vector<conv_params_t> generate_params_vec(
            const conv_config_t &cfg, int *total_blockings = nullptr) const {
        auto schemes = get_blocking_schemes(cfg);
        auto padded_shape = get_conv_shape(cfg, /*pad=*/true);
        blocking_generator_t bg(tiler_params().tune_iters, padded_shape,
                schemes, total_blockings);
        bg.generate(cfg);
        auto blockings = bg.blockings(cfg);
        return std::vector<conv_params_t>(blockings.begin(), blockings.end());
    }

    std::vector<conv_params_t> params_vec_;
    int cur_idx_ = -1;
};

void sort_by_model_scores(params_generator_t &params_gen,
        const conv_config_t &cfg, tiler_mode_t mode) {
    std::unordered_map<int, float> eff_scores;
    for (int i = 0; i < params_gen.configs(); i++) {
        auto &p = params_gen.at(i);
        float eff = get_efficiency(p.blocking(), cfg);
        float score = model::get_score(cfg, p);
        eff_scores.emplace(p.id(), score * eff);
    }
    if (mode == tiler_mode_t::lookup) {
        // Give the lookup entry the highest score.
        eff_scores[params_gen.at(0).id()] = 1.0f;
    }
    params_gen.sort(0, params_gen.configs(),
            [&](const conv_params_t &p) { return -eff_scores.at(p.id()); });
#ifdef DNNL_DEV_MODE
    using namespace ir_utils;
    std::vector<std::string> headers
            = {"Config", "Score", "Eff", "Regs", "SLM size"};
    table_t table("List of configs", headers);
    for (auto &p : params_gen.params_vec()) {
        float score = model::get_score(cfg, p);
        float eff = get_efficiency(p.blocking(), cfg);
        int regs = utils::div_up(grf_usage_bytes(cfg, p), cfg.grf_size());
        int slm_size = slm_usage_bytes_for_params(cfg, p);
        table << p.str() << (int)(score * 1000) / 1000.0 << eff << regs
              << slm_size << std::endl;
    }
    ir_trace() << table.str() << std::endl;
#endif
    MAYBE_UNUSED(&slm_usage_bytes_for_params);
}

struct indexed_dim_t {
    indexed_dim_t() = default;
    indexed_dim_t(const gemm_dim_t &dim) : dim_(dim) {}
    bool is_empty() const { return values_.empty(); }
    const gemm_dim_t &dim() const { return dim_; }

    void add(int value) { values_.emplace(value, -1); }

    void finalize() {
        int idx = 0;
        add(1);
        for (auto &kv : values_) {
            kv.second = idx++;
        }
    }

    int to_index(int value) const {
        auto it = values_.find(value);
        ir_assert(it != values_.end());
        return it->second;
    }

    gemm_dim_t dim_;
    std::map<int, int> values_;
};

struct indexed_tile_t {
    indexed_tile_t() {
        for (int i = 0; i < gemm_dim_t::max_id(); i++) {
            auto d = gemm_dim_t::from_id(i);
            dim_mappers_[i] = indexed_dim_t(d);
        }
    }

    void add(gemm_dim_t d, int value) { dim_mappers_[d.id()].add(value); }

    void add(const gemm_tile_t &t) {
        for (auto d : t) {
            add(d, t[d]);
        }
    }

    void finalize() {
        for (auto &d : dim_mappers_)
            if (!d.is_empty()) d.finalize();
    }

    int to_index(const gemm_dim_t &d, int value) const {
        return dim_mappers_[d.id()].to_index(value);
    }

    std::vector<int> to_index(const gemm_tile_t &t) const {
        std::vector<int> ret;
        for (auto &m : dim_mappers_) {
            if (m.is_empty()) continue;
            ret.push_back(to_index(m.dim(), t.at(m.dim(), 1)));
        }
        return ret;
    }

    std::array<indexed_dim_t, gemm_dim_t::max_id()> dim_mappers_;
};

std::vector<std::vector<int>> to_indexed(
        const std::vector<conv_params_t> &params_vec, prop_kind_t prop_kind,
        bool is_transpose) {
    indexed_tile_t iter;
    indexed_tile_t tg;
    indexed_tile_t loop;
    for (auto &p : params_vec) {
        auto &b = p.blocking();
        iter.add(to_gemm(b.iter(), prop_kind, is_transpose));
        tg.add(to_gemm(b.thread_group(), prop_kind, is_transpose));
        loop.add(to_gemm(b.loop(), prop_kind, is_transpose));
    }
    iter.finalize();
    tg.finalize();
    loop.finalize();

    std::vector<std::vector<int>> ret;
    for (auto &p : params_vec) {
        auto &b = p.blocking();
        auto v0 = iter.to_index(to_gemm(b.iter(), prop_kind, is_transpose));
        auto v1 = tg.to_index(
                to_gemm(b.thread_group(), prop_kind, is_transpose));
        auto v2 = loop.to_index(to_gemm(b.loop(), prop_kind, is_transpose));
        std::vector<int> v;
        v.insert(v.end(), v0.begin(), v0.end());
        v.insert(v.end(), v1.begin(), v1.end());
        v.insert(v.end(), v2.begin(), v2.end());
        if (p.id() >= (int)ret.size()) ret.resize(p.id() + 1);
        ret[p.id()] = v;
    }
    return ret;
}

// Helper class to compute the distance between two parameters. During
// initialization every blocking is converted to a GEMM blocking (BMNK
// conention) and then to an indexed vector. After that two blockings are
// compared via L1 distance between corresponding indexed vectors.
// Simplified example:
//   P1: m8n8k16   -> [1, 0, 0]
//   P2: m16n16k16 -> [2, 1, 0]
//   P3: m1n32k16  -> [0, 2, 0]
class params_distance_t {
public:
    params_distance_t() = default;
    params_distance_t(const params_generator_t &g, prop_kind_t prop_kind,
            bool is_transpose) {
        dists_ = to_indexed(g.params_vec(), prop_kind, is_transpose);
    }

    float dist(int id0, int id1) const {
        auto &d0 = dists_[id0];
        auto &d1 = dists_[id1];
        float ret = 0;
        // Use L1 distance between coordinates.
        for (int i = 0; i < (int)d0.size(); i++) {
            ret += std::abs(d0[i] - d1[i]);
        }
        return ret;
    }

private:
    std::vector<std::vector<int>> dists_;
};

// Helper class to track performance data collected during tuning.
class tune_data_t {
public:
    void add_time(int id, uint64_t nsec) {
        resize(id + 1);
        auto &p = points_[id];
        p.id = id;
        p.nsec = std::min(p.nsec, nsec);
        if (p.repeats == 0) reported_points_++;
        p.repeats++;
        if (nsec < best_point_.nsec) best_point_ = p;
    }

    int best_id() const { return best_point_.id; }
    uint64_t nsec(int id) const { return points_[id].nsec; }
    std::vector<int> best_ids(int n) const {
        auto sorted_points = points_;
        std::sort(sorted_points.begin(), sorted_points.end(),
                [&](const bench_point_t &a, const bench_point_t &b) {
                    return a.nsec < b.nsec;
                });
        std::vector<int> ret;
        for (int i = 0; i < std::min((int)sorted_points.size(), n); i++) {
            auto &p = sorted_points[i];
            if (p.id == -1) break;
            ret.push_back(p.id);
        }
        return ret;
    }
    int reported_points() const { return reported_points_; }

    void resize(int new_size) {
        int size = (int)points_.size();
        if (new_size <= size) return;
        points_.resize(new_size);
        for (int i = size; i < new_size; i++) {
            points_[i].id = i;
        }
    }

private:
    static const uint64_t max_nsec_ = std::numeric_limits<uint64_t>::max();

    struct bench_point_t {
        int id = -1;
        int repeats = 0;
        uint64_t nsec = max_nsec_;

        bool is_ok() const { return nsec != max_nsec_; }
    };

    std::vector<bench_point_t> points_;
    int reported_points_ = 0;
    bench_point_t best_point_;
};

// Tuner class.
class conv_tuner_t {
public:
    conv_tuner_t(const conv_config_t &cfg)
        : conv_key_(cfg.key())
        , params_gen_(cfg)
        , params_dist_(params_gen_, cfg.prb().prop_kind(),
                  cfg.prb().ab_swap_transpose)
        , ops_(cfg.prb().ops()) {
        params_gen_.shuffle(conv_key_hash_t()(cfg.key()));
    }

    int configs() const { return 0; }

    void set_params(conv_config_t &cfg) { params_gen_.set_params(cfg); }

    void notify_create(const conv_config_t &cfg) {
        std::lock_guard<std::mutex> lock(mutex_);
        created_configs_++;
    }

    void set_profile_info(uint64_t stamp, const conv_params_t &params) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto &info = stamp_infos_[stamp];
        info.tuner = this;
        info.params = params;
    }

    void add_time(const conv_params_t &params, uint64_t cur_nsec) {
        tune_data_.add_time(params.id(), cur_nsec);
    }

    void finalize(const conv_params_t &params) {
        bool is_best = (tune_data_.best_id() == params.id());
        if (is_best) {
            conv_lookup_table().set(conv_key_.to_filter(), params);
            best_params_dbg_ = params;
        }
        uint64_t nsec = tune_data_.nsec(params.id());
        double gops_sec = ops_ / nsec;
        maybe_print_header();
        std::ostringstream oss;
        oss << "perf,conv,";
        oss << conv_key_.str(/*csv=*/true) << ",";
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

    static conv_tuner_t *get_tuner(const conv_key_t &key, bool do_lock = true) {
        std::unique_lock<std::mutex> lock(mutex_, std::defer_lock_t());
        if (do_lock) lock.lock();
        auto it = conv2tuner_.find(key);
        return it != conv2tuner_.end() ? &it->second : nullptr;
    }

    static conv_tuner_t *get_tuner(
            const conv_config_t &cfg, bool create_if_not_found = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto key = cfg.key();
        auto *tuner = get_tuner(key, /*do_lock=*/false);
        if (tuner) return tuner;
        if (!create_if_not_found) return nullptr;
        auto ret = conv2tuner_.emplace(key, conv_tuner_t(cfg));
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
    struct stamp_info_t {
        conv_tuner_t *tuner;
        conv_params_t params;
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
        for (auto &k : conv_params_t::csv_keys())
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
                [&](const conv_params_t &p) { return dists.at(p.id()); });

        for (int i = beg; i < end; i++) {
            auto &p = params_gen_.at(i);
            ir_trace() << "  " << p << " [dist:" << dists[p.id()] << "]"
                       << std::endl;
        }
    }

    conv_key_t conv_key_;
    params_generator_t params_gen_;
    params_distance_t params_dist_;
    tune_data_t tune_data_;
    conv_params_t best_params_dbg_;

    int created_configs_ = 0;
    double ops_ = 0;

    static std::unordered_map<conv_key_t, conv_tuner_t, conv_key_hash_t>
            conv2tuner_;
    static std::unordered_map<uint64_t, stamp_info_t> stamp_infos_;
    static std::mutex mutex_;
};

std::unordered_map<conv_key_t, conv_tuner_t, conv_key_hash_t>
        conv_tuner_t::conv2tuner_;
std::unordered_map<uint64_t, conv_tuner_t::stamp_info_t>
        conv_tuner_t::stamp_infos_;
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
        cfg.dims().set(get_conv_shape(cfg, /*pad=*/false));
        if (is_tuning_mode()) {
            tuner_->move_next();
            tuner_->set_params(cfg);
        } else {
            params_gen_.move_next();
            params_gen_.set_params(cfg);
        }
    }

    void notify_out_of_registers(const conv_config_t &cfg) {
        if (is_tuning_mode()) return;
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

        switch (mode_) {
            case tiler_mode_t::env_config:
                params_gen_ = params_generator_t(cfg.params());
                break;
            case tiler_mode_t::env_tiler:
                params_gen_ = params_generator_t(
                        cfg, tiler_params().env_params_idx);
                break;
                break;
            case tiler_mode_t::lookup: {
                auto params = const_conv_lookup_table().find(cfg.key());
                blocking_checker_t blocking_checker(cfg);
                bool transposed = cfg.prb().ab_swap_transpose;
                if (!params.is_empty()
                        && (!transposed
                                || blocking_checker.is_ok(params.blocking()))) {
                    if (transposed) {
                        params_gen_ = params_generator_t(cfg, params);
                    } else {
                        params_gen_ = params_generator_t(params);
                    }
                } else {
                    mode_ = tiler_mode_t::model;
                    params_gen_ = params_generator_t(cfg);
                }
                break;
            }
            case tiler_mode_t::tune:
                tuner_ = conv_tuner_t::get_tuner(
                        cfg, /*create_if_not_found=*/true);
                break;
            default: params_gen_ = params_generator_t(cfg); break;
        }
        if (!is_tuning_mode()) {
            ir_assert(!params_gen_.is_empty()) << "No configurations found.";
            sort_by_model_scores(params_gen_, cfg, mode_);
        }
        if (tiler_params().do_list) print_all();
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
void conv_tiler_t::after_create_hook(const conv_config_t &cfg) {
    if (!cfg.tiler().is_tuning_mode()) return;
    auto *tuner = conv_tuner_t::get_tuner(cfg.key());
    tuner->notify_create(cfg);
}

void conv_tiler_t::before_exec_hook(
        const conv_config_t &cfg, stream_t *stream) {
    if (!cfg.tiler().is_tuning_mode()) return;
    if (!stream->is_profiling_enabled()) return;
    auto *tuner = conv_tuner_t::get_tuner(cfg.key());
    auto *compute_stream = utils::downcast<compute::compute_stream_t *>(stream);
    auto &profiler = compute_stream->profiler();
    tuner->set_profile_info(profiler.stamp(), cfg.params());
    profiler.set_callback(conv_tuner_t::profile_callback);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
