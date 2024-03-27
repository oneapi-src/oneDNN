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

#include <random>

#include "gpu/jit/ir/blocking.hpp"
#include "gpu/jit/ir/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

std::vector<int> tile_info_t::iter_blocks(int size) const {
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

std::vector<int> tile_info_t::thread_group_blocks(int size) const {
    std::vector<int> ret;
    int bound = any(flags & tile_flags_t::thread_group) ? max_thread_group_blk
                                                        : 1;
    for (int i = 1; i <= bound; i *= 2) {
        int size_padded = utils::rnd_up(size, i);
        double eff = (double)size / size_padded;
        if (eff >= 0.75) ret.push_back(i);
    }
    return ret;
}

std::vector<int> tile_info_t::loop_blocks(int size, int iter_blk) const {
    if (!any(flags & tile_flags_t::loop)) return {1};
    if (any(flags & tile_flags_t::loop_span)) return {size};
    if (any(flags & tile_flags_t::loop_iter_unroll)) {
        int blk = math::lcm(div_info.unroll_unit, iter_blk);
        return {blk / iter_blk};
    }
    return get_loop_blocks(size);
}

std::vector<int> tile_info_t::get_factors(int n) {
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

std::vector<int> tile_info_t::get_loop_blocks(int n) {
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

namespace levels {
level_t loop(level_kind_t::loop);
level_t thread_group(level_kind_t::thread_group);
level_t iter(level_kind_t::iter);
} // namespace levels

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

void params_generator_t::set_params(prim_config_t &cfg) {
    auto &params = params_vec_[cur_idx_];
    ir_trace() << "set params #" << cur_idx_ << ": " << params << std::endl;
    cfg.set_params(params);
}

void params_generator_t::shuffle(size_t seed) {
    std::minstd_rand g(static_cast<uint32_t>(seed));
    std::shuffle(params_vec_.begin(), params_vec_.end(), g);
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

int level_tile_set_t::count() const {
    int ret = 1;
    int ntiles = (int)tiles_.size();
    for (int i = 0; i < ntiles; i++) {
        if (deps_[i] != -1) continue;
        ret *= (int)tiles_[i].size();
    }
    return ret;
}

std::vector<blocking_t> level_tile_set_t::product(int simd) const {
    std::vector<blocking_t> ret;
    blocking_t blk;
    blk.set_simd(simd);
    std::vector<int> cur_idxs(dims_.size());
    product_impl(0, cur_idxs, blk, ret);
    return ret;
}

std::vector<blocking_t> level_tile_set_t::sample(int target,
        const std::function<bool(const blocking_t &)> &is_ok, int simd,
        int tries_mult_bound) const {
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
        if (!is_ok(blk)) continue;
        ret.push_back(blk);
        if ((int)ret.size() >= target) break;
    }
    return ret;
}

void level_tile_set_t::set(
        blocking_t &blk, const prb_dim_t &dim, const level_tile_t &tile) {
    if (tile.has(levels::loop)) blk.set_loop(dim, tile[levels::loop]);
    if (tile.has(levels::thread_group))
        blk.set_thread_group(dim, tile[levels::thread_group]);
    if (tile.has(levels::iter)) blk.set_iter(dim, tile[levels::iter]);
}

void level_tile_set_t::product_impl(int idx, std::vector<int> &cur_idxs,
        blocking_t &blk, std::vector<blocking_t> &ret) const {
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

std::vector<level_tile_t> level_tile_set_t::sample(
        ir_utils::fast_random_t &r) const {
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

void blocking_generator_t::generate_all(int vec_size, blocking_checker_t &chk,
        const level_tile_set_t &level_tile_set) {
    auto ts_blockings = level_tile_set.product(vec_size);
    for (;;) {
        bool added = false;
        for (auto &b : ts_blockings) {
            if (!chk.is_ok(b)) continue;
            added = true;
            blockings_.insert(b);
        }
        if (!added && chk.relax_checks()) continue;
        break;
    }
    chk.reset_checks();
}

void blocking_generator_t::generate_sample(int vec_size,
        const blocking_checker_t &chk, const level_tile_set_t &level_tile_set) {
    ir_assert(false);
    int target_size = 1;
    auto is_ok = [&](const blocking_t &blk) { return chk.is_ok(blk); };
    auto ts_blockings = level_tile_set.sample(target_size, is_ok, vec_size);
    blockings_.insert(ts_blockings.begin(), ts_blockings.end());
}

params_generator_t::params_generator_t(const blocking_params_t &params) {
    append_params(params_vec_, params);
    assign_ids(params_vec_);
}

params_generator_t::params_generator_t(int tune_level, int simd_size,
        blocking_checker_t &chk,
        const std::vector<level_tile_set_t> &level_tile_sets,
        const blocking_params_t &params) {
    append_params(params_vec_, params);
    append_params(params_vec_, level_tile_sets, chk, tune_level, simd_size);
    assign_ids(params_vec_);
}

params_generator_t::params_generator_t(int tune_level, int simd_size,
        blocking_checker_t &chk,
        const std::vector<level_tile_set_t> &level_tile_sets, int idx) {
    append_params(params_vec_, level_tile_sets, chk, tune_level, simd_size);
    if (idx != -1) {
        ir_assert(idx >= 0 && idx < configs());
        std::vector<blocking_params_t> temp_vec;
        temp_vec.swap(params_vec_);
        append_params(params_vec_, temp_vec[idx]);
    }
    assign_ids(params_vec_);
}

void params_generator_t::assign_ids(std::vector<blocking_params_t> &vec) {
    for (int i = 0; i < int(vec.size()); i++)
        vec[i].set_id(i);
}

void params_generator_t::append_params(
        std::vector<blocking_params_t> &vec, const blocking_params_t &params) {
    vec.emplace_back(params);
}

void params_generator_t::append_params(std::vector<blocking_params_t> &vec,
        const std::vector<level_tile_set_t> &level_tile_sets,
        blocking_checker_t &chk, int tune_level, int simd_size) {
    blocking_generator_t bg(simd_size, chk, level_tile_sets);
    for (auto &b : bg.blockings()) {
        vec.emplace_back(b);
        if (tune_level > 0) vec.emplace_back(b, /* bufs_hint = */ 0);
    }
}

const tiler_params_t &tiler_params() {
    static tiler_params_t params = []() {
        tiler_params_t ret;
        auto s_opts = gpu_utils::dev_getenv("tiler", std::string());
        if (s_opts.empty()) return ret;
        auto opts = gpu_utils::split(s_opts, ",");
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
            auto sub_opts = gpu_utils::split(opt, ":");
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

params_distance_t::params_distance_t(
        const std::vector<blocking_params_t> &params_vec,
        const std::function<prb_tile_t(const prb_tile_t &)> &convert) {
    indexed_tile_t iter;
    indexed_tile_t tg;
    indexed_tile_t loop;
    for (auto &p : params_vec) {
        auto &b = p.blocking();
        iter.add(convert(b.iter()));
        tg.add(convert(b.thread_group()));
        loop.add(convert(b.loop()));
    }
    iter.finalize();
    tg.finalize();
    loop.finalize();

    std::vector<std::vector<int>> ret;
    for (auto &p : params_vec) {
        auto &b = p.blocking();
        auto v0 = iter.to_index(convert(b.iter()));
        auto v1 = tg.to_index(convert(b.thread_group()));
        auto v2 = loop.to_index(convert(b.loop()));
        std::vector<int> v;
        v.insert(v.end(), v0.begin(), v0.end());
        v.insert(v.end(), v1.begin(), v1.end());
        v.insert(v.end(), v2.begin(), v2.end());
        if (p.id() >= int(dists_.size())) dists_.resize(p.id() + 1);
        dists_[p.id()] = v;
    }
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
