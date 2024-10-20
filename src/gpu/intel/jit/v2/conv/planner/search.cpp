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

#include "gpu/intel/jit/v2/conv/planner/search.hpp"

#include "common/profiler.hpp"
#include "gpu/intel/jit/ir/blocking.hpp"
#include "gpu/intel/jit/utils/utils.hpp"
#include "gpu/intel/jit/v2/conv/model.hpp"
#include "gpu/intel/jit/v2/conv/plan.hpp"
#include "gpu/intel/jit/v2/conv/plan_registry.hpp"
#include "gpu/intel/jit/v2/conv/planner/bench.hpp"
#include "gpu/intel/jit/v2/conv/planner/model_fit.hpp"

#include "oneapi/dnnl/dnnl.hpp"

#include <random>
#include <initializer_list>
#include <unordered_map>
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {
namespace planner {

class search_iterator_t {
public:
    int add(const std::vector<int> &key_values) {
        int key = (int)values_.size();
        values_.push_back(key_values);
        idxs_.push_back(0);
        if (key == 0) {
            idxs_[0] = -1;
            total_ = 1;
        }
        total_ *= (int)key_values.size();
        return key;
    }

    int nkeys() const { return (int)values_.size(); }

    bool has_next() const { return idx_ + 1 < total_; }

    void next() {
        ir_assert(has_next());
        int carry = 1;
        for (int j = 0; j < nkeys(); j++) {
            int new_idx = idxs_[j] + carry;
            int bound = (int)values_[j].size();
            idxs_[j] = new_idx % bound;
            carry = new_idx / bound;
            if (carry == 0) break;
        }
        idx_++;
    }

    int operator()(int key) const { return values_[key][idxs_[key]]; }

private:
    int idx_ = -1;
    int total_ = 0;
    std::vector<std::vector<int>> values_;
    std::vector<int> idxs_;
};

// Flags specifying blocking restrictions for a convolution dimension.
enum class tile_flags_t : uint32_t {
    undef = 0,
    // Dimension participates in loop blocking.
    loop = (1 << 0),
    // Dimension participates in thread group blocking.
    thread_group = (1 << 1),
    // Dimension participates in iteration blocking.
    iter = (1 << 2),
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

inline std::string to_string(tile_flags_t flags) {
    std::ostringstream oss;
    if (any(flags & tile_flags_t::loop)) oss << "l";
    if (any(flags & tile_flags_t::thread_group)) oss << "t";
    if (any(flags & tile_flags_t::iter)) oss << "i";
    return oss.str();
}

inline std::vector<int> pow_range(int a, int b, int step) {
    std::vector<int> ret;
    for (int i = a; i <= b; i *= step)
        ret.push_back(i);
    return ret;
}

struct tile_info_t {
    pvar_t dim;
    tile_flags_t flags = tile_flags_t::undef;

    tile_info_t() = default;
    tile_info_t(const pvar_t &dim) : dim(dim) {}

    void add(tile_flags_t f) { flags = flags | f; }

    std::vector<int> iter_tiles() const {
        if (!any(flags & tile_flags_t::iter)) return {1};
        return pow_range(8, 64, 2);
    }

    std::vector<int> thread_group_tiles() const {
        if (!any(flags & tile_flags_t::thread_group)) return {1};
        return pow_range(1, 16, 2);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << dim << ": " << to_string(flags);
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

class tile_scheme_t {
public:
    tile_scheme_t() = default;
    tile_scheme_t(const std::string &s) {
        std::vector<std::string> parts;
        std::vector<size_t> key_idxs;
        std::string cur;
        int beg = -1;
        for (int i = 0; i <= (int)s.length(); i++) {
            char c = (i <= (int)s.length() ? s[i] : ' ');
            bool is_id = (std::isdigit(c) || std::isalpha(c) || c == '_');
            if (beg != -1 && is_id) continue;
            if (beg == -1 && is_id) {
                beg = i;
            } else if (beg != -1 && !is_id) {
                parts.push_back(s.substr(beg, i - beg));
                beg = -1;
            }
            if (c == '=') key_idxs.push_back(parts.size() - 1);
        }
        key_idxs.push_back(parts.size());
        std::unordered_map<std::string, std::vector<std::string>> k2v;
        for (int i = 0; i < (int)key_idxs.size() - 1; i++) {
            int cur = key_idxs[i];
            int next = key_idxs[i + 1];
            for (int j = cur + 1; j < next; j++) {
                set(parts[cur], parts[j]);
            }
        }
    }

    void unset(const pvar_t &dim) { tile_infos_.unset(dim); }

    std::vector<pvar_t> dims() const { return tile_infos_.keys(); }
    const tile_info_t &tile_info(const pvar_t &dim) const {
        return tile_infos_.at(dim);
    }

private:
    void set(const std::string &key, const std::string &value) {
        if (key == "iter") {
            auto dim = pvar_t(value);
            tile_infos_[dim].add(tile_flags_t::iter);
        } else if (key == "tg") {
            auto dim = pvar_t(value);
            tile_infos_[dim].add(tile_flags_t::thread_group);
        } else {
            ir_error_not_expected();
        }
    }

    pvar_map_t<tile_info_t> tile_infos_;
};

struct dim_tile_t {
    int loop = 0;
    int tg = 0;
    int iter = 0;

    std::string str() const {
        std::ostringstream oss;
        if (loop != 0) oss << "l" << loop;
        if (tg != 0) oss << "t" << tg;
        if (iter != 0) oss << "i" << iter;
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

std::ostream &operator<<(std::ostream &out, const dim_tile_t &tile) {
    out << tile.str();
    return out;
}

struct tiling_desc_t {
    pvar_tile_t iter;
    pvar_tile_t thread_group;

    void set(const pvar_t &dim, const dim_tile_t &tile) {
        if (tile.iter != 1) iter[dim] = tile.iter;
        if (tile.tg != 1) thread_group[dim] = tile.tg;
    }

    void unset(const pvar_t &dim) {
        iter.unset(dim);
        thread_group.unset(dim);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "iter: " << iter.str();
        oss << " thread_group: " << thread_group.str();
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

class dim_tile_set_t {
public:
    dim_tile_set_t(const tile_scheme_t &scheme) : dims_(scheme.dims()) {
        for (auto &d : scheme.dims()) {
            auto &d_tiles = tiles_[d];
            d_tiles = get_dim_tiles(scheme, d);
        }
    }

    std::vector<tiling_desc_t> create_tiling_descs() const {
        std::vector<tiling_desc_t> ret;
        tiling_desc_t tiling_desc;
        std::vector<int> cur_idxs(dims_.size());
        product_impl(0, cur_idxs, tiling_desc, ret);
        return ret;
    }

private:
    void product_impl(int idx, std::vector<int> &cur_idxs,
            tiling_desc_t &tiling_desc, std::vector<tiling_desc_t> &ret) const {
        if (idx == (int)dims_.size()) {
            ret.push_back(tiling_desc);
            return;
        }
        auto &v = tiles_.at(dims_[idx]);
        for (int i = 0; i < (int)v.size(); i++) {
            cur_idxs[idx] = i;
            tiling_desc.set(dims_[idx], v[i]);
            product_impl(idx + 1, cur_idxs, tiling_desc, ret);
            tiling_desc.unset(dims_[idx]);
        }
    }

    static std::vector<dim_tile_t> get_dim_tiles(
            const tile_scheme_t &scheme, const pvar_t &dim) {
        std::vector<dim_tile_t> ret;
        auto &info = scheme.tile_info(dim);
        auto iter_tiles = info.iter_tiles();
        auto tg_tiles = info.thread_group_tiles();
        for (int iter : iter_tiles) {
            for (int tg : tg_tiles) {
                dim_tile_t tile;
                tile.iter = iter;
                tile.tg = tg;
                ret.push_back(tile);
            }
        }
        return ret;
    }

    std::vector<pvar_t> dims_;
    pvar_map_t<std::vector<dim_tile_t>> tiles_;
};

std::vector<tile_scheme_t> get_tile_schemes(prop_kind_t prop, bool is_dw) {
    std::vector<tile_scheme_t> schemes;
    if (prop == prop_kind::forward) {
        schemes.emplace_back("tg=[ic],    iter=[mb,g,oc,ic]");
        schemes.emplace_back("tg=[ic],    iter=[ow,g,oc,ic]");
        schemes.emplace_back("tg=[oc,mb], iter=[mb,g,oc,ic]");
        schemes.emplace_back("tg=[oc,mb], iter=[ow,g,oc,ic]");
        schemes.emplace_back("tg=[oc,ow], iter=[mb,g,oc,ic]");
        schemes.emplace_back("tg=[oc,ow], iter=[ow,g,oc,ic]");
    } else if (prop == prop_kind::backward_data) {
        schemes.emplace_back("tg=[ic,iw], iter=[mb,g,oc,ic]");
        schemes.emplace_back("tg=[ic,mb], iter=[mb,g,oc,ic]");
        schemes.emplace_back("tg=[ic,iw], iter=[iw,g,oc,ic]");
    } else if (prop == prop_kind::backward_weights) {
        schemes.emplace_back("tg=[oc,ic], iter=[mb,g,oc,ic]");
        schemes.emplace_back("tg=[oc,ic], iter=[ow,g,oc,ic]");
    } else {
        ir_error_not_expected();
    }
    for (auto &s : schemes) {
        if (is_dw) {
            s.unset(pvars::ic);
            s.unset(pvars::oc);
        } else {
            s.unset(pvars::g);
        }
    }
    return schemes;
}

// A group of kernel descriptors sharing the same set of requriements.
class search_kernel_desc_group_t {
public:
    search_kernel_desc_group_t() = default;
    search_kernel_desc_group_t(const prb_reqs_t &reqs) : reqs_(reqs) {}

    const prb_reqs_t &reqs() const { return reqs_; }
    const std::vector<kernel_desc_t> &descs() const { return descs_; }

    void add_desc(const kernel_desc_t &desc) {
        ir_assert(desc.reqs.str() == reqs_.str())
                << "Reqs mismatch:\n"
                << desc.cmd_str() << "\ndesc.reqs:" << desc.reqs.str()
                << "\nreqs:\n"
                << reqs_.str();
        if (descs_.empty()) {
            is_dw_ = desc.is_dw;
        } else {
            ir_assert(desc.is_dw == is_dw_);
        }
        descs_.push_back(desc);
    }

    bench_input_params_t bench_input_params(int nprbs) const {
        if (descs_.empty()) return bench_input_params_t();
        auto &kd = descs_.front();
        bench_input_params_t params;
        params.hw = kd.hw;
        params.prop = kd.prop;
        params.src_tag = kd.src_tag;
        params.wei_tag = kd.wei_tag;
        params.dst_tag = kd.dst_tag;
        params.reqs = reqs_;
        params.is_dw = is_dw_;
        params.nprbs = nprbs;
        return params;
    }

private:
    prb_reqs_t reqs_;
    std::vector<kernel_desc_t> descs_;
    bool is_dw_ = false;
};

bench_data_set_t bench_kernel_desc_group(const bench_manager_t &bench_mger,
        const search_kernel_desc_group_t &desc_group, int nprbs, int max_descs);

class kernel_search_manager_t {
public:
    // Number of problems to generate to rank kernel descriptors in a kernel
    // descriptor group.
    static const int bench_nprbs = 50;
    // Number of problems to generate to build performance model.
    static const int model_nprbs = 250;
    // Number of top kernel descriptors in a kernel descriptor group to save to
    // registry.
    static const int registry_top_k = 8;
    // Number of descriptors to search through.
    static const int max_descs = 256;

    kernel_search_manager_t(
            const bench_manager_t &bench_mger, const kernel_desc_t &base_desc)
        : bench_mger_(bench_mger), base_desc_(base_desc) {
        reset_reqs(base_desc_);
    }

    void search() {
        std::cout << "Starting kernel search" << std::endl;
        auto desc_groups = gen_desc_groups();
        auto &registry = plan_registry();
        for (auto &dg : desc_groups) {
            auto bench_data_set = bench_kernel_desc_group(
                    bench_mger_, dg, bench_nprbs, max_descs);
            auto best = bench_data_set.find_best(registry_top_k);
            for (auto &bd : best) {
                auto &d = bd.kernel_desc;
                auto bd_model = bench(bench_mger_, d, model_nprbs);
                if (!bd_model) continue;
                auto model = model_fit(bd_model);
                auto d_ext = try_extensions(bench_mger_, d);
                registry.set(d_ext, model);
            }
        }
        std::cout << "Kernel search completed" << std::endl;
    }

private:
    static void reset_reqs(kernel_desc_t &kernel_desc) {
        if (kernel_desc.prop != prop_kind::backward_data) return;
        // XXX: No stride support in backward by data yet.
        kernel_desc.reqs.add(pvars::sw.var() == 1);
        kernel_desc.reqs.add(pvars::sh.var() == 1);
        kernel_desc.reqs.add(pvars::sd.var() == 1);
    }

    std::vector<search_kernel_desc_group_t> gen_desc_groups() const {
        std::unordered_map<std::string, kernel_desc_t> descs;
        for (auto &s : get_tile_schemes(base_desc_.prop, base_desc_.is_dw)) {
            dim_tile_set_t tile_set(s);
            auto tiling_descs = tile_set.create_tiling_descs();
            for (auto &td : tiling_descs) {
                auto d = base_desc_;
                d.thread_group_tile = td.thread_group;
                d.iter_tile = td.iter;
                if (!finalize_conv_desc(d, bench_mger_.hw())) {
                    std::cout << d.brief_str() << ": \033[1;31mFAIL\033[0m"
                              << std::endl;
                    continue;
                }
                auto d_key = jit::stringify(d);
                if (descs.find(d_key) != descs.end()) continue;
                descs[d_key] = d;
                std::cout << d.brief_str() << ": \033[1;32mOK\033[0m"
                          << std::endl;
            }
        }
        ir_info() << "gen_desc_groups(): descs.size() = " << descs.size()
                  << std::endl;
        std::unordered_map<std::string, search_kernel_desc_group_t> desc_groups;
        for (auto &kv : descs) {
            auto &d = kv.second;
            auto ret = desc_groups.emplace(
                    d.reqs.str(), search_kernel_desc_group_t(d.reqs));
            ret.first->second.add_desc(d);
            for (int dist : {1, 3}) {
                auto _d = d;
                _d.prefetch = prefetch_desc_t {dist, true, true};
                reset_reqs(_d);
                _d.is_finalized = false;
                if (!finalize_conv_desc(_d, bench_mger_.hw())) {
                    std::cout << d.brief_str() << ": \033[1;31mFAIL\033[0m"
                              << std::endl;
                    continue;
                }
                std::cout << _d.brief_str() << ": \033[1;32mOK\033[0m"
                          << std::endl;
                ret.first->second.add_desc(_d);
            }
        }
        std::vector<search_kernel_desc_group_t> ret;
        for (auto &kv : desc_groups) {
            ret.push_back(kv.second);
        }
        std::cout << "Generated " << ret.size()
                  << " kernel descriptor groups\n";
        return ret;
    }

    static std::vector<pvar_tile_t> generate_iter_outer_tiles(
            const kernel_desc_t &desc) {
        std::vector<pvar_tile_t> tiles = {pvar_tile_t()};
        for (auto &d : desc.iter_tile) {
            auto bmnk = to_gemm(d, desc.prop);
            if (!utils::one_of(bmnk, pvars::m, pvars::n)) continue;
            for (int outer : {2, 4}) {
                if (desc.iter_tile.at(d) % outer != 0) continue;
                pvar_tile_t tile_outer;
                tile_outer[d] = outer;
                tiles.push_back(tile_outer);
            }
        }
        return tiles;
    }

    // TODO: Use search_desc.
    void search_desc(const kernel_desc_t &_desc) const {
        auto iter_outer_tiles = generate_iter_outer_tiles(_desc);
        auto &registry = plan_registry();
        for (auto &iter_outer : iter_outer_tiles) {
            auto desc = _desc;
            desc.iter_outer_tile = iter_outer;
            std::cout << "Running benchmark for descriptor: " << desc.cmd_str()
                      << std::endl;
            auto bd = bench(bench_mger_, desc);
            if (!bd) {
                std::cout << "Benchmarking failed" << std::endl;
                continue;
            }
            auto model = model_fit(bd);
            registry.set(desc, model);
            return;
        }
    }

    const bench_manager_t &bench_mger_;
    kernel_desc_t base_desc_;
};

class search_sequence_t {
public:
    search_sequence_t(const std::vector<kernel_desc_t> &descs, int max_entries)
        : max_entries_(max_entries) {
        std::vector<std::vector<pvar_tile_t>> tiles;
        pvar_t prefetch_dim("p");
        for (int i = 0; i < (int)descs.size(); i++) {
            auto &d = descs[i];
            entries_.emplace_back(i, d);
            std::vector<pvar_tile_t> d_tiles;
            auto iter = to_gemm(d.iter_tile, d.prop);
            auto tg = to_gemm(d.thread_group_tile, d.prop);
            d_tiles.push_back(iter);
            d_tiles.push_back(tg);
            pvar_tile_t prefetch_tile;
            prefetch_tile[prefetch_dim] = d.prefetch.dist;
            d_tiles.push_back(prefetch_tile);
            tiles.push_back(std::move(d_tiles));
        }
        tile_to_vec_ = tile_to_vec_t(tiles);
        entry_it_ = entries_.begin();
        std::default_random_engine rng(0);
        std::shuffle(entries_.begin(), entries_.end(), rng);
    }

    explicit operator bool() const {
        return entry_idx_ < max_entries_ && entry_it_ != entries_.end();
    }

    std::pair<int, kernel_desc_t> next() {
        ir_assert((bool)*this);
        auto &e = *entry_it_;
        ++entry_it_;
        return std::make_pair(e.id, e.desc);
    }

    void update(const bench_data_set_t &data_set) {
        entry_idx_++;
        if (batch_entry_idx_++ < rescore_period_) return;
        batch_entry_idx_ = 0;

        const int nbest = 5;
        auto best_ids = data_set.find_best_ids(nbest);
        std::unordered_map<int, float> min_dists;
        for (auto it = entry_it_; it != entries_.end(); ++it) {
            min_dists[it->id] = std::numeric_limits<float>::max();
            for (auto &id : best_ids) {
                min_dists[it->id] = std::min(
                        min_dists[it->id], tile_to_vec_.dist(it->id, id));
            }
        }
        std::sort(entry_it_, entries_.end(),
                [&](const entry_t &a, const entry_t &b) {
                    return min_dists[a.id] < min_dists[b.id];
                });
    }

private:
    struct entry_t {
        int id = -1;
        kernel_desc_t desc;

        entry_t(int id, const kernel_desc_t &desc) : id(id), desc(desc) {}
    };

    static const int rescore_period_ = 16;

    std::vector<entry_t> entries_;
    std::vector<entry_t>::iterator entry_it_;
    tile_to_vec_t tile_to_vec_;

    // The indices below are tracked only for successfully created kernels
    // (update() must be called).
    int batch_entry_idx_ = 0;
    int entry_idx_ = 0;
    int max_entries_ = 0;
};

bench_data_set_t bench_kernel_desc_group(const bench_manager_t &bench_mger,
        const search_kernel_desc_group_t &desc_group, int nprbs,
        int max_descs) {
    auto eng = bench_mger.get_engine();
    bench_runner_t runner(bench_mger, desc_group.bench_input_params(nprbs));
    bench_data_set_t bd_set;
    search_sequence_t seq(desc_group.descs(), max_descs);
    while (seq) {
        auto seq_next = seq.next();
        int kernel_desc_id = seq_next.first;
        auto &kernel_desc = seq_next.second;
        auto bd = runner.bench(kernel_desc);
        if (!bd) continue;
        bd.id = kernel_desc_id;
        bd_set.add(bd);
        seq.update(bd_set);
    }

    return bd_set;
}

void search(const bench_manager_t &bench_mger, const kernel_desc_t &desc) {
    kernel_search_manager_t mger(bench_mger, desc);
    mger.search();
}

void auto_search(const bench_manager_t &bench_mger) {
    // clang-format off
    std::vector<const char *> recipes = {
        "--hw xehpc --prop fwd --src axb:s8 --wei axcb:s8 --dst axb:s8 --fma dpas --simd 16 --regs 256 --2d 1",
        "--hw xehpc --prop fwd --src axb:s8 --wei axcb:s8 --dst axb:s8 --fma dpas --simd 16 --regs 256 --align 1",
        "--hw xehpc --prop fwd --src axb:bf16 --wei axcb:bf16 --dst axb:bf16 --fma dpas --simd 16 --regs 256 --2d 1",
        "--hw xehpc --prop fwd --src axb:bf16 --wei axcb:bf16 --dst axb:bf16 --fma dpas --simd 16 --regs 256 --align 1",
        "--hw xehpc --prop fwd --src axb:f32 --wei axcb:f32 --dst axb:f32 --fma mad --simd 32 --regs 128 --2d 1",
        "--hw xehpc --prop fwd --src axb:f32 --wei axcb:f32 --dst axb:f32 --fma mad --simd 32 --regs 128 --align 1",
        "--hw xehpc --prop bwd_d --src axb:bf16 --wei axbc:bf16 --dst axb:bf16 --fma dpas --simd 16 --regs 256 --2d 1",
        "--hw xehpc --prop bwd_d --src axb:bf16 --wei axbc:bf16 --dst axb:bf16 --fma dpas --simd 16 --regs 256 --align 1",
        "--hw xehpc --prop bwd_d --src axb:f32 --wei axbc:f32 --dst axb:f32 --fma mad --simd 32 --regs 128 --2d 1",
        "--hw xehpc --prop bwd_d --src axb:f32 --wei axbc:f32 --dst axb:f32 --fma mad --simd 32 --regs 128 --align 1",
        "--hw xehpc --prop bwd_w --src axb:bf16 --wei axcb:bf16 --dst axb:bf16 --fma dpas --simd 16 --regs 256 --2d 1",
        "--hw xehpc --prop bwd_w --src axb:bf16 --wei axcb:bf16 --dst axb:bf16 --fma dpas --simd 16 --regs 256 --align 1",
        "--hw xehpc --prop bwd_w --src axb:f32 --wei axcb:f32 --dst axb:f32 --fma mad --simd 16 --regs 128 --2d 1",
        "--hw xehpc --prop bwd_w --src axb:f32 --wei axcb:f32 --dst axb:f32 --fma mad --simd 16 --regs 128 --align 1",
        "--hw xehpc --dw 1 --prop fwd --src axb:s8 --wei axcb:s8 --dst axb:s8 --fma mad --simd 32 --regs 128 --align 1",
        "--hw xehpc --dw 1 --prop fwd --src axb:bf16 --wei axcb:bf16 --dst axb:bf16 --fma mad --simd 32 --regs 128 --align 1",
        "--hw xehpc --dw 1 --prop fwd --src axb:f32 --wei axcb:f32 --dst axb:f32 --fma mad --simd 32 --regs 128 --align 1",
        "--hw xehpc --dw 1 --prop bwd_d --src axb:bf16 --wei axbc:bf16 --dst axb:bf16 --fma mad --simd 32 --regs 128 --align 1",
        "--hw xehpc --dw 1 --prop bwd_d --src axb:f32 --wei axbc:f32 --dst axb:f32 --fma mad --simd 32 --regs 128 --align 1",
        "--hw xehpc --dw 1 --prop bwd_w --src axb:bf16 --wei axcb:bf16 --dst axb:bf16 --fma mad --simd 16 --regs 128 --align 1",
        "--hw xehpc --dw 1 --prop bwd_w --src axb:f32 --wei axcb:f32 --dst axb:f32 --fma mad --simd 32 --regs 128 --align 1",
    };
    // clang-format on
    double t = get_msec();
    for (const char *_r : recipes) {
        auto r = std::string(_r) + " --iter x --tg x";
        kernel_desc_t desc;
        desc.set(r);
        desc.hw = hw_t(bench_mger.get_engine().get());
        search(bench_mger, desc);
    }
    t = get_msec() - t;
    std::cout << "Kernel search done, took: " << t / 1e3 << " sec" << std::endl;
}

} // namespace planner
} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
