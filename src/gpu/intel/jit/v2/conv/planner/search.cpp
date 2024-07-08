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
    prb_dim_t dim;
    tile_flags_t flags = tile_flags_t::undef;

    tile_info_t() = default;
    tile_info_t(const prb_dim_t &dim) : dim(dim) {}

    void add(tile_flags_t f) { flags = flags | f; }

    std::vector<int> iter_tiles() const {
        if (!any(flags & tile_flags_t::iter)) return {1};
        return pow_range(1, 64, 2);
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

    std::vector<prb_dim_t> dims() const { return tile_infos_.keys(); }
    const tile_info_t &tile_info(const prb_dim_t &dim) const {
        return tile_infos_.at(dim);
    }

private:
    void set(const std::string &key, const std::string &value) {
        if (key == "iter") {
            auto dim = prb_dim_t::from_name(value);
            tile_infos_[dim].add(tile_flags_t::iter);
        } else if (key == "tg") {
            auto dim = prb_dim_t::from_name(value);
            tile_infos_[dim].add(tile_flags_t::thread_group);
        } else {
            ir_error_not_expected();
        }
    }

    dim_map_t<prb_dim_t, tile_info_t> tile_infos_;
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
    prb_tile_t iter;
    prb_tile_t thread_group;

    void set(const prb_dim_t &dim, const dim_tile_t &tile) {
        if (tile.iter != 1) iter[dim] = tile.iter;
        if (tile.tg != 1) thread_group[dim] = tile.tg;
    }

    void unset(const prb_dim_t &dim) {
        iter.unset(dim);
        thread_group.unset(dim);
    }
};

class dim_tile_set_t {
public:
    dim_tile_set_t(const tile_scheme_t &scheme, bool is_dw = false)
        : dims_(scheme.dims()) {
        for (auto &d : scheme.dims()) {
            auto &d_tiles = tiles_[d];
            if (is_dw && utils::one_of(d, prb_dims::ic, prb_dims::oc)) {
                dim_tile_t tile;
                tile.iter = 1;
                tile.tg = 1;
                d_tiles = {tile};
            } else
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
            const tile_scheme_t &scheme, const prb_dim_t &dim) {
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

    std::vector<prb_dim_t> dims_;
    dim_map_t<prb_dim_t, std::vector<dim_tile_t>> tiles_;
};

std::vector<tile_scheme_t> get_tile_schemes(prop_kind_t prop) {
    std::vector<tile_scheme_t> schemes;
    if (prop == prop_kind::forward) {
        schemes.emplace_back("tg=[ow], iter=[mb,g,oc,ic]");
        schemes.emplace_back("tg=[oc,mb], iter=[mb,g,oc,ic]");
        schemes.emplace_back("tg=[ic,ow], iter=[ow,g,oc,ic]");
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
    return schemes;
}

class kernel_search_manager_t {
public:
    kernel_search_manager_t(
            const bench_manager_t &bench_mger, const kernel_desc_t &base_desc)
        : bench_mger_(bench_mger), base_desc_(base_desc) {}

    void search() {
        std::cout << "Starting kernel search" << std::endl;
        auto &registry = plan_registry();
        auto descs = gen_descs();
        for (size_t i = 0; i < descs.size(); i++) {
            auto &d = descs[i];
            std::cout << "Running benchmark for descriptor: " << d.cmd_str()
                      << std::endl;
            auto bd = bench(bench_mger_, d);
            if (!bd) std::cout << "Benchmarking failed" << std::endl;
            if (!bd) continue;
            auto model = model_fit(bd);
            registry.set(d, model);
        }
        std::cout << "Kernel search completed" << std::endl;
    }

private:
    std::vector<kernel_desc_t> gen_descs() const {
        std::unordered_set<kernel_desc_t, ir_utils::hasher_t<kernel_desc_t>>
                descs;
        for (auto &s : get_tile_schemes(base_desc_.prop)) {
            dim_tile_set_t tile_set(s, base_desc_.is_dw);
            auto tiling_descs = tile_set.create_tiling_descs();
            for (auto &td : tiling_descs) {
                auto d = base_desc_;
                d.thread_group_tile = td.thread_group;
                d.iter_tile = td.iter;
                if (!is_supported(d)) continue;
                descs.insert(d);
            }
        }
        std::vector<kernel_desc_t> ret;
        ret.insert(ret.end(), descs.begin(), descs.end());
        std::minstd_rand seed;
        std::shuffle(ret.begin(), ret.end(), seed);
        ret.resize(std::min((int)ret.size(), 8));
        std::cout << "Generated " << ret.size() << " kernel descriptors"
                  << std::endl;
        return ret;
    }

    bool is_supported(kernel_desc_t &desc) const {
        if (!desc.is_supported()) return false;
        auto plan = create_conv_plan(desc);
        if (!plan) return false;
        desc.finalize(plan);
        return true;
    }

    const bench_manager_t &bench_mger_;
    kernel_desc_t base_desc_;
};

void search(const bench_manager_t &bench_mger, const kernel_desc_t &desc) {
    kernel_search_manager_t mger(bench_mger, desc);
    mger.search();
}

void auto_search(const bench_manager_t &bench_mger) {
    // clang-format off
    std::vector<const char *> recipes = {
        "--prop fwd --src axb:f32 --wei axcb:f32 --dst axb:f32 --hw xehpc --fma mad --simd 16 --regs 128",
        "--prop fwd --src axb:f32 --wei axcb:f32 --dst axb:f32 --hw xehpc --fma mad --simd 16 --regs 128 --load a:2d,b:2d --store c:2d",
        "--prop fwd --src axb:s8 --wei axcb:s8 --dst axb:s8 --hw xehpc --fma dpas --simd 16 --regs 256 --load a:2d,b:2d --store c:2d --prefetch x3",
        "--prop fwd --src axb:s8 --wei axcb:s8 --dst axb:s8 --hw xehpc --fma dpas --simd 16 --regs 256",

        "--prop bwd_d --src axb:f32 --wei axcb:f32 --dst axb:f32 --hw xehpc --fma mad --simd 16 --regs 128 --spec-reqs sw1sh1sd1",
        "--prop bwd_d --src axb:f32 --wei axcb:f32 --dst axb:f32 --hw xehpc --fma mad --simd 16 --regs 128 --load a:2d,b:2d --store c:2d --spec-reqs sw1sh1sd1",
        "--prop bwd_d --src axb:s8 --wei axcb:s8 --dst axb:s8 --hw xehpc --fma dpas --simd 16 --regs 256 --load a:2d,b:2d --store c:2d --prefetch x3 --spec-reqs sw1sh1sd1",
        "--prop bwd_d --src axb:s8 --wei axcb:s8 --dst axb:s8 --hw xehpc --fma dpas --simd 16 --regs 256 --spec-reqs sw1sh1sd1",

        "--prop bwd_w --src axb:f32 --wei axcb:f32 --dst axb:f32 --hw xehpc --fma mad --simd 16 --regs 128",
        "--prop bwd_w --src axb:f32 --wei axcb:f32 --dst axb:f32 --hw xehpc --fma mad --simd 16 --regs 128 --load a:2d,b:2d --store c:2d",
    };
    // clang-format on
    for (const char *r : recipes) {
        kernel_desc_t desc;
        desc.set(r);
        desc.hw = hw_t(bench_mger.get_engine().get());
        search(bench_mger, desc);
    }
}

} // namespace planner
} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
