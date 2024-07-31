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

#ifndef GPU_INTEL_JIT_V2_CONV_KERNEL_DESC_HPP
#define GPU_INTEL_JIT_V2_CONV_KERNEL_DESC_HPP

#include "gpu/intel/jit/ir/fma.hpp"
#include "gpu/intel/jit/ir/hw.hpp"
#include "gpu/intel/jit/ir/kernel_desc.hpp"
#include "gpu/intel/jit/ir/message.hpp"
#include "gpu/intel/jit/v2/conv/problem.hpp"
#include "gpu/intel/jit/v2/ir/reqs.hpp"
#include "gpu/intel/jit/v2/ir/send.hpp"
#include "gpu/intel/jit/v2/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

struct gpu_primitive_t;

namespace compute {
class kernel_t;
}

namespace jit {

class kernel_info_t;

namespace v2 {
namespace conv {

struct hw_desc_t {
    ngen::HW hw;

    void stringify(std::ostream &out) const { jit::stringify(out, hw); }
    void parse(std::istream &in) { jit::parse(in, hw); }
#if __cplusplus >= 202002L
    bool operator==(const hw_desc_t &other) const = default;
#endif
};

enum class spec_strategy_t { none, max, one_d, two_d };

static auto spec_strategy_names = nstl::to_array({
        make_enum_name(spec_strategy_t::none, "none"),
        make_enum_name(spec_strategy_t::max, "max"),
        make_enum_name(spec_strategy_t::one_d, "1d"),
        make_enum_name(spec_strategy_t::two_d, "2d"),
});
GPU_DEFINE_PARSE_ENUM(spec_strategy_t, spec_strategy_names)

// The class spec_reqs_t represents specialization requirements for problem
// dimensions. It supports a strategy based mode where specialization can be
// tailored to a specific strategy. In the strategy mode, a call to
// specialize(problem_t) is required to finish generation.
class spec_reqs_t {
public:
    spec_reqs_t() = default;
    spec_reqs_t(const prb_tile_t &spec_tile) : spec_tile_(spec_tile) {}
    spec_reqs_t(spec_strategy_t spec_strategy)
        : spec_strategy_(spec_strategy) {}

#if __cplusplus >= 202002L
    bool operator==(const spec_reqs_t &other) const = default;
#endif

    void stringify(std::ostream &out) const {
        if (spec_strategy_ == spec_strategy_t::none && spec_tile_.is_empty()) {
            out << "x";
            return;
        }
        if (spec_strategy_ != spec_strategy_t::none) {
            jit::stringify(out, spec_strategy_);
        } else {
            jit::stringify(out, spec_tile_);
        }
    }

    void parse(std::istream &in) {
        operator=(spec_reqs_t());
        auto s = stream_parse<std::string>(in);
        if (s == "x") return;
        if (is_enum_name<spec_strategy_t>(s)) {
            jit::parse(s, spec_strategy_);
        } else {
            jit::parse(s, spec_tile_);
        }
    }

    bool is_equal(const prb_dim_t &dim, int value) const {
        return spec_tile_.has(dim) && spec_tile_.at(dim) == value;
    }

    expr_t to_expr(const prb_dim_t &dim) const {
        return spec_tile_.has(dim) ? spec_tile_.at(dim) : size_var(dim);
    }

    prb_reqs_t reqs() const {
        prb_reqs_t ret;
        for (auto &d : spec_tile_) {
            ret.add(size_var(d) == spec_tile_.at(d));
        }
        return ret;
    }

    void set(const prb_dim_t &dim, int value) {
        ir_assert(spec_strategy_ == spec_strategy_t::none);
        spec_tile_[dim] = value;
    }

    constraint_set_t as_constraint_set(const kernel_info_t &kernel_info) const {
        constraint_set_t ret;
        auto vars = kernel_info.get_vars();
        for (auto &d : spec_tile_) {
            auto v = vars.find(d.str());
            ir_assert(v != vars.end()) << "Could not find variable " << d.str();
            ret.add_constraint(v->second == spec_tile_.at(d));
        }
        return ret;
    }

    std::string str() const {
        if (spec_strategy_ == spec_strategy_t::none)
            return spec_tile_.str();
        else
            return "(" + to_string(spec_strategy_) + ")";
    }

    void specialize(const problem_t &prb) {
        if (spec_strategy_ == spec_strategy_t::none) return;
        switch (spec_strategy_) {
            case spec_strategy_t::max: spec_tile_ = prb.shape(); break;
            case spec_strategy_t::one_d:
                spec_tile_ = jit::parse<prb_tile_t>(
                        "id1ih1od1oh1kd1kh1dd0dh0pd0ph0sd1sh1");
                break;
            case spec_strategy_t::two_d:
                spec_tile_ = jit::parse<prb_tile_t>("id1od1kd1dd0pd0sd1");
                break;
            default: spec_tile_ = {}; break;
        }
        spec_strategy_ = spec_strategy_t::none;
        return;
    }

    bool has_strategy() const {
        return spec_strategy_ != spec_strategy_t::none;
    }

    IR_DEFINE_DUMP()

    using elements_t = const std::tuple<prb_tile_t &, spec_strategy_t &>;
    using const_elements_t
            = std::tuple<const prb_tile_t &, const spec_strategy_t &>;
    elements_t as_elements() { return {spec_tile_, spec_strategy_}; }
    const_elements_t as_elements() const {
        return {spec_tile_, spec_strategy_};
    }

private:
    prb_tile_t spec_tile_;
    spec_strategy_t spec_strategy_ = spec_strategy_t::none;
};

struct loop_desc_entry_t {
    prb_dim_t dim;
    int idx = -1;
    bool is_outer = true;
    // Whether the dimension range is distributed between thread groups (global
    // k-slicing).
    bool is_global = false;

    loop_desc_entry_t() = default;
    loop_desc_entry_t(const prb_dim_t &dim, int idx, bool is_global)
        : dim(dim), idx(idx), is_global(is_global) {}

    bool is_empty() const { return dim.is_undef(); }

    std::string str() const {
        std::ostringstream oss;
        oss << dim;
        return oss.str();
    }

    IR_DEFINE_DUMP()

#if __cplusplus >= 202002L
    bool operator==(const loop_desc_entry_t &other) const = default;
#endif
};

class loop_desc_t {
public:
    bool is_empty() const { return entries_.empty(); }
    const std::vector<loop_desc_entry_t> &entries() const { return entries_; }
    int ndims() const { return (int)entries_.size(); }
    bool has(const prb_dim_t &dim) const { return !find(dim).is_empty(); }
    loop_desc_entry_t find(const prb_dim_t &dim) const {
        for (auto &e : entries_)
            if (e.dim == dim) return e;
        return loop_desc_entry_t();
    }
    bool is_global(const prb_dim_t &dim) const { return find(dim).is_global; }
    void add(const prb_dim_t &dim, bool is_global = false) {
        if (!entries_.empty()) entries_.back().is_outer = false;
        entries_.emplace_back(dim, ndims(), is_global);
    }
    void remove(const prb_dim_t &dim) {
        for (auto it = entries_.begin(); it != entries_.end(); it++) {
            if (it->dim == dim) {
                entries_.erase(it);
                break;
            }
        }
        update_indices();
    }
    int index(const prb_dim_t &dim) const { return find(dim).idx; }
    std::vector<loop_desc_entry_t>::const_iterator begin() const {
        return entries_.begin();
    }
    std::vector<loop_desc_entry_t>::const_iterator end() const {
        return entries_.end();
    }

    std::string str() const {
        std::ostringstream oss;
        for (size_t i = 0; i < entries_.size(); i++) {
            if (i > 0) oss << ",";
            oss << entries_[i].dim;
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

#if __cplusplus >= 202002L
    bool operator==(const loop_desc_t &other) const = default;
#endif

    void stringify(std::ostream &out) const { out << str(); }

    void parse(std::istream &in) {
        entries_.clear();
        std::string s;
        in >> s;
        auto parts = gpu_utils::split(s, ",");
        for (auto &p : parts)
            add(prb_dim_t::from_name(p));
    }

private:
    void update_indices() {
        for (int i = 0; i < ndims(); i++) {
            entries_[i].idx = i;
        }
    }

    // Ordered from innermost to outermost.
    std::vector<loop_desc_entry_t> entries_;
};

struct load_desc_t {
    send_kind_t a = send_kind_t::undef;
    send_kind_t b = send_kind_t::undef;

    std::string str() const {
        std::vector<std::string> parts;
        if (a != send_kind_t::undef) parts.emplace_back("a:" + to_string(a));
        if (b != send_kind_t::undef) parts.emplace_back("b:" + to_string(b));
        if (parts.empty()) return "x";
        return gpu_utils::join(",", parts);
    }

    IR_DEFINE_DUMP()

#if __cplusplus >= 202002L
    bool operator==(const load_desc_t &other) const = default;
#endif

    void stringify(std::ostream &out) const { out << str(); }
    void parse(std::istream &in);
};

struct store_desc_t {
    send_kind_t c = send_kind_t::undef;

    std::string str() const {
        if (c != send_kind_t::undef) return "c:" + to_string(c);
        return "x";
    }

    IR_DEFINE_DUMP()

#if __cplusplus >= 202002L
    bool operator==(const store_desc_t &other) const = default;
#endif

    void stringify(std::ostream &out) const { out << str(); }
    void parse(std::istream &in);
};

struct prefetch_desc_t {
    int dist = 0;
    bool a = false;
    bool b = false;

    std::string str() const {
        if (!a && !b) return "x0";
        std::ostringstream oss;
        oss << "x" << dist;
        if (a && b) return oss.str();
        oss << "." << (a ? "a" : "b");
        return oss.str();
    }

    IR_DEFINE_DUMP()

#if __cplusplus >= 202002L
    bool operator==(const prefetch_desc_t &other) const = default;
#endif

    void stringify(std::ostream &out) const { out << str(); }
    void parse(std::istream &in);
};

layout_desc_t make_conv_layout_desc(
        tensor_kind_t tensor_kind, bool src_dst_with_group = false);
layout_desc_t make_conv_algo_layout_desc(
        prop_kind_t prop, tensor_kind_t tensor_kind);
layout_tag_t make_conv_layout_tag(
        tensor_kind_t tensor_kind, const std::string &s);
layout_tag_t make_conv_layout_tag(
        tensor_kind_t tensor_kind, int conv_ndims, const memory_desc_t &md);

struct plan_t;

class kernel_desc_t : public kernel_desc_base_t {
public:
    prop_kind_t prop = prop_kind::undef;
    bool is_dw = false;
    layout_tag_t src_tag;
    layout_tag_t wei_tag;
    layout_tag_t dst_tag;
    spec_reqs_t spec_reqs;
    hw_desc_t hw_desc;
    fma_kind_t fma = fma_kind_t::undef;
    int simd = 0;
    int regs = 0;
    prb_tile_t iter_tile;
    prb_tile_t thread_group_tile;
    loop_desc_t loop_desc;
    load_desc_t load;
    store_desc_t store;
    prefetch_desc_t prefetch;
    prb_reqs_t reqs;

    hw_t hw;
    bool is_finalized = false;

    bool is_empty() const { return prop == prop_kind::undef; }
    bool is_supported() const;
    void set(const std::string &s);
    void set_defaults();
    void finalize(const plan_t &plan);

    bool fits(const problem_t &prb, bool check_tags = true) const {
        ir_check(prb.prop() == prop) << "Propagation kind does not match";
        if (check_tags) {
            ir_check(prb.src_tag().matches(src_tag, prb.shape()))
                    << "Source tag  " << prb.src_tag()
                    << " does not match kernel descriptor tag " << src_tag;
            ir_check(prb.wei_tag().matches(wei_tag, prb.shape()))
                    << "Weights tag " << prb.wei_tag()
                    << " does not match kernel descriptor tag " << wei_tag;
            ir_check(prb.dst_tag().matches(dst_tag, prb.shape()))
                    << "Destination tag " << prb.dst_tag()
                    << " does not match kernel descriptor tag " << dst_tag;
        }
        ir_check(prb.is_depthwise() == is_dw)
                << "Mixing depthwise/non-depthwise descriptor and problem";
        ir_check(reqs.fits(prb.shape()));
        return true;
    }

    std::string cmd_str() const;
    std::string str() const;

    IR_DEFINE_DUMP()

    static void init_parse_iface(parse_iface_t<kernel_desc_t> *iface);

    // Helper methods.
    const type_t &a_type() const {
        return pick_a(prop, src_tag.type(), wei_tag.type(), dst_tag.type());
    }
    const type_t &b_type() const {
        return pick_b(prop, src_tag.type(), wei_tag.type(), dst_tag.type());
    }
    const type_t &c_type() const {
        return pick_c(prop, src_tag.type(), wei_tag.type(), dst_tag.type());
    }

    send_kind_t access_kind(send_op_t op, tensor_kind_t tensor) const {
        switch (op) {
            case send_op_t::load:
                switch (tensor) {
                    case tensor_kind_t::a: return load.a;
                    case tensor_kind_t::b: return load.b;
                    default: ir_error_not_expected();
                }
                break;
            case send_op_t::store:
                switch (tensor) {
                    case tensor_kind_t::c: return store.c;
                    default: ir_error_not_expected();
                }
                break;
            default: ir_error_not_expected();
        }
        return send_kind_t::undef;
    }

    std::string kernel_name() const override { return "gen_conv_v2"; }

    exec_config_t exec_cfg() const override {
        exec_config_t ret(hw);
        ret.set_regs(regs);
        ret.set_simd(simd);
        ret.set_vec_size(simd);
        return ret;
    }

    bool with_dpas() const override {
        return utils::one_of(fma, fma_kind_t::dpas, fma_kind_t::dpasw);
    }

    status_t init_kernel_info(kernel_info_t &kernel_info) const override;
    status_t create_kernel(compute::kernel_t &kernel,
            gpu_primitive_t *primitive, impl::engine_t *engine) const override;
    status_t create_generator(const compute::compute_engine_t &engine,
            compute::kernel_t &kernel) const;
    serialized_t serialize() const override;
    static kernel_desc_t deserialize(const serialized_t &s);
    static parse_iface_t<kernel_desc_t> cli_iface();
    static void show_help();
};

class grid_t {
public:
    static const int N = 3;

    grid_t() = default;
    grid_t(const std::string &prefix) {
        for (int i = 0; i < N; i++)
            entries_[i].idx_var
                    = var_t::make(type_t::s32(), prefix + std::to_string(i));
    }

    void add_mapping(const prb_dim_t &dim, int idx) {
        ir_assert(idx >= 0 && idx < N);
        ir_assert(index_var(dim).is_empty());
        entries_[idx].dims.push_back(dim);
    }

    void unset(const prb_dim_t &dim) {
        for (int i = 0; i < N; i++) {
            auto &dims = entries_[i].dims;
            for (auto it = dims.begin(); it != dims.end(); it++) {
                if (*it == dim) {
                    dims.erase(it);
                    return;
                }
            }
        }
    }

    int index(const prb_dim_t &dim) const {
        for (int i = 0; i < N; i++) {
            for (auto &d : entries_[i].dims) {
                if (d == dim) return i;
            }
        }
        return -1;
    }

    expr_t index_var(int idx) const {
        if (idx == -1) return expr_t();
        return entries_[idx].idx_var;
    }

    expr_t index_var(const prb_dim_t &dim) const {
        return index_var(index(dim));
    }

    const std::vector<prb_dim_t> &dims(int idx) const {
        ir_assert(idx >= 0 && idx < N);
        return entries_[idx].dims;
    }

    std::vector<prb_dim_t> all_dims() const {
        std::vector<prb_dim_t> ret;
        for (int i = 0; i < N; i++) {
            auto &e = entries_[i];
            ret.insert(ret.end(), e.dims.begin(), e.dims.end());
        }
        return ret;
    }

    size_t size(size_t idx, const prb_tile_t &tile) const {
        ir_assert(idx < N);
        size_t ret = 1;
        for (auto &d : entries_[idx].dims) {
            ret *= gpu_utils::into<size_t>(tile.get(d, 1));
        }
        return ret;
    }

private:
    struct entry_t {
        expr_t idx_var;
        std::vector<prb_dim_t> dims;

        int ndims() const { return (int)dims.size(); }
    };

    entry_t entries_[N];
};

grid_t create_thread_group_grid(const kernel_desc_t &desc);
grid_t create_thread_grid(const kernel_desc_t &desc);

class kernel_params_t : public kernel_params_base_t {
public:
    problem_t prb;

    status_t init_dispatch_kernel_info(kernel_info_t &kernel_info,
            const kernel_desc_base_t &_desc) const override;
};

} // namespace conv
} // namespace v2
} // namespace jit
#if __cplusplus >= 202002L
template <>
struct trivial_key_validator_t<jit::v2::conv::kernel_desc_t> {
    static bool is_valid(const jit::v2::conv::kernel_desc_t &t) {
        auto tmp = jit::v2::conv::kernel_desc_t::deserialize(t.serialize());
        return (t.prop == tmp.prop) && (t.is_dw == tmp.is_dw)
                && (t.src_tag == tmp.src_tag) && (t.wei_tag == tmp.wei_tag)
                && (t.dst_tag == tmp.dst_tag) && (t.spec_reqs == tmp.spec_reqs)
                && (t.hw == tmp.hw) && (t.fma == tmp.fma)
                && (t.simd == tmp.simd) && (t.regs == tmp.regs)
                && (t.iter_tile == tmp.iter_tile)
                && (t.thread_group_tile == tmp.thread_group_tile)
                && (t.loop_desc == tmp.loop_desc) && (t.load == tmp.load)
                && (t.prefetch == tmp.prefetch) && (t.store == tmp.store)
                && (t.is_finalized == tmp.is_finalized);
    }
};
#endif
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
