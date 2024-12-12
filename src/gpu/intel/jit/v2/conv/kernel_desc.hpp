/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include "gpu/intel/gpu_post_ops.hpp"
#include "gpu/intel/jit/ir/fma.hpp"
#include "gpu/intel/jit/ir/hw.hpp"
#include "gpu/intel/jit/ir/kernel_desc.hpp"
#include "gpu/intel/jit/ir/message.hpp"
#include "gpu/intel/jit/ir/primitive_plan.hpp"
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
    ngen::HW hw = ngen::HW::Unknown;

    void stringify(std::ostream &out) const { jit::stringify(out, hw); }
    void parse(std::istream &in) { jit::parse(in, hw); }
#if __cplusplus >= 202002L
    bool operator==(const hw_desc_t &other) const = default;
#endif
};

// Represents specialization requirements for problem dimensions. A call to
// desc.specialize(problem_t) is required to finish generation.
enum class spec_strategy_t { none, max, min_dims };

static auto spec_strategy_names = nstl::to_array({
        make_enum_name(spec_strategy_t::none, "none"),
        make_enum_name(spec_strategy_t::max, "max"),
        make_enum_name(spec_strategy_t::min_dims, "min_dims"),
});
GPU_DEFINE_PARSE_ENUM(spec_strategy_t, spec_strategy_names)

struct loop_desc_entry_t {
    pvar_t dim;
    int idx = -1;
    bool is_outer = true;
    // Whether the dimension range is distributed between thread groups (global
    // k-slicing).
    bool is_global = false;

    loop_desc_entry_t() = default;
    loop_desc_entry_t(const pvar_t &dim, int idx, bool is_global)
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
    bool has(const pvar_t &dim) const { return !find(dim).is_empty(); }
    loop_desc_entry_t find(const pvar_t &dim) const {
        for (auto &e : entries_)
            if (e.dim == dim) return e;
        return loop_desc_entry_t();
    }
    bool is_global(const pvar_t &dim) const { return find(dim).is_global; }
    void add(const pvar_t &dim, bool is_global = false) {
        if (!entries_.empty()) entries_.back().is_outer = false;
        entries_.emplace_back(dim, ndims(), is_global);
    }
    void remove(const pvar_t &dim) {
        for (auto it = entries_.begin(); it != entries_.end(); it++) {
            if (it->dim == dim) {
                entries_.erase(it);
                break;
            }
        }
        update_indices();
    }
    int index(const pvar_t &dim) const { return find(dim).idx; }
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
            add(pvar_t(p));
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

struct align_desc_t {
    struct align_t {
        int value = 1;
        // If true, then value in bytes, otherwise in elements.
        bool in_bytes = false;

#if __cplusplus >= 202002L
        bool operator==(const align_t &other) const = default;
#endif

        bool is_default() const { return value == 1; }
        std::string str() const;
        void parse(const std::string &s);
    };
    align_t src;
    align_t wei;
    align_t dst;

    bool is_default() const {
        return src.is_default() && wei.is_default() && dst.is_default();
    }

    std::string str() const;

    IR_DEFINE_DUMP()

#if __cplusplus >= 202002L
    bool operator==(const align_desc_t &other) const = default;
#endif

    void stringify(std::ostream &out) const { out << str(); }
    void parse(std::istream &in);
};

struct prefetch_desc_t {
    int dist = 0;
    bool a = false;
    bool b = false;

    prefetch_desc_t() = default;
    prefetch_desc_t(int dist, bool a, bool b) : dist(dist), a(a), b(b) {}

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

enum class extension_kind_t : uint32_t {
    undef = 0,
    out_b1 = 1,
    out_b2 = 2,
    out_b4 = 4,
    bias = 8,
};

static auto extension_kind_names = nstl::to_array({
        make_enum_name(extension_kind_t::undef, "undef"),
        make_enum_name(extension_kind_t::out_b1, "out_b1"),
        make_enum_name(extension_kind_t::out_b2, "out_b2"),
        make_enum_name(extension_kind_t::out_b4, "out_b4"),
        make_enum_name(extension_kind_t::bias, "bias"),
});
GPU_DEFINE_PARSE_ENUM(extension_kind_t, extension_kind_names)

struct extensions_t {
    extension_kind_t kinds = extension_kind_t::undef;

    void add(extension_kind_t kind);
    bool has(extension_kind_t kind) const;
    std::string str() const;
    IR_DEFINE_DUMP()
    void stringify(std::ostream &out) const { out << str(); }
    void parse(std::istream &in);

    static extension_kind_t out_size(int size);
};

pvar_tile_t min_dims_tile(const problem_t &prb);

struct plan_t;
class grid_t;

class kernel_desc_t : public kernel_desc_base_t {
public:
    prop_kind_t prop = prop_kind::undef;
    bool is_dw = false;
    layout_tag_t src_tag;
    layout_tag_t wei_tag;
    layout_tag_t dst_tag;
    type_t bias_type;
    spec_strategy_t spec_strategy = spec_strategy_t::none;
    hw_desc_t hw_desc;
    fma_kind_t fma = fma_kind_t::undef;
    int simd = 0;
    int regs = 0;
    pvar_tile_t iter_tile;
    pvar_tile_t iter_outer_tile;
    pvar_tile_t thread_group_tile;
    loop_desc_t loop_desc;

    bool use_stream_k = false;
    bool use_2d_access = false;
    align_desc_t align;

    prefetch_desc_t prefetch;
    prb_reqs_t reqs;
    extensions_t ext;
    gpu_post_ops_t post_ops;

    hw_t hw;
    bool is_finalized = false;

    bool is_empty() const { return prop == prop_kind::undef; }
    bool is_supported() const;
    void set(const std::string &s);
    void set_defaults();
    void finalize(const prb_reqs_t &final_reqs);
    bool can_fit(const problem_t &prb) const;
    void fit_to(const problem_t &prb);
    status_t set_post_ops(const post_ops_t &post_ops,
            const memory_desc_t *out_md, const convolution_pd_t *pd);
    bool matches(const problem_t &prb) const;
    std::string cmd_str() const;
    std::string brief_str() const;
    std::string str() const;

    IR_DEFINE_DUMP()

    static const parse_iface_t<kernel_desc_t> &parse_iface();
    static void init_parse_iface(parse_iface_t<kernel_desc_t> *iface);

    // Helper methods.
    const layout_tag_t &layout_tag(tensor_kind_t kind) const {
        switch (kind) {
            case tensor_kind_t::a:
                return pick_a(prop, src_tag, wei_tag, dst_tag);
            case tensor_kind_t::b:
                return pick_b(prop, src_tag, wei_tag, dst_tag);
            case tensor_kind_t::c:
                return pick_c(prop, src_tag, wei_tag, dst_tag);
            default: ir_error_not_expected();
        }
        return src_tag;
    }
    const type_t &a_type() const { return layout_tag(tensor_kind_t::a).type(); }
    const type_t &b_type() const { return layout_tag(tensor_kind_t::b).type(); }
    const type_t &c_type() const { return layout_tag(tensor_kind_t::c).type(); }
    bool with_bias_fwd() const {
        return prop == prop_kind::forward && !bias_type.is_undef();
    }
    bool with_bias_bwd_w() const {
        return prop == prop_kind::backward_weights && !bias_type.is_undef();
    }

    send_kind_t access_kind(send_op_t op, tensor_kind_t tensor) const;
    std::string kernel_name() const override { return "gen_conv_v2"; }

    exec_config_t exec_cfg(const impl::engine_t *engine) const override {
        return exec_config_t(hw_t(engine), regs, simd);
    }

    compute::range_t local_range() const override;

    bool with_dpas() const override {
        return utils::one_of(fma, fma_kind_t::dpas, fma_kind_t::dpasw);
    }

    bool has_spec_strategy() const {
        return spec_strategy != spec_strategy_t::none;
    }

    void specialize(const problem_t &prb) {
        if (!has_spec_strategy()) return;
        switch (spec_strategy) {
            case spec_strategy_t::max:
                reqs.add(prb.shape());
                reqs.simplify();
                break;
            case spec_strategy_t::min_dims:
                reqs.add(min_dims_tile(prb));
                reqs.simplify();
                break;
            case spec_strategy_t::none: break;
            default: ir_error_not_expected();
        }
        spec_strategy = spec_strategy_t::none;
    }

    void init_kernel_iface(kernel_iface_t &kernel_iface) const override;
    void init_kernel_info(kernel_info_t &kernel_info,
            const kernel_params_base_t &params,
            const impl::engine_t *engine) const override;
    status_t create_kernel(compute::kernel_t &kernel,
            gpu_primitive_t *primitive, impl::engine_t *engine) const override;
    status_t create_generator(const compute::compute_engine_t &engine,
            compute::kernel_t &kernel) const;
    status_t init_primitive_plan(primitive_init_plan_t &plan,
            const problem_t &prb, convolution_pd_t *pd) const;
    serialized_t serialize() const override;
    static kernel_desc_t deserialize(const serialized_t &s);
    static void show_help();
};

class arg_helper_t {
public:
    arg_helper_t(const kernel_desc_t &desc);
    int key(const std::string &name) const;
    bool is_input(const std::string &name) const;
    bool is_output(const std::string &name) const;
    std::string post_op_name(size_t idx) const;
    int post_op_key(size_t idx) const;

private:
    bool is_fwd() const { return desc_.prop == prop_kind::forward; }
    bool is_bwd_d() const { return desc_.prop == prop_kind::backward_data; }
    bool is_bwd_w() const { return desc_.prop == prop_kind::backward_weights; }

    const kernel_desc_t &desc_;
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
    grid_t(const std::array<expr_t, N> &idx_vars) {
        for (int i = 0; i < N; i++) {
            entries_[i].idx_var = idx_vars[i];
        }
    }

    void add_mapping(const pvar_t &dim, int idx) {
        ir_assert(idx >= 0 && idx < N);
        ir_assert(index_var(dim).is_empty());
        entries_[idx].dims.push_back(dim);
    }

    void unset(const pvar_t &dim) {
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

    int index(const pvar_t &dim) const {
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

    expr_t index_var(const pvar_t &dim) const { return index_var(index(dim)); }

    const std::vector<pvar_t> &dims(int idx) const {
        ir_assert(idx >= 0 && idx < N);
        return entries_[idx].dims;
    }

    std::vector<pvar_t> all_dims() const {
        std::vector<pvar_t> ret;
        for (int i = 0; i < N; i++) {
            auto &e = entries_[i];
            ret.insert(ret.end(), e.dims.begin(), e.dims.end());
        }
        return ret;
    }

    size_t size(size_t idx, const pvar_tile_t &tile) const {
        ir_assert(idx < N);
        size_t ret = 1;
        for (auto &d : entries_[idx].dims) {
            ret *= into<size_t>(tile.get(d, 1));
        }
        return ret;
    }

private:
    struct entry_t {
        expr_t idx_var;
        std::vector<pvar_t> dims;

        int ndims() const { return (int)dims.size(); }
    };

    entry_t entries_[N];
};

grid_t create_thread_group_grid(const kernel_desc_t &desc);
grid_t create_thread_grid(const kernel_desc_t &desc);

class kernel_params_t : public kernel_params_base_t {
public:
    problem_t prb;
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
                && (t.dst_tag == tmp.dst_tag)
                && (t.spec_strategy == tmp.spec_strategy)
                && (t.hw_desc == tmp.hw_desc) && (t.fma == tmp.fma)
                && (t.simd == tmp.simd) && (t.regs == tmp.regs)
                && (t.iter_tile == tmp.iter_tile)
                && (t.thread_group_tile == tmp.thread_group_tile)
                && (t.loop_desc == tmp.loop_desc)
                && (t.prefetch == tmp.prefetch) && (t.align == tmp.align)
                && (t.use_stream_k == tmp.use_stream_k)
                && (t.use_2d_access == tmp.use_2d_access)
                && (t.is_finalized == tmp.is_finalized);
    }
};
#endif
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
