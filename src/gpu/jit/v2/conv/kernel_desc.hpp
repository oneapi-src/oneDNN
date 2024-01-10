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

#ifndef GPU_JIT_V2_CONV_PLAN_DESC_HPP
#define GPU_JIT_V2_CONV_PLAN_DESC_HPP

#include "gpu/jit/ir/fma.hpp"
#include "gpu/jit/ir/kernel_desc.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/v2/conv/problem.hpp"
#include "gpu/jit/v2/ir/reqs.hpp"
#include "gpu/jit/v2/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

struct gpu_primitive_t;

namespace compute {
class kernel_t;
}

namespace jit {

class kernel_info_t;

namespace v2 {
namespace conv {

struct loop_nest_entry_t {
    prb_dim_t dim;
    int idx = -1;
    bool is_outer = true;
    // Whether the dimension range is distributed between thread groups (global
    // k-slicing).
    bool is_global = false;

    loop_nest_entry_t() = default;
    loop_nest_entry_t(const prb_dim_t &dim, int idx, bool is_global)
        : dim(dim), idx(idx), is_global(is_global) {}

    bool is_empty() const { return dim.is_undef(); }

    std::string str() const {
        std::ostringstream oss;
        oss << dim;
        return oss.str();
    }

    IR_DEFINE_DUMP()

    bool operator==(const loop_nest_entry_t &other) const {
        return (dim == other.dim) && (idx == other.idx)
                && (is_outer == other.is_outer)
                && (is_global == other.is_global);
    }

    bool operator!=(const loop_nest_entry_t &other) const {
        return !operator==(other);
    }

    size_t get_hash() const {
        return ir_utils::get_hash(dim, idx, is_outer, is_global);
    }

    void serialize(std::ostream &out) const {
        ir_utils::serialize(dim, out);
        ir_utils::serialize(idx, out);
        ir_utils::serialize(is_outer, out);
        ir_utils::serialize(is_global, out);
    }

    void deserialize(std::istream &in) {
        ir_utils::deserialize(dim, in);
        ir_utils::deserialize(idx, in);
        ir_utils::deserialize(is_outer, in);
        ir_utils::deserialize(is_global, in);
    }
};

class loop_nest_t {
public:
    bool is_empty() const { return entries_.empty(); }
    const std::vector<loop_nest_entry_t> &entries() const { return entries_; }
    int ndims() const { return (int)entries_.size(); }
    bool has(const prb_dim_t &dim) const { return !find(dim).is_empty(); }
    loop_nest_entry_t find(const prb_dim_t &dim) const {
        for (auto &e : entries_)
            if (e.dim == dim) return e;
        return loop_nest_entry_t();
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
    std::vector<loop_nest_entry_t>::const_iterator begin() const {
        return entries_.begin();
    }
    std::vector<loop_nest_entry_t>::const_iterator end() const {
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

    bool operator==(const loop_nest_t &other) const {
        return entries_ == other.entries_;
    }

    bool operator!=(const loop_nest_t &other) const {
        return !operator==(other);
    }

    size_t get_hash() const { return ir_utils::get_hash(entries_); }

    void serialize(std::ostream &out) const {
        ir_utils::serialize(entries_, out);
    }
    void deserialize(std::istream &in) { ir_utils::deserialize(entries_, in); }

private:
    void update_indices() {
        for (int i = 0; i < ndims(); i++) {
            entries_[i].idx = i;
        }
    }

    // Ordered from innermost to outermost.
    std::vector<loop_nest_entry_t> entries_;
};

inline loop_nest_t str_to_loop_nest(const std::string &s) {
    auto parts = gpu_utils::split(s, ",");
    loop_nest_t ret;
    for (auto &p : parts)
        ret.add(prb_dim_t::from_name(p));
    return ret;
}

layout_desc_t make_conv_layout_desc(
        tensor_kind_t tensor_kind, bool src_dst_with_group = false);
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
    hw_t hw;
    fma_kind_t fma = fma_kind_t::undef;
    int simd = 0;
    int regs = 0;
    prb_tile_t iter_tile;
    prb_tile_t thread_group_tile;
    loop_nest_t loop_nest;
    send_kind_t a_access_kind = send_kind_t::undef;
    send_kind_t b_access_kind = send_kind_t::undef;
    send_kind_t c_access_kind = send_kind_t::undef;
    prb_reqs_t reqs;
    bool is_finalized = false;

    bool is_empty() const { return prop == prop_kind::undef; }
    bool is_supported() const;
    void set(const std::string &s);
    void set_defaults();
    void finalize(const plan_t &plan);

    bool fits(const problem_t &prb, bool check_tags = true) const {
        if (prb.prop() != prop) return false;
        if (check_tags) {
            if (!prb.src_tag().matches(src_tag, prb.shape())) return false;
            if (!prb.wei_tag().matches(wei_tag, prb.shape())) return false;
            if (!prb.dst_tag().matches(dst_tag, prb.shape())) return false;
        }
        if (prb.is_depthwise() != is_dw) return false;
        if (!reqs.fits(prb.shape())) return false;
        return true;
    }

    std::string cmd_str() const;
    std::string str() const;

    IR_DEFINE_DUMP()

    bool operator==(const kernel_desc_t &other) const {
        return (prop == other.prop) && (is_dw == other.is_dw)
                && (src_tag == other.src_tag) && (wei_tag == other.wei_tag)
                && (dst_tag == other.dst_tag) && (hw == other.hw)
                && (fma == other.fma) && (simd == other.simd)
                && (regs == other.regs) && (iter_tile == other.iter_tile)
                && (thread_group_tile == other.thread_group_tile)
                && (loop_nest == other.loop_nest)
                && (a_access_kind == other.a_access_kind)
                && (b_access_kind == other.b_access_kind)
                && (c_access_kind == other.c_access_kind)
                && (is_finalized == other.is_finalized);
    }

    bool operator!=(const kernel_desc_t &other) const {
        return !operator==(other);
    }

    size_t get_hash() const {
        return ir_utils::get_hash(prop, is_dw, src_tag, wei_tag, dst_tag, hw,
                fma, simd, regs, iter_tile, thread_group_tile, loop_nest,
                a_access_kind, b_access_kind, c_access_kind, is_finalized);
    }

    void serialize(std::ostream &out) const {
        ir_assert(is_finalized);
        ir_utils::serialize(prop, out);
        ir_utils::serialize(is_dw, out);
        ir_utils::serialize(src_tag, out);
        ir_utils::serialize(wei_tag, out);
        ir_utils::serialize(dst_tag, out);
        ir_utils::serialize(hw, out);
        ir_utils::serialize(fma, out);
        ir_utils::serialize(simd, out);
        ir_utils::serialize(regs, out);
        ir_utils::serialize(iter_tile, out);
        ir_utils::serialize(thread_group_tile, out);
        ir_utils::serialize(loop_nest, out);
        ir_utils::serialize(a_access_kind, out);
        ir_utils::serialize(b_access_kind, out);
        ir_utils::serialize(c_access_kind, out);
        ir_utils::serialize(reqs, out);
    }

    void deserialize(std::istream &in) {
        ir_utils::deserialize(prop, in);
        ir_utils::deserialize(is_dw, in);
        ir_utils::deserialize(src_tag, in);
        ir_utils::deserialize(wei_tag, in);
        ir_utils::deserialize(dst_tag, in);
        ir_utils::deserialize(hw, in);
        ir_utils::deserialize(fma, in);
        ir_utils::deserialize(simd, in);
        ir_utils::deserialize(regs, in);
        ir_utils::deserialize(iter_tile, in);
        ir_utils::deserialize(thread_group_tile, in);
        ir_utils::deserialize(loop_nest, in);
        ir_utils::deserialize(a_access_kind, in);
        ir_utils::deserialize(b_access_kind, in);
        ir_utils::deserialize(c_access_kind, in);
        ir_utils::deserialize(reqs, in);
        is_finalized = true;
    }

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

    send_kind_t access_kind(tensor_kind_t tensor) const {
        if (tensor == tensor_kind_t::a) return a_access_kind;
        if (tensor == tensor_kind_t::b) return b_access_kind;
        if (tensor == tensor_kind_t::c) return c_access_kind;
        ir_error_not_expected();
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
            gpu_primitive_t *primitive, engine_t *engine) const override;
    status_t create_generator(const compute::compute_engine_t &engine,
            compute::kernel_t &kernel) const;
    serialized_t serialize() const override;
    static kernel_desc_t deserialize(const serialized_t &s);
    static ir_utils::cli_iface_t<kernel_desc_t> cli_iface();
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

    int size(int idx, const prb_tile_t &tile) const {
        ir_assert(idx >= 0 && idx < N);
        int ret = 1;
        for (auto &d : entries_[idx].dims) {
            ret *= tile.get(d, 1);
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
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
