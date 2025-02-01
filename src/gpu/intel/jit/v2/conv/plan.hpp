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

#ifndef GPU_INTEL_JIT_V2_CONV_PLAN_HPP
#define GPU_INTEL_JIT_V2_CONV_PLAN_HPP

#include "gpu/intel/jit/v2/conv/kernel_desc.hpp"
#include "gpu/intel/jit/v2/conv/problem.hpp"
#include "gpu/intel/jit/v2/ir/plan.hpp"
#include "gpu/intel/jit/v2/ir/reqs.hpp"
#include "gpu/intel/jit/v2/ir/send.hpp"
#include "gpu/intel/jit/v2/ir/tensor.hpp"

#include <sstream>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

class coord_info_t {
public:
    void add_dim(const pvar_t &dim, bool is_loop, bool is_global_loop,
            dim_t tg_tile, const expr_t &thr_idx, dim_t iter_tile,
            const prb_reqs_t &reqs) {
        auto &e = entries_[dim];
        e.dim = dim;
        e.tg_size = tg_tile;
        e.iter_size = iter_tile;
        e.loop_idx = expr_t(0);
        e.loop_size = expr_t(1);
        bool is_dim_1 = reqs.is_equal(dim, 1);
        if (is_loop && !is_dim_1) {
            if (is_global_loop) {
                e.loop_size = const_var_t::make(
                        type_t::s32(), e.dim.str() + "_loop_size");
                e.is_global_loop = true;
            } else {
                e.loop_size = div_up(reqs.to_expr(e.dim), tg_tile * iter_tile);
            }
            e.loop_idx = is_one(e.loop_size)
                    ? expr_t(0)
                    : var_t::make(type_t::s32(), e.dim.str() + "_loop_idx");
        }
        e.tg_idx = expr_t(0);
        e.thr_idx = (tg_tile == 1 ? expr_t(0) : thr_idx);
        if (!is_dim_1 && (!is_loop || is_global_loop)) {
            e.tg_idx = var_t::make(type_t::s32(), dim.str() + "_tg_idx");
        }
        auto iter_idx = e.tg_idx;
        iter_idx = iter_idx * e.tg_size + e.thr_idx;
        iter_idx = iter_idx * e.loop_size + e.loop_idx;
        iter_idx = simplify_rewrite(iter_idx);

        e.tg_idx = simplify_rewrite(e.tg_idx);
        e.thr_idx = simplify_rewrite(e.thr_idx);
        e.iter_idx = simplify_rewrite(iter_idx);
        e.loop_idx = simplify_rewrite(e.loop_idx);
        e.loop_size = simplify_rewrite(e.loop_size);
    }

    std::vector<pvar_t> dims() const { return entries_.keys(); }

    bool is_loop(const pvar_t &dim) const { return entries_.at(dim).is_loop(); }
    bool is_global_loop(const pvar_t &dim) const {
        return entries_.at(dim).is_global_loop;
    }
    const expr_t &tg_index(const pvar_t &dim) const {
        return entries_.at(dim).tg_idx;
    }
    const expr_t &thr_index(const pvar_t &dim) const {
        return entries_.at(dim).thr_idx;
    }
    const expr_t &iter_index(const pvar_t &dim) const {
        return entries_.at(dim).iter_idx;
    }
    const expr_t &loop_size(const pvar_t &dim) const {
        return entries_.at(dim).loop_size;
    }
    const expr_t &loop_index(const pvar_t &dim) const {
        return entries_.at(dim).loop_idx;
    }

    pvar_coord_t<expr_t> iter_coord() const;
    pvar_coord_t<expr_t> tg_iter_coord() const;
    pvar_tile_t tg_iter_tile() const;

    std::string str() const {
        std::ostringstream oss;
        bool is_first = true;
        for (auto &d : entries_) {
            if (!is_first) oss << std::endl;
            auto &e = entries_[d];
            oss << e;
            is_first = false;
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    struct entry_t {
        pvar_t dim;
        expr_t tg_idx;
        expr_t thr_idx;
        expr_t iter_idx;
        expr_t loop_idx;

        dim_t tg_size = 0;
        dim_t iter_size = 0;
        expr_t loop_size;
        bool is_global_loop = false;

        bool is_loop() const { return !is_one(loop_size); }

        std::string str() const {
            std::ostringstream oss;
            oss << "tg_idx:   " << tg_idx << std::endl;
            oss << "thr_idx:  " << thr_idx << std::endl;
            oss << "loop_idx: " << loop_idx;
            return ir_utils::add_tag(dim.str(), oss.str());
        }

        IR_DEFINE_DUMP()
    };

    pvar_map_t<entry_t> entries_;
};

class virt_grid_t {
public:
    void add(const expr_t &var, const expr_t &expr) {
        auto ret = idxs_.emplace(var, expr);
        gpu_assert(ret.second);
    }

    const object_map_t<expr_t, expr_t> &idxs() const { return idxs_; }

private:
    object_map_t<expr_t, expr_t> idxs_;
};

struct prefetch_plan_t : public base_plan_t {
    send_plan_t a_prefetch;
    send_plan_t b_prefetch;

    using base_plan_t::base_plan_t;

    int grf_usage_bytes() const {
        int ret = 0;
        ret += a_prefetch.grf_usage_bytes();
        ret += b_prefetch.grf_usage_bytes();
        return ret;
    }

    prb_reqs_t reqs() const {
        prb_reqs_t ret;
        ret.add(a_prefetch.reqs());
        ret.add(b_prefetch.reqs());
        ret.simplify();
        return ret;
    }

    std::string str() const {
        if (!*this || (!a_prefetch && !b_prefetch)) return "(empty)";
        std::ostringstream oss;
        oss << ir_utils::add_tag("a_prefetch", a_prefetch.str()) << std::endl;
        oss << ir_utils::add_tag("b_prefetch", b_prefetch.str());
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

struct x2r_plan_t : public base_plan_t {
    tensor_kind_t tensor_kind = tensor_kind_t::undef;
    send_plan_t load;
    reorder_plan_t reorder;
    layout_t layout;
    layout_t bias_layout;

    using base_plan_t::base_plan_t;

    int grf_usage_bytes() const {
        int ret = 0;
        ret += load.grf_usage_bytes();
        ret += reorder.grf_usage_bytes();
        return ret;
    }

    prb_reqs_t reqs() const {
        prb_reqs_t ret;
        ret.add(load.reqs());
        ret.simplify();
        return ret;
    }

    std::string str() const {
        if (!*this) return "(empty)";
        std::ostringstream oss;
        auto prefix = to_string(tensor_kind);
        oss << prefix << "_layout: " << layout.str() << std::endl;
        oss << ir_utils::add_tag(prefix + "_load", load.str()) << std::endl;
        if (reorder)
            oss << ir_utils::add_tag(prefix + "_reorder", reorder.str())
                << std::endl;
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

struct fma_plan_t : public base_plan_t {
    layout_t a_layout;
    layout_t b_layout;
    layout_t c_layout;
    pvar_tile_t inst_tile;
    fma_kind_t fma = fma_kind_t::undef;
    int simd = 0;

    using base_plan_t::base_plan_t;

    int grf_usage_bytes() const {
        int ret = 0;
        ret += utils::rnd_up(c_layout.size(), grf_size());
        return ret;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "a_layout: " << a_layout.str_with_size(hw) << std::endl;
        oss << "b_layout: " << b_layout.str_with_size(hw) << std::endl;
        oss << "c_layout: " << c_layout.str_with_size(hw) << std::endl;
        oss << "fma: " << to_string(fma) << std::endl;
        oss << "simd: " << simd << std::endl;
        oss << "inst_tile: " << inst_tile;
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

struct x2r_fma_plan_t : public base_plan_t {
    struct stage_t {
        x2r_plan_t x2r;
        fma_plan_t fma;

        stage_t() = default;
        stage_t(const x2r_plan_t &x2r) : x2r(x2r) {}
        stage_t(const fma_plan_t &fma) : fma(fma) {}
        bool is_x2r() const { return (bool)x2r; }
        bool is_fma() const { return (bool)fma; }

        prb_reqs_t reqs() const {
            if (is_x2r()) return x2r.reqs();
            return prb_reqs_t();
        }

        std::string str() const {
            if (is_x2r()) return x2r.str();
            return fma.str();
        }
    };

    pvar_tile_t outer;
    layout_t c_layout;
    layout_t bias_layout;
    std::vector<stage_t> stages;

    x2r_fma_plan_t(const hw_t &hw) : base_plan_t(hw) {}
    void add_stage(const x2r_plan_t &x2r) { stages.emplace_back(x2r); }
    void add_stage(const fma_plan_t &fma) { stages.emplace_back(fma); }
    int nstages() const { return (int)stages.size(); }

    int grf_usage_bytes() const {
        int ret = 0;
        ret += utils::rnd_up(c_layout.size(), grf_size());
        std::unordered_map<tensor_kind_t, int,
                ir_utils::enum_hash_t<tensor_kind_t>>
                tensor_usage;
        for (auto &s : stages) {
            if (!s.is_x2r()) continue;
            auto &usage = tensor_usage[s.x2r.tensor_kind];
            usage = std::max(usage, s.x2r.grf_usage_bytes());
        }
        for (auto &kv : tensor_usage)
            ret += kv.second;
        return ret;
    }

    prb_reqs_t reqs() const {
        prb_reqs_t ret;
        for (auto &s : stages)
            ret.add(s.reqs());
        ret.simplify();
        return ret;
    }

    std::string str() const {
        if (!*this) return "(empty)";
        std::ostringstream oss;
        if (!outer.is_empty()) oss << "outer: " << outer << std::endl;
        for (int i = 0; i < nstages(); i++) {
            auto &s = stages[i];
            auto tag = (s.is_x2r() ? "x2r_" : "fma_") + std::to_string(i);
            oss << ir_utils::add_tag(tag, s.str()) << std::endl;
        }
        oss << "c_layout: " << c_layout;
        return oss.str();
    }
};

struct slm_reduce_plan_t : public base_plan_t {
    send_plan_t store;
    send_plan_t load;
    reduce_plan_t reduce;
    // C layout and tile coordinate after reduction and redistribution in
    // threadgroup.
    layout_t c_layout;
    pvar_coord_t<expr_t> c_coord;

    using base_plan_t::base_plan_t;

    int grf_usage_bytes() const {
        int ret = 0;
        ret += utils::rnd_up(load.reg_layout().size(), grf_size());
        return ret;
    }

    int slm_usage_bytes() const {
        if (!*this) return 0;
        int k_local
                = ir_utils::safe_div(reduce.src.elems(), reduce.dst.elems());
        return utils::rnd_up(store.reg_layout().size(), grf_size()) * k_local;
    }

    std::string str() const {
        if (!*this) return "(empty)";
        std::ostringstream oss;
        oss << ir_utils::add_tag("store", store.str()) << std::endl;
        oss << ir_utils::add_tag("load", load.str()) << std::endl;
        oss << ir_utils::add_tag("reduce", reduce.str()) << std::endl;
        oss << "c_layout: " << c_layout << std::endl;
        oss << "c_coord:  " << c_coord;
        return oss.str();
    }
};

struct epilogue_store_plan_t : public base_plan_t {
    pvar_tile_t tile;
    reorder_plan_t reorder;
    reorder_plan_t bias_reorder;
    send_plan_t c_store;
    send_plan_t bias_store;

    std::string str() const {
        if (!*this) return "(empty)";
        std::ostringstream oss;
        oss << "tile: " << tile << std::endl;
        if (reorder)
            oss << ir_utils::add_tag("reorder", bias_reorder.str())
                << std::endl;
        if (bias_reorder)
            oss << ir_utils::add_tag("bias_reorder", bias_reorder.str())
                << std::endl;
        oss << ir_utils::add_tag("c_store", c_store.str()) << std::endl;
        if (bias_store)
            oss << ir_utils::add_tag("bias_store", bias_store.str());
        return oss.str();
    }
};

struct epilogue_plan_t : public base_plan_t {
    slm_reduce_plan_t slm_reduce;
    layout_t c_reg_layout;
    pvar_coord_t<expr_t> c_coord;
    layout_t bias_layout;
    expr_t bias_reduce_cond;

    epilogue_store_plan_t store;

    using base_plan_t::base_plan_t;

    int grf_usage_bytes() const { return 0; }
    int slm_usage_bytes() const { return slm_reduce.slm_usage_bytes(); }

    std::string str() const {
        if (!*this) return "(empty)";
        std::ostringstream oss;
        if (slm_reduce)
            oss << ir_utils::add_tag("slm_reduce", slm_reduce.str())
                << std::endl;
        if (store) oss << ir_utils::add_tag("store", store.str()) << std::endl;
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

struct plan_t : public base_plan_t {
    kernel_desc_t desc;
    coord_info_t coord_info;
    grid_t tg_grid;
    grid_t thr_grid;
    virt_grid_t virt_grid;

    prefetch_plan_t prefetch;
    x2r_fma_plan_t x2r_fma;
    epilogue_plan_t epilogue;

    plan_t(const hw_t &hw = hw_t())
        : base_plan_t(hw), prefetch(hw), x2r_fma(hw), epilogue(hw) {}

    int grf_usage_bytes() const {
        int ret = 0;
        ret += x2r_fma.grf_usage_bytes();
        ret += epilogue.grf_usage_bytes();
        return ret;
    }

    int slm_usage_bytes() const {
        int ret = 0;
        ret += epilogue.slm_usage_bytes();
        return ret;
    }

    std::string str() const {
        if (!*this) return "(empty)";
        std::ostringstream oss;
        oss << ir_utils::add_tag("prefetch", prefetch.str()) << std::endl;
        oss << ir_utils::add_tag("x2r_fma", x2r_fma.str()) << std::endl;
        oss << ir_utils::add_tag("epilogue", epilogue.str());
        return ir_utils::add_tag("Plan", oss.str());
    }

    IR_DEFINE_DUMP()
};

plan_t create_conv_plan(const kernel_desc_t &desc, const hw_t &hw);
plan_t create_conv_plan(const kernel_desc_t &desc, const problem_t &prb);
prb_reqs_t generate_reqs(const kernel_desc_t &desc);

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
