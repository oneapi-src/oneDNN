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

#ifndef GPU_INTEL_JIT_V2_CONV_PLAN_HPP
#define GPU_INTEL_JIT_V2_CONV_PLAN_HPP

#include "gpu/intel/jit/v2/conv/kernel_desc.hpp"
#include "gpu/intel/jit/v2/conv/problem.hpp"
#include "gpu/intel/jit/v2/ir/plan_utils.hpp"
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
    void add_dim(const prb_dim_t &dim, bool is_loop, bool is_global_loop,
            int tg_tile, const expr_t &thr_idx, int iter_tile,
            const spec_reqs_t &spec_reqs) {
        auto &e = entries_[dim];
        e.dim = dim;
        e.tg_size = tg_tile;
        e.iter_size = iter_tile;
        e.loop_idx = expr_t(0);
        e.loop_size = expr_t(1);
        bool is_dim_1 = spec_reqs.is_equal(dim, 1);
        if (is_loop && !is_dim_1) {
            e.loop_idx = var_t::make(type_t::s32(), e.dim.str() + "_loop_idx");
            if (is_global_loop) {
                e.loop_size = const_var_t::make(
                        type_t::s32(), e.dim.str() + "_loop_size");
                e.is_global_loop = true;
            } else {
                e.loop_size = binary_op_t::make(op_kind_t::_div_up,
                        size_var(e.dim), tg_tile * iter_tile);
            }
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

    std::vector<prb_dim_t> dims() const { return entries_.keys(); }

    bool is_loop(const prb_dim_t &dim) const {
        return entries_.at(dim).is_loop();
    }
    bool is_global_loop(const prb_dim_t &dim) const {
        return entries_.at(dim).is_global_loop;
    }
    const expr_t &tg_index(const prb_dim_t &dim) const {
        return entries_.at(dim).tg_idx;
    }
    const expr_t &thr_index(const prb_dim_t &dim) const {
        return entries_.at(dim).thr_idx;
    }
    const expr_t &iter_index(const prb_dim_t &dim) const {
        return entries_.at(dim).iter_idx;
    }
    const expr_t &loop_size(const prb_dim_t &dim) const {
        return entries_.at(dim).loop_size;
    }
    const expr_t &loop_index(const prb_dim_t &dim) const {
        return entries_.at(dim).loop_idx;
    }

    prb_coord_t<expr_t> iter_coord() const;
    prb_coord_t<expr_t> tg_iter_coord() const;
    prb_tile_t tg_iter_tile() const;

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
        prb_dim_t dim;
        expr_t tg_idx;
        expr_t thr_idx;
        expr_t iter_idx;
        expr_t loop_idx;

        int tg_size = 0;
        int iter_size = 0;
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

    dim_map_t<prb_dim_t, entry_t> entries_;
};

class virt_grid_t {
public:
    void add(const expr_t &var, const expr_t &expr) {
        auto ret = idxs_.emplace(var, expr);
        ir_assert(ret.second);
    }

    const object_map_t<expr_t, expr_t> &idxs() const { return idxs_; }

private:
    object_map_t<expr_t, expr_t> idxs_;
};

struct reorder_plan_t : public base_plan_t {
    layout_t src;
    layout_t dst;

    using base_plan_t::base_plan_t;

    reorder_plan_t() = default;
    reorder_plan_t(const hw_t &hw, const layout_t &src, const layout_t &dst)
        : base_plan_t(hw), src(src), dst(dst) {}

    int grf_usage_bytes() const {
        int ret = 0;
        ret += utils::rnd_up(dst.size(), grf_size());
        return ret;
    }

    std::string str() const {
        if (!*this) return "(empty)";
        std::ostringstream oss;
        oss << "src_layout: " << src.str() << std::endl;
        oss << "dst_layout: " << dst.str();
        return oss.str();
    }

    IR_DEFINE_DUMP()
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
        if (!*this) return "(empty)";
        std::ostringstream oss;
        oss << ir_utils::add_tag("a_prefetch", a_prefetch.str()) << std::endl;
        oss << ir_utils::add_tag("b_prefetch", b_prefetch.str());
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

struct x2r_plan_t : public base_plan_t {
    send_plan_t a_load;
    send_plan_t b_load;
    reorder_plan_t a_reorder;
    reorder_plan_t b_reorder;
    layout_t a_layout;
    layout_t b_layout;

    using base_plan_t::base_plan_t;

    int grf_usage_bytes() const {
        int ret = 0;
        ret += a_load.grf_usage_bytes();
        ret += b_load.grf_usage_bytes();
        return ret;
    }

    prb_reqs_t reqs() const {
        prb_reqs_t ret;
        ret.add(a_load.reqs());
        ret.add(b_load.reqs());
        ret.simplify();
        return ret;
    }

    std::string str() const {
        if (!*this) return "(empty)";
        std::ostringstream oss;
        oss << "a_layout: " << a_layout.str() << std::endl;
        oss << "b_layout: " << b_layout.str() << std::endl;
        oss << ir_utils::add_tag("a_load", a_load.str()) << std::endl;
        oss << ir_utils::add_tag("b_load", b_load.str()) << std::endl;
        if (a_reorder)
            oss << ir_utils::add_tag("a_reorder", a_reorder.str()) << std::endl;
        if (b_reorder)
            oss << ir_utils::add_tag("b_reorder", b_reorder.str()) << std::endl;
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

struct fma_plan_t : public base_plan_t {
    layout_t a_layout;
    layout_t b_layout;
    layout_t c_layout;
    prb_tile_t inst_tile;
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

struct epilogue_plan_t : public base_plan_t {
    prb_tile_t tile;
    reorder_plan_t reorder;
    send_plan_t c_store;

    epilogue_plan_t(const hw_t &hw) : base_plan_t(hw) {}

    int grf_usage_bytes() const { return 0; }

    std::string str() const {
        if (!*this) return "(empty)";
        std::ostringstream oss;
        oss << "tile: " << tile << std::endl;
        if (reorder)
            oss << ir_utils::add_tag("reorder", reorder.str()) << std::endl;
        oss << ir_utils::add_tag("c_store", c_store.str());
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
    x2r_plan_t x2r;
    fma_plan_t fma;
    epilogue_plan_t epilogue;

    plan_t(const hw_t &hw = hw_t())
        : base_plan_t(hw), prefetch(hw), x2r(hw), fma(hw), epilogue(hw) {}

    int grf_usage_bytes() const {
        int ret = 0;
        ret += x2r.grf_usage_bytes();
        ret += fma.grf_usage_bytes();
        ret += epilogue.grf_usage_bytes();
        return ret;
    }

    prb_reqs_t reqs() const;

    std::string str() const {
        if (!*this) return "(empty)";
        std::ostringstream oss;
        oss << ir_utils::add_tag("prefetch", prefetch.str()) << std::endl;
        oss << ir_utils::add_tag("x2r", x2r.str()) << std::endl;
        oss << ir_utils::add_tag("fma", fma.str()) << std::endl;
        oss << ir_utils::add_tag("epilogue", epilogue.str());
        return ir_utils::add_tag("Plan", oss.str());
    }

    IR_DEFINE_DUMP()
};

plan_t create_conv_plan(const kernel_desc_t &desc);
plan_t create_conv_plan_and_finalize_desc(kernel_desc_t &desc);

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
