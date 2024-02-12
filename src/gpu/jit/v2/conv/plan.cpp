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

#include "gpu/jit/v2/conv/plan.hpp"

#include <algorithm>
#include <string>

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {
namespace conv {

layout_tag_t append_groups(
        tensor_kind_t tensor_kind, const layout_tag_t &layout_tag, bool is_dw) {
    bool is_src = (tensor_kind == tensor_kind_t::src);
    bool is_dst = (tensor_kind == tensor_kind_t::dst);
    if (!is_src && !is_dst) return layout_tag;
    auto xc_dim = (is_src ? prb_dims::ic : prb_dims::oc);
    auto xc_letter = 'a' + layout_tag.desc().dim_index(xc_dim);
    auto new_g_letter = xc_letter;
    auto new_xc_letter = xc_letter + 1;
    auto &raw_tag = layout_tag.raw_tag();
    auto &entries = raw_tag.entries();
    layout_raw_tag_t new_raw_tag;
    for (auto &e : entries) {
        if (e.letter == xc_letter) {
            if (is_dw) {
                new_raw_tag.add_entry(new_g_letter, e.block, e.is_blocked);
            } else if (e.is_outer()) {
                new_raw_tag.add_entry(new_g_letter, 0, false);
                new_raw_tag.add_entry(new_xc_letter, e.block, e.is_blocked);
            } else {
                new_raw_tag.add_entry(new_xc_letter, e.block, e.is_blocked);
            }
        } else {
            char letter = e.letter;
            if (letter >= new_xc_letter) letter++;
            new_raw_tag.add_entry(letter, e.block, e.is_blocked);
        }
    }
    auto desc = make_conv_layout_desc(tensor_kind, /*src_dst_with_group=*/true);
    return layout_tag_t(desc, layout_tag.type(), new_raw_tag);
}

layout_t make_conv_layout(
        tensor_kind_t tensor_kind, const layout_tag_t &_tag, bool is_dw) {
    auto tag = append_groups(tensor_kind, _tag, is_dw);
    layout_t ret(tag.desc(), tag.type());
    dim_map_t<prb_dim_t, int> blocks;
    auto rem_size = [](const prb_dim_t &dim,
                            const dim_map_t<prb_dim_t, int> &blocks) {
        auto dim_size = size_var(dim);
        if (!blocks.has(dim)) return dim_size;
        return binary_op_t::make(op_kind_t::_div_up, dim_size, blocks[dim]);
    };
    auto &entries = tag.raw_tag().entries();
    for (auto it = entries.rbegin(); it != entries.rend(); it++) {
        prb_dim_t dim = tag.desc().prb_dim(it->index());
        int block_size = it->block;
        expr_t block_size_expr;
        if (block_size > 0) {
            blocks[dim] = blocks.get(dim, 1) * block_size;
            block_size_expr = expr_t(block_size);
        } else {
            block_size_expr = rem_size(dim, blocks);
        }
        ret.add_block(dim, block_size_expr);
    }
    return ret;
}

class dim_mapper_manager_t {
public:
    dim_mapper_manager_t() = default;
    dim_mapper_manager_t(prop_kind_t prop) : prop_(prop) {
        src_mapper_ = init_src_mapper();
        wei_mapper_ = init_wei_mapper();
        dst_mapper_ = init_dst_mapper();
    }

    const dim_mapper_t &mapper(tensor_kind_t tensor) const {
        switch (tensor) {
            case tensor_kind_t::src: return src_mapper_;
            case tensor_kind_t::wei: return wei_mapper_;
            case tensor_kind_t::dst: return dst_mapper_;
            case tensor_kind_t::a:
                return mapper(pick_a(prop_, tensor_kind_t::src,
                        tensor_kind_t::wei, tensor_kind_t::dst));
            case tensor_kind_t::b:
                return mapper(pick_b(prop_, tensor_kind_t::src,
                        tensor_kind_t::wei, tensor_kind_t::dst));
            case tensor_kind_t::c:
                return mapper(pick_c(prop_, tensor_kind_t::src,
                        tensor_kind_t::wei, tensor_kind_t::dst));
            default: ir_error_not_expected();
        }
        return src_mapper_;
    }

private:
    expr_t kw_idx = index_var(prb_dims::kw);
    expr_t kh_idx = index_var(prb_dims::kh);
    expr_t kd_idx = index_var(prb_dims::kd);
    expr_t id_idx = index_var(prb_dims::id);
    expr_t ih_idx = index_var(prb_dims::ih);
    expr_t iw_idx = index_var(prb_dims::iw);
    expr_t od_idx = index_var(prb_dims::od);
    expr_t oh_idx = index_var(prb_dims::oh);
    expr_t ow_idx = index_var(prb_dims::ow);
    expr_t od_bwd_d_idx = var_t::make(type_t::s32(), "od_bwd_d_idx");
    expr_t oh_bwd_d_idx = var_t::make(type_t::s32(), "oh_bwd_d_idx");
    expr_t ow_bwd_d_idx = var_t::make(type_t::s32(), "ow_bwd_d_idx");
    expr_t kd_bwd_d_idx = var_t::make(type_t::s32(), "kd_bwd_d_idx");
    expr_t kh_bwd_d_idx = var_t::make(type_t::s32(), "kh_bwd_d_idx");
    expr_t kw_bwd_d_idx = var_t::make(type_t::s32(), "kw_bwd_d_idx");

    dim_mapper_t init_src_mapper() const {
        auto pd = size_var(prb_dims::pd);
        auto ph = size_var(prb_dims::ph);
        auto pw = size_var(prb_dims::pw);
        auto sd = size_var(prb_dims::sd);
        auto sh = size_var(prb_dims::sh);
        auto sw = size_var(prb_dims::sw);
        auto dd = size_var(prb_dims::dd);
        auto dh = size_var(prb_dims::dh);
        auto dw = size_var(prb_dims::dw);
        dim_mapper_t mapper;
        mapper.set_dim(prb_dims::mb);
        mapper.set_dim(prb_dims::g);
        mapper.set_dim(prb_dims::ic);
        if (utils::one_of(
                    prop_, prop_kind::forward, prop_kind::backward_weights)) {
            auto dd_inc = dd + 1;
            auto dh_inc = dh + 1;
            auto dw_inc = dw + 1;
            auto neg_pd = -pd;
            auto neg_ph = -ph;
            auto neg_pw = -pw;
            mapper.set_dim(
                    prb_dims::id, sd * od_idx + neg_pd + kd_idx * dd_inc);
            mapper.set_dim(
                    prb_dims::ih, sh * oh_idx + neg_ph + kh_idx * dh_inc);
            mapper.set_dim(
                    prb_dims::iw, sw * ow_idx + neg_pw + kw_idx * dw_inc);
        } else {
            mapper.set_dim(prb_dims::id);
            mapper.set_dim(prb_dims::ih);
            mapper.set_dim(prb_dims::iw);
        }
        mapper.set_layout_desc(
                make_conv_algo_layout_desc(prop_, tensor_kind_t::src));
        return mapper;
    }

    dim_mapper_t init_wei_mapper() const {
        dim_mapper_t mapper;
        mapper.set_dim(prb_dims::g);
        mapper.set_dim(prb_dims::oc);
        mapper.set_dim(prb_dims::ic);
        mapper.set_dim(prb_dims::kd);
        mapper.set_dim(prb_dims::kh);
        mapper.set_dim(prb_dims::kw);
        mapper.set_layout_desc(
                make_conv_algo_layout_desc(prop_, tensor_kind_t::wei));
        return mapper;
    }

    dim_mapper_t init_dst_mapper() const {
        dim_mapper_t mapper;
        mapper.set_dim(prb_dims::mb);
        mapper.set_dim(prb_dims::g);
        mapper.set_dim(prb_dims::oc);
        if (utils::one_of(
                    prop_, prop_kind::forward, prop_kind::backward_weights)) {
            mapper.set_dim(prb_dims::od);
            mapper.set_dim(prb_dims::oh);
            mapper.set_dim(prb_dims::ow);
        } else {
            mapper.set_dim(prb_dims::od, od_bwd_d_idx - kd_bwd_d_idx);
            mapper.set_dim(prb_dims::oh, oh_bwd_d_idx - kh_bwd_d_idx);
            mapper.set_dim(prb_dims::ow, ow_bwd_d_idx - kw_bwd_d_idx);
        }
        mapper.set_layout_desc(
                make_conv_algo_layout_desc(prop_, tensor_kind_t::dst));
        return mapper;
    }

    prop_kind_t prop_ = prop_kind::undef;
    dim_mapper_t src_mapper_;
    dim_mapper_t wei_mapper_;
    dim_mapper_t dst_mapper_;
};

class multiply_info_t {
public:
    multiply_info_t() = default;
    multiply_info_t(fma_kind_t fma, int simd) : fma_(fma), simd_(simd) {}

    multiply_hint_t hint(tensor_kind_t abc) const {
        if (!utils::one_of(abc, tensor_kind_t::a, tensor_kind_t::b))
            return multiply_hint_t();
        bool src1 = (abc == tensor_kind_t::b);
        bool src2 = (abc == tensor_kind_t::a);
        multiply_hint_t hint;
        hint.fma = fma_;
        hint.simd = simd_;
        hint.src1 = src1;
        hint.src2 = src2;
        hint.bmnk_map = bmnk_map_;
        return hint;
    }

    fma_kind_t fma() const { return fma_; }
    int simd() const { return simd_; }

    void set(const prb_dim_t &dim, const prb_dim_t &bmnk_dim) {
        ir_assert(utils::one_of(
                bmnk_dim, prb_dims::b, prb_dims::m, prb_dims::n, prb_dims::k));
        bmnk_map_[dim] = bmnk_dim.kind();
    }

    bool has(tensor_kind_t abc, const prb_dim_t &dim) const {
        switch (abc) {
            case tensor_kind_t::a: return is_b(dim) || is_m(dim) || is_k(dim);
            case tensor_kind_t::b: return is_b(dim) || is_k(dim) || is_n(dim);
            case tensor_kind_t::c: return is_b(dim) || is_m(dim) || is_n(dim);
            default: ir_error_not_expected();
        }
        return false;
    }

    bool is(const prb_dim_t &dim, prb_dim_kind_t kind) const {
        ir_assert(utils::one_of(kind, prb_dim_kind_t::b, prb_dim_kind_t::m,
                prb_dim_kind_t::n, prb_dim_kind_t::k));
        if (!bmnk_map_.has(dim)) return false;
        return bmnk_map_[dim] == kind;
    }

    bool is_b(const prb_dim_t &dim) const { return is(dim, prb_dim_kind_t::b); }
    bool is_m(const prb_dim_t &dim) const { return is(dim, prb_dim_kind_t::m); }
    bool is_n(const prb_dim_t &dim) const { return is(dim, prb_dim_kind_t::n); }
    bool is_k(const prb_dim_t &dim) const { return is(dim, prb_dim_kind_t::k); }

private:
    fma_kind_t fma_ = fma_kind_t::undef;
    int simd_ = 0;
    dim_map_t<prb_dim_t, prb_dim_kind_t> bmnk_map_;
};

class layout_info_t {
public:
    layout_info_t() = default;

    layout_info_t(const multiply_info_t &mul_info, const prb_tile_t &iter_tile,
            const layout_t &a, const layout_t &b) {
        for (auto &d : iter_tile) {
            if (iter_tile[d] % mul_info.simd() != 0) continue;
            if (mul_info.is_n(d) && is_one(b.stride(d))) {
                block_t block;
                block.dim = d;
                block.size = mul_info.simd();
                block.stride = expr_t(1);
                b_inner_ = layout_t(b.desc(), b.type(), 0, {block});
                break;
            }
        }
    }

    bool is_compatible(tensor_kind_t abc, const layout_t &layout) const {
        switch (abc) {
            case tensor_kind_t::a: return layout.is_blocked_by(a_inner_);
            case tensor_kind_t::b: return layout.is_blocked_by(b_inner_);
            default: ir_error_not_expected();
        }
        return false;
    }

private:
    layout_t a_inner_;
    layout_t b_inner_;
};

class plan_builder_t {
public:
    plan_builder_t() = default;
    plan_builder_t(const kernel_desc_t &desc) : desc_(desc) {}

    plan_t build() {
        init_dim_mapper_manager();
        init_tiles();
        init_layouts();
        if (!init_info()) return plan_t();
        return init_plan();
    }

private:
    void init_dim_mapper_manager() {
        dim_mapper_manager_ = dim_mapper_manager_t(desc_.prop);
    }

    void init_tiles() {
        tg_grid_ = create_thread_group_grid(desc_);
        thr_grid_ = create_thread_grid(desc_);
        for (auto &d : conv_index_dims(desc_.prop)) {
            bool is_loop = desc_.loop_nest.has(d);
            bool is_global_loop = desc_.loop_nest.is_global(d);
            int tg_tile = desc_.thread_group_tile.get(d, 1);
            int iter_tile = desc_.iter_tile.get(d, 1);
            auto thr_idx = thr_grid_.index_var(d);
            coord_info_.add_dim(
                    d, is_loop, is_global_loop, tg_tile, thr_idx, iter_tile);
            iter_coord_[d]
                    = simplify_rewrite(iter_tile * coord_info_.iter_index(d));
        }
    }

    void init_layouts() {
        auto src_layout = make_conv_layout(
                tensor_kind_t::src, desc_.src_tag, desc_.is_dw);
        auto wei_layout = make_conv_layout(
                tensor_kind_t::wei, desc_.wei_tag, desc_.is_dw);
        auto dst_layout = make_conv_layout(
                tensor_kind_t::dst, desc_.dst_tag, desc_.is_dw);
        auto &a_mapper = dim_mapper_manager_.mapper(tensor_kind_t::a);
        auto &b_mapper = dim_mapper_manager_.mapper(tensor_kind_t::b);
        a_layout_ = pick_a(desc_.prop, src_layout, wei_layout, dst_layout);
        b_layout_ = pick_b(desc_.prop, src_layout, wei_layout, dst_layout);
        c_layout_ = pick_c(desc_.prop, src_layout, wei_layout, dst_layout);
        a_iter_view_
                = view_t(a_mapper, a_layout_, iter_coord_, desc_.iter_tile);
        b_iter_view_
                = view_t(b_mapper, b_layout_, iter_coord_, desc_.iter_tile);
    }

    bool init_info() {
        mul_info_ = multiply_info_t(desc_.fma, desc_.simd);
        for (auto &d : conv_index_dims(desc_.prop)) {
            mul_info_.set(d, to_gemm(d, desc_.prop));
        }
        layout_info_ = layout_info_t(
                mul_info_, desc_.iter_tile, a_layout_, b_layout_);
        return true;
    }


    plan_t init_plan() {
        plan_t plan(desc_.hw);
        if (!try_init_plan(plan)) return plan_t();
        if (!check_plan(plan)) return plan_t();

        reqs_ = plan.reqs();
        plan = plan_t(desc_.hw);
        if (!try_init_plan(plan) || !check_plan(plan)) {
            ir_error_not_expected();
            return plan_t();
        }
        return plan;
    }

    bool try_init_plan(plan_t &plan) const {
        plan.desc = desc_;
        plan.tg_grid = tg_grid_;
        plan.thr_grid = thr_grid_;
        plan.coord_info = coord_info_;
        ir_check(init_x2r_plan(plan.x2r));
        ir_check(init_fma_plan(plan.x2r, plan.fma));
        ir_check(init_epilogue_plan(plan.fma, plan.epilogue));
        return true;
    }

    bool init_x_g2r_plan(tensor_kind_t abc, const view_t &view,
            layout_t &reg_layout, send_plan_t &load) const {
        auto params = get_send_params(abc, send_op_t::load, view);
        load = create_send_plan(params, view, /*allow_fail=*/true);
        ir_check(load) << "init_x_x2r_plan: cannot create send plan";
        bool ok = layout_info_.is_compatible(abc, load.reg_layout());
        if (params.hint_2d && !ok) {
            params.downgrade_to_1d();
            load = create_send_plan(params, view);
            ok = layout_info_.is_compatible(abc, load.reg_layout());
        }
        ir_check(ok) << "init_x_x2r_plan: incompatible layout";
        reg_layout = load.reg_layout();
        return true;
    }

    bool init_x2r_plan(x2r_plan_t &plan) const {
        ir_check(init_x_g2r_plan(
                tensor_kind_t::a, a_iter_view_, plan.a_layout, plan.a_load));
        ir_check(init_x_g2r_plan(
                tensor_kind_t::b, b_iter_view_, plan.b_layout, plan.b_load));
        return true;
    }

    static type_t get_acc_type(const type_t &a, const type_t &b) {
        ir_assert(a.size() == b.size());
        if (a.is_int()) return type_t::s32();
        return type_t::f32();
    }

    static bool try_vectorize_by(const prb_dim_t &vec_dim, const layout_t &a,
            const layout_t &b, const multiply_info_t &mul_info,
            prb_tile_t &inst_tile) {
        if (mul_info.is_k(vec_dim)) return false;
        bool a_vec = mul_info.has(tensor_kind_t::a, vec_dim);
        bool b_vec = mul_info.has(tensor_kind_t::b, vec_dim);
        bool a_ok = !a_vec || a.is_blocked_by(vec_dim, mul_info.simd());
        bool b_ok = !b_vec || b.is_blocked_by(vec_dim, mul_info.simd());
        if (!a_ok || !b_ok) return false;
        inst_tile[vec_dim] = mul_info.simd();
        return true;
    }

    layout_t create_c_layout(const layout_t &a_layout, const layout_t &b_layout,
            const prb_dim_t &inner_dim, int inner_block_size) const {
        auto &c_desc = c_layout_.desc();
        auto c_type = get_acc_type(a_layout.type(), b_layout.type());
        ir_assert(a_layout.has_const_sizes());
        ir_assert(b_layout.has_const_sizes());
        layout_t c(c_desc, c_type);
        for (auto &b : a_layout.blocks()) {
            if (mul_info_.is_k(b.dim)) continue;
            c.add_block(b.dim, b.size);
        }
        for (auto &b : b_layout.blocks()) {
            if (mul_info_.is_k(b.dim)) continue;
            c.add_block(b.dim, b.size);
        }
        c.block_by({block_t(inner_dim, inner_block_size)});
        return c;
    }

    bool init_fma_plan(const x2r_plan_t &x2r, fma_plan_t &plan) const {
        ir_assert(desc_.fma == fma_kind_t::mad);
        auto &a = x2r.a_layout;
        auto &b = x2r.b_layout;
        prb_tile_t inst_tile;
        for (auto &d : desc_.iter_tile) {
            inst_tile[d] = 1;
        }
        layout_t c;
        for (auto &d : desc_.iter_tile) {
            if (try_vectorize_by(d, a, b, mul_info_, inst_tile)) {
                c = create_c_layout(a, b, d, desc_.simd);
                break;
            }
        }
        ir_check(!c.is_empty()) << "init_fma_plan: cannot vectorize";
        plan.simd = desc_.simd;
        plan.fma = desc_.fma;
        plan.a_layout = a;
        plan.b_layout = b;
        plan.c_layout = c;
        plan.inst_tile = inst_tile;
        return true;
    }

    bool init_epilogue_plan(
            const fma_plan_t &fma, epilogue_plan_t &plan) const {
        auto &c_mapper = dim_mapper_manager_.mapper(tensor_kind_t::c);
        auto c_iter_view
                = view_t(c_mapper, c_layout_, iter_coord_, desc_.iter_tile);
        int target_elems = 128 / c_layout_.type().size();
        auto it_beg = begin(c_iter_view.layout());
        auto it_end = end(c_iter_view.layout());
        auto tile_last = it_beg;
        for (auto it = it_beg; it != it_end; ++it) {
            if (it.elems() > target_elems) break;
            tile_last = it;
        }
        auto full_tile = desc_.iter_tile;
        for (auto &d : full_tile) {
            if (mul_info_.is_k(d)) full_tile.unset(d);
        }
        auto params = get_send_params(
                tensor_kind_t::c, send_op_t::store, c_iter_view);
        auto c_store = create_send_plan(params, c_iter_view);
        auto tile = c_store.entry_tile();
        plan.tile = tile;
        plan.c_store = c_store;
        if (fma.c_layout != c_store.reg_layout()) {
            auto fma_layout = fma.c_layout.map(tile);
            auto store_layout = c_store.reg_layout().map(tile);
            if (fma_layout != store_layout) {
                plan.reorder = reorder_plan_t(desc_.hw);
                plan.reorder.src = fma_layout;
                plan.reorder.dst = store_layout;
            }
        }
        return true;
    }

    bool check_plan(const plan_t &plan) const {
        int bound = desc_.hw.grf_size() * desc_.regs;
        int usage_bytes = plan.grf_usage_bytes();
        ir_check(usage_bytes <= bound) << "check_plan: out of registers";
        return true;
    }

    send_params_t get_send_params(tensor_kind_t abc, send_op_t op,
            const view_t &view,
            send_kind_t send_kind = send_kind_t::undef) const {
        send_params_t params;
        params.hw = desc_.hw;
        params.kind = (send_kind != send_kind_t::undef
                        ? send_kind
                        : desc_.access_kind(op, abc));
        params.op = op;
        if (params.kind == send_kind_t::_2d)
            params.hint_2d = send_2d_hint_t(view, op, mul_info_.hint(abc));
        params.skip_mask = skip_mask(view);
        params.init_max_entry_reg_size();
        return params;
    }

    std::vector<prb_dim_t> skip_mask(const view_t &view) const {
        std::vector<prb_dim_t> ret;
        auto &mask_desc = view.mask_desc();
        for (int i = 0; i < mask_desc.nmasks(); i++) {
            prb_dim_t d = mask_desc[i].dim;
            if (!view.dim_mapper().has(d)) continue;
            if (!view.dim_mapper().expr(d).is_same(index_var(d))) continue;
            if (coord_info_.needs_mask(d)) continue;
            ret.push_back(d);
        }
        return ret;
    }

    kernel_desc_t desc_;

    dim_mapper_manager_t dim_mapper_manager_;
    multiply_info_t mul_info_;
    layout_info_t layout_info_;
    coord_info_t coord_info_;
    prb_coord_t<expr_t> iter_coord_;
    grid_t tg_grid_;
    grid_t thr_grid_;
    layout_t a_layout_;
    layout_t b_layout_;
    layout_t c_layout_;
    view_t a_iter_view_;
    view_t b_iter_view_;
    prb_reqs_t reqs_;
};

prb_reqs_t plan_t::reqs() const {
    prb_reqs_t ret;
    ret.add(x2r.reqs());
    ret.add(epilogue.c_store.reqs());
    ret.simplify();
    return ret;
}

plan_t create_conv_plan(const kernel_desc_t &desc) {
    if (!desc.is_supported()) return plan_t();
    plan_builder_t builder(desc);
    auto plan = builder.build();
    return plan;
}

plan_t create_conv_plan_and_finalize_desc(kernel_desc_t &desc) {
    auto plan = create_conv_plan(desc);
    if (plan) desc.finalize(plan);
    return plan;
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
