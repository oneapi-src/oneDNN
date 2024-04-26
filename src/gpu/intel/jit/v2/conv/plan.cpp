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

#include "gpu/intel/jit/v2/conv/plan.hpp"

#include <algorithm>
#include <string>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

prb_coord_t<expr_t> coord_info_t::iter_coord() const {
    prb_coord_t<expr_t> ret;
    for (auto &d : entries_) {
        auto &e = entries_.at(d);
        ret[d] = simplify_rewrite(e.iter_size * e.iter_idx);
    }
    return ret;
}

prb_coord_t<expr_t> coord_info_t::tg_iter_coord() const {
    prb_coord_t<expr_t> ret;
    for (auto &d : entries_) {
        auto &e = entries_.at(d);
        auto idx = e.iter_size * e.iter_idx;
        if (!is_const(e.thr_idx)) {
            idx = substitute(idx, e.thr_idx, expr_t(0));
        }
        ret[d] = simplify_rewrite(idx);
    }
    return ret;
}

prb_tile_t coord_info_t::tg_iter_tile() const {
    prb_tile_t ret;
    for (auto &d : entries_) {
        auto &e = entries_.at(d);
        ret[d] = e.tg_size * e.iter_size;
    }
    return ret;
}

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
                new_raw_tag.add_entry(new_xc_letter, 1, false);
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

layout_t make_conv_layout(tensor_kind_t tensor_kind, const layout_tag_t &_tag,
        bool is_dw, const spec_reqs_t &spec_reqs) {
    auto tag = append_groups(tensor_kind, _tag, is_dw);
    layout_t ret(tag.desc(), tag.type());
    dim_map_t<prb_dim_t, int> blocks;
    auto rem_size = [&](const prb_dim_t &dim,
                            const dim_map_t<prb_dim_t, int> &blocks) {
        auto dim_size = size_var(dim);
        bool is_dim_1 = spec_reqs.is_equal(dim, 1);
        if (is_dim_1) return expr_t(1);
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
    dim_mapper_manager_t(prop_kind_t prop, const spec_reqs_t &reqs)
        : prop_(prop), reqs_(reqs) {
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

    dim_mapper_t init_src_mapper() const {
        auto pd = reqs_.to_expr(prb_dims::pd);
        auto ph = reqs_.to_expr(prb_dims::ph);
        auto pw = reqs_.to_expr(prb_dims::pw);
        auto sd = reqs_.to_expr(prb_dims::sd);
        auto sh = reqs_.to_expr(prb_dims::sh);
        auto sw = reqs_.to_expr(prb_dims::sw);
        auto dd = reqs_.to_expr(prb_dims::dd);
        auto dh = reqs_.to_expr(prb_dims::dh);
        auto dw = reqs_.to_expr(prb_dims::dw);
        dim_mapper_t mapper;
        mapper.set_dim(prb_dims::mb);
        mapper.set_dim(prb_dims::g);
        mapper.set_dim(prb_dims::ic);
        if (utils::one_of(
                    prop_, prop_kind::forward, prop_kind::backward_weights)) {
            auto dd_inc = const_fold(dd + 1);
            auto dh_inc = const_fold(dh + 1);
            auto dw_inc = const_fold(dw + 1);
            auto neg_pd = const_fold(-pd);
            auto neg_ph = const_fold(-ph);
            auto neg_pw = const_fold(-pw);
            mapper.set_dim(prb_dims::id,
                    simplify_rewrite(sd * od_idx + neg_pd + kd_idx * dd_inc),
                    true);
            mapper.set_dim(prb_dims::ih,
                    simplify_rewrite(sh * oh_idx + neg_ph + kh_idx * dh_inc),
                    true);
            mapper.set_dim(prb_dims::iw,
                    simplify_rewrite(sw * ow_idx + neg_pw + kw_idx * dw_inc),
                    true);
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
            auto pd = reqs_.to_expr(prb_dims::pd);
            auto ph = reqs_.to_expr(prb_dims::ph);
            auto pw = reqs_.to_expr(prb_dims::pw);
            auto sd = reqs_.to_expr(prb_dims::sd);
            auto sh = reqs_.to_expr(prb_dims::sh);
            auto sw = reqs_.to_expr(prb_dims::sw);
            auto dd = reqs_.to_expr(prb_dims::dd);
            auto dh = reqs_.to_expr(prb_dims::dh);
            auto dw = reqs_.to_expr(prb_dims::dw);

            auto dd_inc = const_fold(dd + 1);
            auto dh_inc = const_fold(dh + 1);
            auto dw_inc = const_fold(dw + 1);

            mapper.set_dim(prb_dims::od,
                    simplify_rewrite((id_idx + pd - (kd_idx * dd_inc)) / sd),
                    true);
            mapper.set_dim(prb_dims::oh,
                    simplify_rewrite((ih_idx + ph - (kh_idx * dh_inc)) / sh),
                    true);
            mapper.set_dim(prb_dims::ow,
                    simplify_rewrite((iw_idx + pw - (kw_idx * dw_inc)) / sw),
                    true);
        }
        mapper.set_layout_desc(
                make_conv_algo_layout_desc(prop_, tensor_kind_t::dst));
        return mapper;
    }

    prop_kind_t prop_ = prop_kind::undef;
    spec_reqs_t reqs_;
    dim_mapper_t src_mapper_;
    dim_mapper_t wei_mapper_;
    dim_mapper_t dst_mapper_;
};

class multiply_info_t {
public:
    multiply_info_t() = default;
    multiply_info_t(fma_kind_t fma, int simd, const prb_tile_t &iter_tile,
            const dim_map_t<prb_dim_t, prb_dim_kind_t> &bmnk_map,
            const type_t &a_type, const layout_desc_t &a_desc,
            const type_t &b_type, const layout_desc_t &b_desc,
            const layout_desc_t &c_desc)
        : fma_(fma)
        , simd_(simd)
        , iter_tile_(iter_tile)
        , bmnk_map_(bmnk_map)
        , a_type_(a_type)
        , b_type_(b_type) {
        init_acc_type();
        if (!init(a_desc, b_desc, c_desc)) return;
        is_valid_ = true;
    }

    explicit operator bool() const { return is_valid_; }

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
    const type_t &a_type() const { return a_type_; }
    const type_t &b_type() const { return b_type_; }
    const type_t &acc_type() const { return acc_type_; }

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
    prb_dim_kind_t to_bmnk(const prb_dim_t &dim) const {
        return bmnk_map_.at(dim);
    }

    prb_tile_t inst_tile() const {
        prb_tile_t ret;
        switch (fma_) {
            case fma_kind_t::mad: ret = b_inner_.int_dim_sizes(); break;
            case fma_kind_t::dpas: {
                auto a_tile = a_inner_.int_dim_sizes();
                auto b_tile = b_inner_.int_dim_sizes();
                ret = a_tile;
                for (auto &d : b_tile) {
                    if (ret.has(d)) ir_assert(ret[d] == b_tile[d]);
                    ret[d] = b_tile[d];
                }
                return ret;
            }
            default: ir_error_not_expected();
        }
        for (auto &d : iter_tile_) {
            if (!ret.has(d)) ret[d] = 1;
        }
        return ret;
    }

    bool is_compatible(tensor_kind_t abc, const layout_t &layout) const {
        if (!fma_type_supported(layout.type())) return false;
        switch (abc) {
            case tensor_kind_t::a: return layout.is_blocked_by(a_inner_);
            case tensor_kind_t::b: return layout.is_blocked_by(b_inner_);
            default: ir_error_not_expected();
        }
        return false;
    }

    layout_t to_compatible_layout(
            tensor_kind_t abc, const layout_t &layout) const {
        auto ret = layout;
        switch (abc) {
            case tensor_kind_t::a: ret.block_by(a_inner_.blocks()); break;
            case tensor_kind_t::b: ret.block_by(b_inner_.blocks()); break;
            default: ir_error_not_expected();
        }
        ret = get_fma_type_layout(ret);
        return ret;
    }

    layout_t acc_layout(const layout_t &a_layout, const layout_t &b_layout,
            const layout_t &c_layout) const {
        ir_assert(a_layout.has_const_sizes());
        ir_assert(b_layout.has_const_sizes());
        layout_t acc(c_layout.desc(), acc_type());
        for (auto &b : a_layout.blocks()) {
            if (is_k(b.dim)) continue;
            acc.add_block(b.dim, b.size);
        }
        for (auto &b : b_layout.blocks()) {
            if (is_k(b.dim) || is_b(b.dim)) continue;
            acc.add_block(b.dim, b.size);
        }
        acc.block_by(c_inner_.blocks());
        return acc;
    }

private:
    void init_acc_type() {
        ir_assert(a_type_.size() == b_type_.size());
        switch (fma_) {
            case fma_kind_t::mad:
                acc_type_ = a_type_.is_fp() ? type_t::f32() : type_t::s32();
                break;
            case fma_kind_t::dpas:
                acc_type_ = a_type_.is_fp() ? type_t::f32() : type_t::s32();
                break;
            default: ir_error_not_expected();
        }
    }

    bool fma_type_supported(const type_t &type) const {
        switch (fma_) {
            case fma_kind_t::mad:
                return utils::one_of(type, type_t::f32(), type_t::s16());
                break;
            case fma_kind_t::dpas:
                return utils::one_of(type, type_t::u8(), type_t::s8(),
                        type_t::f16(), type_t::bf16());
                break;
            default: ir_error_not_expected();
        }
        return false;
    }

    layout_t get_fma_type_layout(const layout_t &layout) const {
        if (fma_ == fma_kind_t::mad) {
            auto blocks = layout.blocks();
            if (utils::one_of(layout.type(), type_t::s8(), type_t::u8())) {

                for (auto &b : blocks) {
                    b.stride *= 2;
                }
                return layout_t(
                        layout.desc(), type_t::s16(), layout.base(), blocks);
            }
            if (utils::one_of(layout.type(), type_t::f16(), type_t::bf16(),
                        type_t::f32()))
                return layout_t(
                        layout.desc(), type_t::f32(), layout.base(), blocks);
        }
        return layout;
    }

    bool init(const layout_desc_t &a_desc, const layout_desc_t &b_desc,
            const layout_desc_t &c_desc) {
        switch (fma_) {
            case fma_kind_t::mad: return init_mad(a_desc, b_desc, c_desc);
            case fma_kind_t::dpas: return init_dpas(a_desc, b_desc, c_desc);
            default: ir_error_not_expected();
        }
        return false;
    }

    bool init_mad(const layout_desc_t &a_desc, const layout_desc_t &b_desc,
            const layout_desc_t &c_desc) {
        bool found = false;
        for (auto &d : iter_tile_) {
            if (iter_tile_[d] % simd_ != 0) continue;
            if (is_n(d) || is_b(d)) {
                found = true;
                block_t block;
                block.dim = d;
                block.size = simd_;
                block.stride = expr_t(1);
                b_inner_ = layout_t(b_desc, b_type_, 0, {block});
                break;
            }
        }
        ir_check(found) << "init_mad: cannot find dimension to vectorize.";
        c_inner_ = layout_t(c_desc, acc_type_, 0, b_inner_.blocks());
        return true;
    }

    bool init_dpas(const layout_desc_t &a_desc, const layout_desc_t &b_desc,
            const layout_desc_t &c_desc) {
        prb_dim_t m_dim;
        prb_dim_t n_dim;
        prb_dim_t k_dim;
        for (auto &d : iter_tile_) {
            switch (to_bmnk(d)) {
                case prb_dim_kind_t::m:
                    ir_assert(m_dim.is_undef());
                    m_dim = d;
                    break;
                case prb_dim_kind_t::n:
                    ir_assert(n_dim.is_undef());
                    n_dim = d;
                    break;
                case prb_dim_kind_t::k:
                    ir_assert(k_dim.is_undef());
                    k_dim = d;
                    break;
                default: ir_error_not_expected();
            }
        }
        ir_check(!m_dim.is_undef() && !n_dim.is_undef() && !k_dim.is_undef())
                << "init_dpas: cannot initialize MNK dimensions.";
        int m_size = iter_tile_.at(m_dim);
        int n_size = iter_tile_.at(n_dim);
        int k_size = iter_tile_.at(k_dim);
        int sdepth = 8;
        int rcount = 8;
        int type_size = a_type_.size();
        ir_check(m_size % rcount == 0)
                << "init_dpas: M dimension size is invalid: " << m_size;
        ir_check(n_size % simd_ == 0)
                << "init_dpas: N dimension size is invalid: " << n_size;
        ir_check((k_size * type_size) % (sdepth * 4) == 0)
                << "init_dpas: K dimension size is invalid: " << k_size;

        auto _dpas = dpas_t::make(
                /*is_dpasw=*/false, simd_, sdepth, rcount, acc_type_, b_type_,
                a_type_);
        auto &dpas = _dpas.as<dpas_t>();
        a_inner_ = to_v2_layout(
                dpas.b_layout(), a_desc, std::vector<prb_dim_t> {k_dim, m_dim});
        b_inner_ = to_v2_layout(
                dpas.a_layout(), b_desc, std::vector<prb_dim_t> {n_dim, k_dim});
        c_inner_ = to_v2_layout(
                dpas.c_layout(), c_desc, std::vector<prb_dim_t> {n_dim, m_dim});
        return true;
    }

    static layout_t to_v2_layout(const jit::layout_t &layout,
            const layout_desc_t &desc, const std::vector<prb_dim_t> &dims) {
        layout_t ret(desc, layout.type());
        for (auto &b : layout.blocks()) {
            auto dim = dims[b.dim_idx];
            ret.add_block(dim, b.block);
        }
        return ret;
    }

    bool is_valid_ = false;
    fma_kind_t fma_ = fma_kind_t::undef;
    int simd_ = 0;
    prb_tile_t iter_tile_;
    dim_map_t<prb_dim_t, prb_dim_kind_t> bmnk_map_;
    type_t a_type_;
    type_t b_type_;
    type_t acc_type_;
    layout_t a_inner_;
    layout_t b_inner_;
    layout_t c_inner_;
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
        dim_mapper_manager_ = dim_mapper_manager_t(desc_.prop, desc_.spec_reqs);
    }

    void init_tiles() {
        tg_grid_ = create_thread_group_grid(desc_);
        thr_grid_ = create_thread_grid(desc_);
        for (auto &d : conv_index_dims(desc_.prop)) {
            bool is_loop = desc_.loop_desc.has(d);
            bool is_global_loop = desc_.loop_desc.is_global(d);
            int tg_tile = desc_.thread_group_tile.get(d, 1);
            int iter_tile = desc_.iter_tile.get(d, 1);
            auto thr_idx = thr_grid_.index_var(d);
            coord_info_.add_dim(d, is_loop, is_global_loop, tg_tile, thr_idx,
                    iter_tile, desc_.spec_reqs);
        }
    }

    void init_layouts() {
        auto src_layout = make_conv_layout(tensor_kind_t::src, desc_.src_tag,
                desc_.is_dw, desc_.spec_reqs);
        auto wei_layout = make_conv_layout(tensor_kind_t::wei, desc_.wei_tag,
                desc_.is_dw, desc_.spec_reqs);
        auto dst_layout = make_conv_layout(tensor_kind_t::dst, desc_.dst_tag,
                desc_.is_dw, desc_.spec_reqs);
        auto &a_mapper = dim_mapper_manager_.mapper(tensor_kind_t::a);
        auto &b_mapper = dim_mapper_manager_.mapper(tensor_kind_t::b);
        a_layout_ = pick_a(desc_.prop, src_layout, wei_layout, dst_layout);
        b_layout_ = pick_b(desc_.prop, src_layout, wei_layout, dst_layout);
        c_layout_ = pick_c(desc_.prop, src_layout, wei_layout, dst_layout);
        a_iter_view_ = view_t(
                a_mapper, a_layout_, coord_info_.iter_coord(), desc_.iter_tile);
        b_iter_view_ = view_t(
                b_mapper, b_layout_, coord_info_.iter_coord(), desc_.iter_tile);
    }

    dim_map_t<prb_dim_t, prb_dim_kind_t> to_bmnk_map() const {
        dim_map_t<prb_dim_t, prb_dim_kind_t> ret;
        for (auto &d : conv_index_dims(desc_.prop)) {
            ret[d] = to_gemm(d, desc_.prop).kind();
        }
        return ret;
    }

    bool init_info() {
        auto &a_mapper = dim_mapper_manager_.mapper(tensor_kind_t::a);
        auto &b_mapper = dim_mapper_manager_.mapper(tensor_kind_t::b);
        auto &c_mapper = dim_mapper_manager_.mapper(tensor_kind_t::c);
        auto &a_desc = a_mapper.layout_desc();
        auto &b_desc = b_mapper.layout_desc();
        auto &c_desc = c_mapper.layout_desc();
        mul_info_ = multiply_info_t(desc_.fma, desc_.simd, desc_.iter_tile,
                to_bmnk_map(), a_layout_.type(), a_desc, b_layout_.type(),
                b_desc, c_desc);
        if (!mul_info_) return false;
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
        plan.virt_grid = virt_grid_;
        plan.coord_info = coord_info_;
        ir_check(init_x2r_plan(plan.x2r));
        ir_check(init_prefetch_plan(plan.x2r, plan.virt_grid, plan.prefetch));
        ir_check(init_fma_plan(plan.x2r, plan.fma));
        ir_check(init_epilogue_plan(plan.fma, plan.epilogue));
        return true;
    }

    bool init_x_prefetch_plan(tensor_kind_t abc,
            const prb_coord_t<expr_t> &coord, const prb_tile_t &tile,
            const x2r_plan_t &x2r, virt_grid_t &virt_grid,
            send_plan_t &prefetch) const {
        auto &mapper = dim_mapper_manager_.mapper(abc);
        auto &layout = (abc == tensor_kind_t::a ? a_layout_ : b_layout_);
        grid_splitter_t grid_splitter;
        for (auto &d : thr_grid_.all_dims()) {
            grid_splitter.add(
                    thr_grid_.index_var(d), desc_.thread_group_tile[d]);
        }
        auto view = view_t::split(mapper, layout, coord, tile, grid_splitter);
        for (auto &kv : grid_splitter.virt_grid_idxs()) {
            virt_grid.add(kv.first, kv.second);
        }
        // Try 2D messages first.
        auto params = get_send_params(
                abc, send_op_t::prefetch, view, send_kind_t::_2d);
        prefetch = create_send_plan(params, view, /*allow_fail=*/true);
        if (!prefetch || !x2r.reqs().implies(prefetch.reqs())) {
            // If 2D failed, try compressed prefetch.
            params = get_send_params(abc, send_op_t::prefetch, view,
                    send_kind_t::compressed_prefetch);
            prefetch = create_send_plan(params, view, /*allow_fail=*/true);
            if (!prefetch) return false;
        }
        return true;
    }

    bool init_prefetch_plan(const x2r_plan_t &x2r, virt_grid_t &virt_grid,
            prefetch_plan_t &plan) const {
        if (desc_.prefetch.a) {
            ir_check(init_x_prefetch_plan(tensor_kind_t::a,
                    coord_info_.tg_iter_coord(), coord_info_.tg_iter_tile(),
                    x2r, virt_grid, plan.a_prefetch));
        }
        if (desc_.prefetch.b) {
            ir_check(init_x_prefetch_plan(tensor_kind_t::b,
                    coord_info_.tg_iter_coord(), coord_info_.tg_iter_tile(),
                    x2r, virt_grid, plan.b_prefetch));
        }
        return true;
    }

    bool init_x_g2r_plan(tensor_kind_t abc, const view_t &view,
            reorder_plan_t &reorder, layout_t &reg_layout,
            send_plan_t &load) const {
        auto params = get_send_params(abc, send_op_t::load, view);
        load = create_send_plan(params, view, /*allow_fail=*/true);
        ir_check(load) << "init_x_x2r_plan: cannot create send plan"
                       << std::endl
                       << params << std::endl
                       << ir_utils::add_tag("view", view.str());
        if (mul_info_.is_compatible(abc, load.reg_layout())) {
            reg_layout = load.reg_layout();
        } else {
            auto src = load.reg_layout();
            auto dst = mul_info_.to_compatible_layout(abc, load.reg_layout());
            reorder = reorder_plan_t(desc_.hw, src, dst);
            reg_layout = reorder.dst;
        }
        return true;
    }

    bool init_x2r_plan(x2r_plan_t &plan) const {
        ir_check(init_x_g2r_plan(tensor_kind_t::a, a_iter_view_, plan.a_reorder,
                plan.a_layout, plan.a_load));
        ir_check(init_x_g2r_plan(tensor_kind_t::b, b_iter_view_, plan.b_reorder,
                plan.b_layout, plan.b_load));
        return true;
    }

    bool init_fma_plan(const x2r_plan_t &x2r, fma_plan_t &plan) const {
        auto &a = x2r.a_layout;
        auto &b = x2r.b_layout;
        auto inst_tile = mul_info_.inst_tile();
        auto acc_layout = mul_info_.acc_layout(a, b, c_layout_);
        ir_check(!acc_layout.is_empty()) << "init_fma_plan: cannot vectorize.";
        plan.simd = desc_.simd;
        plan.fma = desc_.fma;
        plan.a_layout = a;
        plan.b_layout = b;
        plan.c_layout = acc_layout;
        plan.inst_tile = inst_tile;
        return true;
    }

    bool init_epilogue_plan(
            const fma_plan_t &fma, epilogue_plan_t &plan) const {
        auto &c_mapper = dim_mapper_manager_.mapper(tensor_kind_t::c);
        auto c_iter_view = view_t(
                c_mapper, c_layout_, coord_info_.iter_coord(), desc_.iter_tile);
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
        auto tg_iter_tile = coord_info_.tg_iter_tile();
        auto dim_sizes = view.base_layout().dim_sizes();
        for (int i = 0; i < mask_desc.nmasks(); i++) {
            prb_dim_t dim = mask_desc[i].dim;
            ir_assert(view.dim_mapper().has(dim));
            // Assume that dimensions with non-trivial mapping always require
            // masking.
            if (!view.dim_mapper().expr(dim).is_same(index_var(dim))) continue;
            // Assume global k-slciing implies masking.
            if (coord_info_.is_global_loop(dim)) continue;
            // Check if the mask can be proven with known dimension requirements.
            if (!reqs_.can_prove(dim_sizes.at(dim) % tg_iter_tile.at(dim) == 0))
                continue;
            // Mask is not required for this dimension.
            ret.push_back(dim);
        }
        return ret;
    }

    kernel_desc_t desc_;

    dim_mapper_manager_t dim_mapper_manager_;
    multiply_info_t mul_info_;
    coord_info_t coord_info_;
    grid_t tg_grid_;
    grid_t thr_grid_;
    virt_grid_t virt_grid_;
    layout_t a_layout_;
    layout_t b_layout_;
    layout_t c_layout_;
    view_t a_iter_view_;
    view_t b_iter_view_;
    prb_reqs_t reqs_;
};

prb_reqs_t plan_t::reqs() const {
    prb_reqs_t ret;
    ret.add(desc.spec_reqs.reqs());
    ret.add(prefetch.reqs());
    ret.add(x2r.reqs());
    ret.add(epilogue.c_store.reqs());
    ret.simplify();
    return ret;
}

plan_t create_conv_plan(const kernel_desc_t &desc) {
    if (!desc.is_supported()) return plan_t();
    ir_assert(!desc.spec_reqs.has_strategy())
            << "Kernel descriptor strategies are required to be specialized "
               "before plan creation";
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
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
