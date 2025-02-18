/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "gpu/intel/jit/v2/conv/tensor_utils.hpp"

#include "gpu/intel/jit/conv/problem.hpp"
#include "gpu/intel/jit/pass/simplify.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

layout_desc_t make_conv_layout_desc(
        tensor_kind_t tensor_kind, bool src_dst_with_group) {
    bool is_wei = (tensor_kind == tensor_kind_t::wei);
    pvar_map_t<char> letter_map;
    for (auto &d : conv_layout_dims(tensor_kind, src_dst_with_group)) {
        char c = ' ';
#define CASE(key, value) \
    if (d == pvars::key) c = (value)
        CASE(g, 'g');
        CASE(mb, 'n');
        CASE(ic, is_wei ? 'i' : 'c');
        CASE(oc, is_wei ? 'o' : 'c');
        CASE(id, 'd');
        CASE(od, 'd');
        CASE(kd, is_wei ? 'd' : 'z');
        CASE(ih, 'h');
        CASE(oh, 'h');
        CASE(kh, is_wei ? 'h' : 'y');
        CASE(iw, 'w');
        CASE(ow, 'w');
        CASE(kw, is_wei ? 'w' : 'x');
#undef CASE
        gpu_assert(c != ' ');
        letter_map[d] = c;
    }
    return layout_desc_t(letter_map);
}

layout_desc_t make_conv_algo_layout_desc(
        prop_kind_t prop, tensor_kind_t tensor_kind) {
    auto desc = make_conv_layout_desc(tensor_kind, /*src_dst_with_group=*/true);
    switch (tensor_kind) {
        case tensor_kind_t::bias:
        case tensor_kind_t::wei: return desc;
        case tensor_kind_t::src:
            if (prop == prop_kind::backward_data) return desc;
            break;
        case tensor_kind_t::dst:
            if (prop != prop_kind::backward_data) return desc;
            break;
        default: gpu_error_not_expected();
    }
    pvar_map_t<char> letter_map;
    bool is_src = (tensor_kind == tensor_kind_t::src);
    pvar_t xd = (is_src ? pvars::od : pvars::id);
    pvar_t xh = (is_src ? pvars::oh : pvars::ih);
    pvar_t xw = (is_src ? pvars::ow : pvars::iw);
    for (int i = 0; i < desc.ndims(); i++) {
        auto d = desc.prb_dim(i);
        if (utils::one_of(d, pvars::id, pvars::od)) {
            letter_map[xd] = 'd';
            letter_map[pvars::kd] = 'z';
        } else if (utils::one_of(d, pvars::ih, pvars::oh)) {
            letter_map[xh] = 'h';
            letter_map[pvars::kh] = 'y';
        } else if (utils::one_of(d, pvars::iw, pvars::ow)) {
            letter_map[xw] = 'w';
            letter_map[pvars::kw] = 'x';
        } else {
            letter_map[d] = desc.layout_letter(d);
        }
    }
    return layout_desc_t(letter_map);
}

layout_tag_t make_conv_layout_tag(
        tensor_kind_t tensor_kind, const std::string &s) {
    if (s.empty()) return layout_tag_t();
    bool is_wei = (tensor_kind == tensor_kind_t::wei);
    auto desc = make_conv_layout_desc(tensor_kind);
    auto parts = gpu_utils::split(s, ":");
    auto type = (parts.size() > 1 ? type_t(parts[1]) : type_t::f32());
    auto str_tag = desc.to_abx_tag(parts[0]);
    auto raw_tag = layout_raw_tag_t(str_tag, is_wei ? 6 : 5);
    return layout_tag_t(desc, type, raw_tag);
}

layout_tag_t append_groups(
        tensor_kind_t tensor_kind, const layout_tag_t &layout_tag, bool is_dw) {
    if (layout_tag.is_any()) return layout_tag;
    bool is_src = (tensor_kind == tensor_kind_t::src);
    bool is_dst = (tensor_kind == tensor_kind_t::dst);
    bool is_bias = (tensor_kind == tensor_kind_t::bias);
    if (!is_src && !is_dst && !is_bias) return layout_tag;
    auto xc_dim = (is_src ? pvars::ic : pvars::oc);
    auto xc_letter = dim_idx::as_tag(layout_tag.desc().dim_index(xc_dim));
    auto new_g_letter = xc_letter;
    auto new_xc_letter = into<char>(xc_letter + 1);
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

uint32_t append_groups(tensor_kind_t tensor_kind, uint32_t mask, bool is_dw) {
    bool is_src = (tensor_kind == tensor_kind_t::src);
    bool is_dst = (tensor_kind == tensor_kind_t::dst);
    bool is_bias = (tensor_kind == tensor_kind_t::bias);
    if (!is_src && !is_dst && !is_bias) return mask;
    uint32_t c_mask = (mask >> 1) & 0x1;
    uint32_t n_mask = mask & 0x1;
    uint32_t dhw_mask = (mask >> 2);
    return n_mask | (c_mask << 1) | (c_mask << 2) | (dhw_mask << 3);
}

layout_t make_conv_layout(tensor_kind_t tensor_kind, const layout_tag_t &_tag,
        bool is_dw, const prb_reqs_t &reqs, uint32_t _mask) {
    auto tag = append_groups(tensor_kind, _tag, is_dw);
    auto mask = append_groups(tensor_kind, _mask, is_dw);
    layout_t ret(tag.desc(), tag.type());
    pvar_map_t<int> blocks;
    auto rem_size = [&](const pvar_t &dim, const pvar_map_t<int> &blocks) {
        uint32_t dim_mask = (mask & (1 << tag.desc().dim_index(dim)));
        if (dim_mask == 0) return expr_t(1);
        auto dim_size = reqs.to_expr(dim);
        if (!blocks.has(dim)) return dim_size;
        return div_up(dim_size, blocks[dim]);
    };
    auto &entries = tag.raw_tag().entries();
    for (auto it = entries.rbegin(); it != entries.rend(); it++) {
        pvar_t dim = tag.desc().prb_dim(it->index());
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

std::string blocked_to_str_tag(const memory_desc_t &md) {
    auto &blk = md.format_desc.blocking;
    int ndims = md.ndims;
    std::vector<dim_t> full_inner_blks(ndims, 1);
    std::vector<std::string> parts;
    dim_t stride = 1;
    for (int i = blk.inner_nblks - 1; i >= 0; i--) {
        dim_idx_t idx = into<dim_idx_t>(blk.inner_idxs[i]);
        dim_t block = blk.inner_blks[i];
        parts.push_back(std::string(1, dim_idx::as_tag(idx)));
        parts.push_back(std::to_string(block));
        full_inner_blks[idx] *= block;
        stride *= block;
    }
    std::vector<bool> seen(ndims);
    dims_t rem_dims;
    for (int i = 0; i < ndims; i++) {
        rem_dims[i] = md.padded_dims[i] / full_inner_blks[i];
    }
    for (int i = 0; i < ndims; i++) {
        bool found = false;
        dim_t min_dim = std::numeric_limits<dim_t>::max();
        for (int j = 0; j < ndims; j++) {
            if (!seen[j] && blk.strides[j] == stride) {
                min_dim = std::min(min_dim, rem_dims[j]);
            }
        }
        for (int j = ndims - 1; j >= 0; j--) {
            if (!seen[j] && blk.strides[j] == stride) {
                // Size-one blocks have to be added first.
                if (min_dim == 1 && rem_dims[j] != min_dim) continue;
                bool is_blocked = (full_inner_blks[j] != 1);
                parts.push_back(std::string(1, dim_idx::as_tag(j, is_blocked)));
                stride *= rem_dims[j];
                seen[j] = true;
                found = true;
                break;
            }
        }
        if (!found) gpu_error_not_expected();
    }
    std::ostringstream oss;
    for (int i = (int)parts.size() - 1; i >= 0; i--)
        oss << parts[i];
    return oss.str();
}

layout_raw_tag_t normalize_conv_tag(tensor_kind_t tensor_kind,
        dim_idx_t conv_ndims, const layout_raw_tag_t &tag) {
    bool is_wei = (tensor_kind == tensor_kind_t::wei);
    bool add_groups = (is_wei && tag.ndims() == conv_ndims);
    int old_sp_ndims = conv_ndims - 2;
    int new_sp_ndims = 3;
    layout_raw_tag_t ret = tag;
    if (add_groups) ret.add_dim('a', 0);
    char sp_letter = dim_idx::as_tag(2u + ret.ndims() - conv_ndims);
    int entry_idx = ret.entry_index(sp_letter);
    for (int i = old_sp_ndims; i < new_sp_ndims; i++) {
        ret.add_dim(sp_letter, entry_idx);
    }
    return ret;
}

layout_tag_t make_conv_layout_tag(tensor_kind_t tensor_kind,
        dim_idx_t conv_ndims, const memory_desc_t &md) {
    bool is_any = (md.format_kind == format_kind::any);
    bool is_blocked = (md.format_kind == format_kind::blocked);
    gpu_assert(is_any || is_blocked);
    auto desc = make_conv_layout_desc(tensor_kind);
    type_t type(md.data_type);
    if (is_any) return layout_tag_t(desc, type, layout_raw_tag_t::any());
    auto str_tag = blocked_to_str_tag(md);
    auto raw_tag = layout_raw_tag_t(str_tag);
    raw_tag = normalize_conv_tag(tensor_kind, conv_ndims, raw_tag);
    return layout_tag_t(desc, type, raw_tag);
}

dim_mapper_manager_t::dim_mapper_manager_t(
        prop_kind_t prop, const prb_reqs_t &reqs)
    : prop_(prop), reqs_(reqs) {
    src_mapper_ = init_src_mapper();
    wei_mapper_ = init_wei_mapper();
    dst_mapper_ = init_dst_mapper();
    bias_mapper_ = init_bias_mapper();
}

const dim_mapper_t &dim_mapper_manager_t::mapper(tensor_kind_t tensor) const {
    switch (tensor) {
        case tensor_kind_t::src: return src_mapper_;
        case tensor_kind_t::wei: return wei_mapper_;
        case tensor_kind_t::dst: return dst_mapper_;
        case tensor_kind_t::a:
            return mapper(pick_a(prop_, tensor_kind_t::src, tensor_kind_t::wei,
                    tensor_kind_t::dst));
        case tensor_kind_t::b:
            return mapper(pick_b(prop_, tensor_kind_t::src, tensor_kind_t::wei,
                    tensor_kind_t::dst));
        case tensor_kind_t::c:
            return mapper(pick_c(prop_, tensor_kind_t::src, tensor_kind_t::wei,
                    tensor_kind_t::dst));
        case tensor_kind_t::bias: return bias_mapper_;
        default: gpu_error_not_expected();
    }
    return src_mapper_;
}

dim_mapper_t dim_mapper_manager_t::init_src_mapper() const {
    auto pd = reqs_.to_expr(pvars::pd);
    auto ph = reqs_.to_expr(pvars::ph);
    auto pw = reqs_.to_expr(pvars::pw);
    auto sd = reqs_.to_expr(pvars::sd);
    auto sh = reqs_.to_expr(pvars::sh);
    auto sw = reqs_.to_expr(pvars::sw);
    auto dd = reqs_.to_expr(pvars::dd);
    auto dh = reqs_.to_expr(pvars::dh);
    auto dw = reqs_.to_expr(pvars::dw);
    dim_mapper_t mapper;
    mapper.set_dim(pvars::mb);
    mapper.set_dim(pvars::g);
    mapper.set_dim(pvars::ic);
    if (utils::one_of(prop_, prop_kind::forward, prop_kind::backward_weights)) {
        auto dd_inc = const_fold(dd + 1);
        auto dh_inc = const_fold(dh + 1);
        auto dw_inc = const_fold(dw + 1);
        auto neg_pd = const_fold(-pd);
        auto neg_ph = const_fold(-ph);
        auto neg_pw = const_fold(-pw);
        mapper.set_dim(pvars::id,
                simplify_rewrite(sd * od_idx + neg_pd + kd_idx * dd_inc), true);
        mapper.set_dim(pvars::ih,
                simplify_rewrite(sh * oh_idx + neg_ph + kh_idx * dh_inc), true);
        mapper.set_dim(pvars::iw,
                simplify_rewrite(sw * ow_idx + neg_pw + kw_idx * dw_inc), true);
    } else {
        mapper.set_dim(pvars::id);
        mapper.set_dim(pvars::ih);
        mapper.set_dim(pvars::iw);
    }
    mapper.set_layout_desc(
            make_conv_algo_layout_desc(prop_, tensor_kind_t::src));
    return mapper;
}

dim_mapper_t dim_mapper_manager_t::init_wei_mapper() const {
    dim_mapper_t mapper;
    mapper.set_dim(pvars::g);
    mapper.set_dim(pvars::oc);
    mapper.set_dim(pvars::ic);
    mapper.set_dim(pvars::kd);
    mapper.set_dim(pvars::kh);
    mapper.set_dim(pvars::kw);
    mapper.set_layout_desc(
            make_conv_algo_layout_desc(prop_, tensor_kind_t::wei));
    return mapper;
}

dim_mapper_t dim_mapper_manager_t::init_bias_mapper() const {
    dim_mapper_t mapper;
    mapper.set_dim(pvars::g);
    mapper.set_dim(pvars::oc);
    mapper.set_layout_desc(
            make_conv_algo_layout_desc(prop_, tensor_kind_t::bias));
    return mapper;
}

dim_mapper_t dim_mapper_manager_t::init_dst_mapper() const {
    dim_mapper_t mapper;
    mapper.set_dim(pvars::mb);
    mapper.set_dim(pvars::g);
    mapper.set_dim(pvars::oc);
    if (utils::one_of(prop_, prop_kind::forward, prop_kind::backward_weights)) {
        mapper.set_dim(pvars::od);
        mapper.set_dim(pvars::oh);
        mapper.set_dim(pvars::ow);
    } else {
        auto pd = reqs_.to_expr(pvars::pd);
        auto ph = reqs_.to_expr(pvars::ph);
        auto pw = reqs_.to_expr(pvars::pw);
        auto sd = reqs_.to_expr(pvars::sd);
        auto sh = reqs_.to_expr(pvars::sh);
        auto sw = reqs_.to_expr(pvars::sw);
        auto dd = reqs_.to_expr(pvars::dd);
        auto dh = reqs_.to_expr(pvars::dh);
        auto dw = reqs_.to_expr(pvars::dw);

        auto dd_inc = const_fold(dd + 1);
        auto dh_inc = const_fold(dh + 1);
        auto dw_inc = const_fold(dw + 1);

        mapper.set_dim(pvars::od,
                simplify_rewrite((id_idx + pd - (kd_idx * dd_inc)) / sd), true);
        mapper.set_dim(pvars::oh,
                simplify_rewrite((ih_idx + ph - (kh_idx * dh_inc)) / sh), true);
        mapper.set_dim(pvars::ow,
                simplify_rewrite((iw_idx + pw - (kw_idx * dw_inc)) / sw), true);
    }
    mapper.set_layout_desc(
            make_conv_algo_layout_desc(prop_, tensor_kind_t::dst));
    return mapper;
}

dim_mapper_t extend_mapper(
        const dim_mapper_t &mapper, const pvar_t &extra_dim, char letter) {
    auto new_mapper = mapper;
    new_mapper.set_dim(extra_dim);
    auto &desc = mapper.layout_desc();
    auto new_letter_map = desc.letter_map();
    new_letter_map[extra_dim] = letter;
    auto new_desc = layout_desc_t(new_letter_map);
    new_mapper.set_layout_desc(new_desc);
    return new_mapper;
}

std::vector<pvar_t> skip_mask(
        const view_t &view, const pvar_tile_t &tile, const prb_reqs_t &reqs) {
    std::vector<pvar_t> ret;
    auto &mask_desc = view.mask_desc();
    auto dim_sizes = view.base_layout().dim_sizes();
    for (int i = 0; i < mask_desc.nmasks(); i++) {
        pvar_t dim = mask_desc[i].dim;
        gpu_assert(view.dim_mapper().has(dim));
        // Assume that dimensions with non-trivial mapping always require
        // masking.
        if (!view.dim_mapper().expr(dim).is_same(dim.index_var())) continue;
        // Check if the mask can be proven with known dimension requirements.
        if (!reqs.can_prove(dim_sizes.at(dim) % tile.at(dim) == 0)) continue;
        // Mask is not required for this dimension.
        ret.push_back(dim);
    }
    return ret;
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
