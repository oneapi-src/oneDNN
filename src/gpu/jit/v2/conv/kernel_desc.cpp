/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gpu/jit/v2/conv/kernel_desc.hpp"

#include "common/memory_desc_wrapper.hpp"
#include "gpu/jit/codegen/kernel.hpp"
#include "gpu/jit/ir/kernel_info.hpp"
#include "gpu/jit/v2/conv/kernel.hpp"
#include "gpu/jit/v2/conv/problem.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {
namespace conv {

layout_desc_t make_conv_layout_desc(
        tensor_kind_t tensor_kind, bool src_dst_with_group) {
    bool is_wei = (tensor_kind == tensor_kind_t::wei);
    dim_map_t<prb_dim_t, char> letter_map;
    for (auto &d : conv_layout_dims(tensor_kind, src_dst_with_group)) {
        char c = ' ';
        switch (d.kind()) {
            case prb_dim_kind_t::g: c = 'g'; break;
            case prb_dim_kind_t::mb: c = 'n'; break;
            case prb_dim_kind_t::ic: c = is_wei ? 'i' : 'c'; break;
            case prb_dim_kind_t::oc: c = is_wei ? 'o' : 'c'; break;
            case prb_dim_kind_t::id:
            case prb_dim_kind_t::od:
            case prb_dim_kind_t::kd: c = 'd'; break;
            case prb_dim_kind_t::ih:
            case prb_dim_kind_t::oh:
            case prb_dim_kind_t::kh: c = 'h'; break;
            case prb_dim_kind_t::iw:
            case prb_dim_kind_t::ow:
            case prb_dim_kind_t::kw: c = 'w'; break;
            default: ir_error_not_expected();
        }
        letter_map[d] = c;
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

std::string blocked_to_str_tag(const memory_desc_t &md) {
    auto &blk = md.format_desc.blocking;
    int ndims = md.ndims;
    std::vector<dim_t> full_inner_blks(ndims, 1);
    std::vector<std::string> parts;
    dim_t stride = 1;
    for (int i = blk.inner_nblks - 1; i >= 0; i--) {
        int idx = blk.inner_idxs[i];
        dim_t block = blk.inner_blks[i];
        char letter = 'a' + idx;
        parts.push_back(std::string(1, letter));
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
                char letter = (is_blocked ? 'A' : 'a') + j;
                parts.push_back(std::string(1, letter));
                stride *= rem_dims[j];
                seen[j] = true;
                found = true;
                break;
            }
        }
        if (!found) ir_error_not_expected();
    }
    std::ostringstream oss;
    for (int i = (int)parts.size() - 1; i >= 0; i--)
        oss << parts[i];
    return oss.str();
}

layout_raw_tag_t normalize_conv_tag(tensor_kind_t tensor_kind, int conv_ndims,
        const layout_raw_tag_t &tag) {
    bool is_wei = (tensor_kind == tensor_kind_t::wei);
    bool add_groups = (is_wei && tag.ndims() == conv_ndims);
    int old_sp_ndims = conv_ndims - 2;
    int new_sp_ndims = 3;
    layout_raw_tag_t ret = tag;
    if (add_groups) ret.add_dim('a', 0);
    char sp_letter = 'c' + ret.ndims() - conv_ndims;
    int entry_idx = ret.entry_index(sp_letter);
    for (int i = old_sp_ndims; i < new_sp_ndims; i++) {
        ret.add_dim(sp_letter, entry_idx);
    }
    return ret;
}

layout_tag_t make_conv_layout_tag(
        tensor_kind_t tensor_kind, int conv_ndims, const memory_desc_t &md) {
    bool is_any = (md.format_kind == format_kind::any);
    bool is_blocked = (md.format_kind == format_kind::blocked);
    ir_assert(is_any || is_blocked);
    auto desc = make_conv_layout_desc(tensor_kind);
    type_t type(md.data_type);
    if (is_any) return layout_tag_t(desc, type, layout_raw_tag_t::any());
    auto str_tag = blocked_to_str_tag(md);
    auto raw_tag = layout_raw_tag_t(str_tag);
    raw_tag = normalize_conv_tag(tensor_kind, conv_ndims, raw_tag);
    return layout_tag_t(desc, type, raw_tag);
}

int estimate_grf_usage_bytes(const kernel_desc_t &desc) {
    int a_type_size = desc.a_type().size();
    int b_type_size = desc.b_type().size();
    int c_type_size = desc.c_type().size();
    auto iter = to_gemm(desc.iter_tile, desc.prop);
    int b_iter = iter.at(prb_dims::b);
    int m_iter = iter.at(prb_dims::m);
    int n_iter = iter.at(prb_dims::n);
    int k_iter = iter.at(prb_dims::k);
    int a_elems = b_iter * m_iter * k_iter;
    int b_elems = b_iter * k_iter * n_iter;
    int c_elems = m_iter * n_iter;
    int a_size = a_elems * a_type_size;
    int a_reorder_size = 0;
    int b_size = b_elems * b_type_size;
    int b_reorder_size = 0;
    int c_size = c_elems * c_type_size;
    int abc_size = 0;
    abc_size += a_size + a_reorder_size;
    abc_size += b_size + b_reorder_size;
    abc_size += c_size;
    return abc_size;
}

bool is_tg_size_ok(const kernel_desc_t &desc) {
    int max_tg_size = desc.hw.max_tg_size(desc.regs, desc.simd);
    return desc.thread_group_tile.elems() <= max_tg_size;
}

bool is_grf_usage_ok(const kernel_desc_t &desc) {
    int size = estimate_grf_usage_bytes(desc);
    if (size > desc.hw.grf_size() * desc.regs) { return false; }
    return true;
}

bool kernel_desc_t::is_supported() const {
    if (!is_valid()) return false;
    if (!is_tg_size_ok(*this)) return false;
    if (!is_grf_usage_ok(*this)) return false;
    return true;
}

void init_kernel_info_div_magic(
        kernel_info_t &kernel_info, const kernel_desc_t &desc) {
    auto tg_grid = create_thread_group_grid(desc);
    for (auto &d : tg_grid.all_dims()) {
        auto var_size = var_t::make(type_t::u32(), d.str() + "_grid_size");
        auto var_magic = var_t::make(type_t::u64(), d.str() + "_magic");
        kernel_info.register_internal_arg(var_magic);
        kernel_info.register_internal_arg(var_size);
    }
}

void init_dispatch_kernel_info_div_magic(
        kernel_info_t &kernel_info, const prb_tile_t &tg_dims) {
    for (auto &d : tg_dims) {
        uint32_t size = tg_dims.at(d);
        uint64_t magic = ir_utils::idiv_magicgu_packed(size);
        kernel_info.set_internal_arg(d.str() + "_grid_size", size);
        kernel_info.set_internal_arg(d.str() + "_magic", magic);
    }
}

status_t kernel_desc_t::init_kernel_info(kernel_info_t &kernel_info) const {
    auto tensor_config = get_tensor_config(prop);
    for (auto &t : tensor_config.tensors()) {
        auto buf = make_buffer(t.name);
        kernel_info.register_user_arg(buf, t.arg_key, t.is_input);
    }
    for (auto &d : conv_dims()) {
        auto var = var_t::make(type_t::s32(), d.str());
        kernel_info.register_internal_arg(var);
    }

    init_kernel_info_div_magic(kernel_info, *this);
    return status::success;
}

status_t kernel_desc_t::create_kernel(compute::kernel_t &kernel,
        gpu_primitive_t *primitive, engine_t *engine) const {
    return primitive->create_kernel(
            engine, kernel, kernel_name().c_str(), *this);
}

status_t kernel_desc_t::create_generator(
        const compute::compute_engine_t &engine,
        compute::kernel_t &kernel) const {
    ir_generator_t<kernel_t> ir_gen(*this);
    return engine.create_kernel(&kernel, &ir_gen, cache_blob_t());
}

serialized_t kernel_desc_t::serialize() const {
    std::ostringstream oss;
    serialize(oss);
    auto str = oss.str();
    serialized_t s;
    s.set_data(std::vector<uint8_t>(str.begin(), str.end()));
    return s;
}

kernel_desc_t kernel_desc_t::deserialize(const serialized_t &s) {
    auto &data = s.get_data();
    std::string str(data.begin(), data.end());
    std::istringstream iss(str);
    kernel_desc_t desc;
    desc.deserialize(iss);
    return desc;
}

grid_t create_thread_group_grid(const kernel_desc_t &desc) {
    grid_t grid("tg_idx");
    switch (desc.prop) {
        case prop_kind::forward:
            grid.add_mapping(prb_dims::oc, 0);
            grid.add_mapping(prb_dims::g, 1);
            grid.add_mapping(prb_dims::od, 1);
            grid.add_mapping(prb_dims::oh, 1);
            grid.add_mapping(prb_dims::ow, 1);
            grid.add_mapping(prb_dims::mb, 2);
            break;
        case prop_kind::backward_data:
            grid.add_mapping(prb_dims::ic, 0);
            grid.add_mapping(prb_dims::g, 1);
            grid.add_mapping(prb_dims::id, 1);
            grid.add_mapping(prb_dims::ih, 1);
            grid.add_mapping(prb_dims::iw, 1);
            grid.add_mapping(prb_dims::mb, 2);
            break;
        case prop_kind::backward_weights:
            grid.add_mapping(prb_dims::oc, 0);
            grid.add_mapping(prb_dims::ic, 1);
            grid.add_mapping(prb_dims::kd, 1);
            grid.add_mapping(prb_dims::kh, 1);
            grid.add_mapping(prb_dims::kw, 1);
            grid.add_mapping(prb_dims::od, 1);
            grid.add_mapping(prb_dims::oh, 1);
            grid.add_mapping(prb_dims::ow, 1);
            grid.add_mapping(prb_dims::g, 2);
            grid.add_mapping(prb_dims::mb, 2);
            break;
        default: ir_error_not_expected();
    }
    return grid;
}

grid_t create_thread_grid(const kernel_desc_t &desc) {
    grid_t grid("thr_idx");
    switch (desc.prop) {
        case prop_kind::forward:
            grid.add_mapping(prb_dims::oc, 0);
            grid.add_mapping(prb_dims::mb, 1);
            grid.add_mapping(prb_dims::ow, 1);
            grid.add_mapping(prb_dims::ic, 2);
            break;
        case prop_kind::backward_data:
            grid.add_mapping(prb_dims::ic, 0);
            grid.add_mapping(prb_dims::mb, 1);
            grid.add_mapping(prb_dims::iw, 1);
            grid.add_mapping(prb_dims::oc, 2);
            break;
        case prop_kind::backward_weights:
            grid.add_mapping(prb_dims::oc, 0);
            grid.add_mapping(prb_dims::ic, 1);
            break;
        default: ir_error_not_expected();
    }
    for (auto &d : grid.all_dims()) {
        if (!desc.thread_group_tile.has(d)) grid.unset(d);
    }
    return grid;
}

status_t kernel_params_t::init_dispatch_kernel_info(
        kernel_info_t &kernel_info, const kernel_desc_base_t &_desc) const {
    auto &desc = static_cast<const kernel_desc_t &>(_desc);
    auto tg_grid = create_thread_group_grid(desc);
    auto thr_grid = create_thread_grid(desc);
    CHECK(desc.init_kernel_info(kernel_info));
    auto &dims = prb.shape();
    for (auto &d : dims) {
        kernel_info.set_internal_arg(d.str(), dims.at(d));
    }
    prb_tile_t tg_dims;
    for (auto &d : tg_grid.all_dims()) {
        int tg_size = desc.thread_group_tile.get(d, 1);
        int iter_size = desc.iter_tile.get(d, 1);
        tg_dims[d] = utils::div_up(dims.at(d), tg_size * iter_size);
    }
    init_dispatch_kernel_info_div_magic(kernel_info, tg_dims);
    size_t gws[3] = {};
    size_t lws[3] = {};
    for (int i = 0; i < 3; i++) {
        int tg_dim = thr_grid.size(i, desc.thread_group_tile);
        lws[i] = tg_dim * (i == 0 ? desc.simd : 1);
        gws[i] = tg_grid.size(i, tg_dims) * lws[i];
    }
    auto nd_range = compute::nd_range_t(gws, lws);
    kernel_info.set_nd_range(nd_range);
    return status::success;
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
