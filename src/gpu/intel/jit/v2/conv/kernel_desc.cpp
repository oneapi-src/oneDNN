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

#include "gpu/intel/jit/v2/conv/kernel_desc.hpp"

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/jit/codegen/kernel.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/utils/utils.hpp"
#include "gpu/intel/jit/v2/conv/kernel.hpp"
#include "gpu/intel/jit/v2/conv/plan.hpp"
#include "gpu/intel/jit/v2/conv/problem.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

load_desc_t str_to_load_desc(const std::string &s) {
    auto parts = gpu_utils::split(s, ",");
    load_desc_t ret;
    for (auto &p : parts) {
        auto p_parts = gpu_utils::split(p, ":");
        ir_assert(p_parts.size() == 2);
        auto tensor = p_parts[0];
        auto kind = p_parts[1];
        if (tensor == "a") {
            ret.a = str_to_send_kind(kind);
        } else if (tensor == "b") {
            ret.b = str_to_send_kind(kind);
        } else {
            ir_error_not_expected() << p;
        }
    }
    return ret;
}

store_desc_t str_to_store_desc(const std::string &s) {
    auto parts = gpu_utils::split(s, ",");
    store_desc_t ret;
    for (auto &p : parts) {
        auto p_parts = gpu_utils::split(p, ":");
        ir_assert(p_parts.size() == 2);
        auto tensor = p_parts[0];
        auto kind = p_parts[1];
        if (tensor == "c") {
            ret.c = str_to_send_kind(kind);
        } else {
            ir_error_not_expected() << p;
        }
    }
    return ret;
}

prefetch_desc_t str_to_prefetch_desc(const std::string &s) {
    auto parts = gpu_utils::split(s, ".");
    ir_assert(utils::one_of((int)parts.size(), 1, 2));
    ir_assert(parts[0].size() >= 2);
    int dist = std::stoi(parts[0].substr(1));
    ir_assert(dist >= 0);
    bool a = (dist > 0);
    bool b = (dist > 0);
    if (parts.size() == 2 && dist > 0) {
        ir_assert(utils::one_of(parts[1], "a", "b", "ab"));
        a = (parts[1].find("a") != std::string::npos);
        b = (parts[1].find("b") != std::string::npos);
    }
    prefetch_desc_t ret;
    ret.dist = dist;
    ret.a = a;
    ret.b = b;
    return ret;
}

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
            case prb_dim_kind_t::od: c = 'd'; break;
            case prb_dim_kind_t::kd: c = is_wei ? 'd' : 'z'; break;
            case prb_dim_kind_t::ih:
            case prb_dim_kind_t::oh: c = 'h'; break;
            case prb_dim_kind_t::kh: c = is_wei ? 'h' : 'y'; break;
            case prb_dim_kind_t::iw:
            case prb_dim_kind_t::ow: c = 'w'; break;
            case prb_dim_kind_t::kw: c = is_wei ? 'w' : 'x'; break;
            default: ir_error_not_expected();
        }
        letter_map[d] = c;
    }
    return layout_desc_t(letter_map);
}

layout_desc_t make_conv_algo_layout_desc(
        prop_kind_t prop, tensor_kind_t tensor_kind) {
    auto desc = make_conv_layout_desc(tensor_kind, /*src_dst_with_group=*/true);
    switch (tensor_kind) {
        case tensor_kind_t::wei: return desc;
        case tensor_kind_t::src:
            if (prop == prop_kind::backward_data) return desc;
            break;
        case tensor_kind_t::dst:
            if (prop != prop_kind::backward_data) return desc;
            break;
        default: ir_error_not_expected();
    }
    dim_map_t<prb_dim_t, char> letter_map;
    bool is_src = (tensor_kind == tensor_kind_t::src);
    prb_dim_t xd = (is_src ? prb_dims::od : prb_dims::id);
    prb_dim_t xh = (is_src ? prb_dims::oh : prb_dims::ih);
    prb_dim_t xw = (is_src ? prb_dims::ow : prb_dims::iw);
    for (int i = 0; i < desc.ndims(); i++) {
        auto d = desc.prb_dim(i);
        switch (d.kind()) {
            case prb_dim_kind_t::id:
            case prb_dim_kind_t::od:
                letter_map[xd] = 'd';
                letter_map[prb_dims::kd] = 'z';
                break;
            case prb_dim_kind_t::ih:
            case prb_dim_kind_t::oh:
                letter_map[xh] = 'h';
                letter_map[prb_dims::kh] = 'y';
                break;
            case prb_dim_kind_t::iw:
            case prb_dim_kind_t::ow:
                letter_map[xw] = 'w';
                letter_map[prb_dims::kw] = 'x';
                break;
            default: letter_map[d] = desc.layout_letter(d); break;
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
    ir_check(prop != prop_kind::undef)
            << "Invalid prop: " << ir_utils::to_string(prop);
    ir_check(!hw.is_undef()) << "Invalid hw: " << jit::to_string(hw.to_ngen());
    ir_check(fma != fma_kind_t::undef)
            << "Invalid fma: " << jit::to_string(fma);
    ir_check(simd != 0) << "Invalid simd: " << simd;
    ir_check(regs != 0) << "Invalid regs: " << regs;
    ir_check(is_tg_size_ok(*this))
            << "Invalid thread_group_tile: " << thread_group_tile;
    ir_check(is_grf_usage_ok(*this)) << "GRF usage exceeded";
    return true;
}

void kernel_desc_t::set(const std::string &s) {
    operator=(kernel_desc_t());
    if (s.empty()) return;
    auto iface = cli_iface();
    iface.parse(s, this);
    set_defaults();
}

void kernel_desc_t::set_defaults() {
    if (loop_desc.is_empty()) {
        switch (prop) {
            case prop_kind::forward_training:
            case prop_kind::forward_inference:
                loop_desc.add(prb_dims::kw);
                loop_desc.add(prb_dims::kh);
                loop_desc.add(prb_dims::kd);
                loop_desc.add(prb_dims::ic);
                break;
            case prop_kind::backward_data:
                loop_desc.add(prb_dims::kw);
                loop_desc.add(prb_dims::kh);
                loop_desc.add(prb_dims::kd);
                loop_desc.add(prb_dims::oc);
                break;
            case prop_kind::backward_weights:
                loop_desc.add(prb_dims::mb);
                loop_desc.add(prb_dims::ow);
                loop_desc.add(prb_dims::oh);
                loop_desc.add(prb_dims::od);
                break;
            default: ir_error_not_expected(); break;
        }
    }
}

void kernel_desc_t::finalize(const plan_t &plan) {
    is_finalized = true;
    reqs = plan.reqs();
}

std::string kernel_desc_t::cmd_str() const {
    return cli_iface().cmd_str(this);
}

std::string kernel_desc_t::str() const {
    std::ostringstream oss;
    oss << "Propagation:        " << ir_utils::to_string(prop) << std::endl;
    oss << "Depthwise:          " << ir_utils::to_string(is_dw) << std::endl;
    oss << "Source tag:         " << src_tag << std::endl;
    oss << "Weights tag:        " << wei_tag << std::endl;
    oss << "Destination tag:    " << dst_tag << std::endl;
    oss << "Specialization:     " << spec_reqs << std::endl;
    oss << "HW:                 " << ir_utils::to_lower(hw.str()) << std::endl;
    oss << "FMA kind:           " << to_string(fma) << std::endl;
    oss << "SIMD:               " << simd << std::endl;
    oss << "Registers:          " << regs << std::endl;
    oss << "Iteration tile:     " << iter_tile << std::endl;
    oss << "Thread group tile:  " << thread_group_tile << std::endl;
    oss << "Loop desc:          " << loop_desc << std::endl;
    oss << "Load:               " << load.str() << std::endl;
    oss << "Prefetch:           " << prefetch.str() << std::endl;
    oss << "Store:              " << store.str() << std::endl;
    if (reqs) oss << ir_utils::add_tag("Reqs", reqs.str()) << std::endl;
    oss << "Command:            " << cmd_str();
    return ir_utils::add_tag("Desc", oss.str());
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
        gpu_primitive_t *primitive, impl::engine_t *engine) const {
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
    return serialized_t::from_data(
            std::vector<uint8_t>(str.begin(), str.end()));
}

kernel_desc_t kernel_desc_t::deserialize(const serialized_t &s) {
    auto &data = s.get_data();
    std::string str(data.begin(), data.end());
    std::istringstream iss(str);
    kernel_desc_t desc;
    desc.deserialize(iss);
    return desc;
}

ir_utils::cli_iface_t<kernel_desc_t> kernel_desc_t::cli_iface() {
#define MAKE_SETTER(lhs, rhs) \
    *static_cast<void (*)(kernel_desc_t *, const std::string &)>( \
            [](kernel_desc_t *desc, const std::string &value) { \
                desc->lhs = rhs; \
            })
#define MAKE_GETTER(value) \
    *static_cast<std::string (*)(const kernel_desc_t *)>( \
            [](const kernel_desc_t *desc) { return value; })
    ir_utils::cli_iface_t<kernel_desc_t> iface;
    iface.add_arg("--prop", "Propagation kind (fwd, bwd_d or bwd_w).",
            MAKE_GETTER(ir_utils::to_string(desc->prop)),
            MAKE_SETTER(prop, ir_utils::str_to_prop_kind(value)));
    iface.add_arg("--dw",
            "Whether the problem is a depthwise convolution (0 or 1).",
            MAKE_GETTER(std::string(desc->is_dw ? "1" : "0")),
            MAKE_SETTER(is_dw, ir_utils::str_to_bool(value)));
    iface.add_arg("--src", "Source layout tag. Examples: axb:f32, aBx16b:f16).",
            MAKE_GETTER(desc->src_tag.str()),
            MAKE_SETTER(
                    src_tag, make_conv_layout_tag(tensor_kind_t::src, value)));
    iface.add_arg("--wei", "Weights layout tag (e.g. axcb:f32).",
            MAKE_GETTER(desc->wei_tag.str()),
            MAKE_SETTER(
                    wei_tag, make_conv_layout_tag(tensor_kind_t::wei, value)));
    iface.add_arg("--dst", "Destination layout tag (e.g. axb:f32).",
            MAKE_GETTER(desc->dst_tag.str()),
            MAKE_SETTER(
                    dst_tag, make_conv_layout_tag(tensor_kind_t::dst, value)));
    iface.add_arg("--spec-reqs",
            "Specialization requirements for problem dimensions (e.g. "
            "kd1kw1kh1 for convolution without filter).",
            MAKE_GETTER(desc->spec_reqs.str()),
            MAKE_SETTER(spec_reqs, str_to_spec_reqs(value)));
    iface.add_arg("--hw", "Hardware (xehpc).",
            MAKE_GETTER(ir_utils::to_lower(jit::to_string(desc->hw.to_ngen()))),
            MAKE_SETTER(hw, str_to_hw(value)));
    iface.add_arg("--fma", "FMA kind (mad).", MAKE_GETTER(to_string(desc->fma)),
            MAKE_SETTER(fma, str_to_fma_kind(value)));
    iface.add_arg("--simd", "SIMD size (16 or 32).",
            MAKE_GETTER(std::to_string(desc->simd)),
            MAKE_SETTER(simd, std::stoi(value)));
    iface.add_arg("--regs", "Number of registers (128 or 256).",
            MAKE_GETTER(std::to_string(desc->regs)),
            MAKE_SETTER(regs, std::stoi(value)));
    iface.add_arg("--iter", "Iteration tile (e.g. mb32ic16oc16).",
            MAKE_GETTER(desc->iter_tile.str()),
            MAKE_SETTER(iter_tile, str_to_prb_tile(value)));
    iface.add_arg("--tg", "Threadgroup tile (e.g. ow4oc4).",
            MAKE_GETTER(desc->thread_group_tile.str()),
            MAKE_SETTER(thread_group_tile, str_to_prb_tile(value)));
    iface.add_arg("--loop-desc",
            "Loop description, variables ordered from innermost to outermost "
            "(e.g. kw,kh,kd,ic).",
            MAKE_GETTER(desc->loop_desc.str()),
            MAKE_SETTER(loop_desc, str_to_loop_desc(value)));
    iface.add_arg("--load",
            "Load type (block, scattered [default], 2d) for A and B, e.g. "
            "a:2d,b:block.",
            MAKE_GETTER(desc->load.str()),
            MAKE_SETTER(load, str_to_load_desc(value)));
    iface.add_arg("--store",
            "Store type (block, scattered [default], 2d) for C,  e.g. c:2d.",
            MAKE_GETTER(desc->store.str()),
            MAKE_SETTER(store, str_to_store_desc(value)));
    iface.add_arg("--prefetch",
            "Prefetch description specifying distance and whether A/B are "
            "prefetched. Examples: x3 (distance is 3, both A/B are "
            "prefetched), x2.a (distance is 2, only A is prefetched), x0 (no "
            "prefetch, default).",
            MAKE_GETTER(desc->prefetch.str()),
            MAKE_SETTER(prefetch, str_to_prefetch_desc(value)));
    return iface;
#undef MAKE_SETTER
#undef MAKE_GETTER
}

void kernel_desc_t::show_help() {
    kernel_desc_t desc;
    desc.set("--help");
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
    compute::range_t gws = compute::range_t::empty();
    compute::range_t lws = compute::range_t::empty();
    for (size_t i = 0; i < compute::range_t::max_ndims; i++) {
        size_t tg_dim = thr_grid.size(i, desc.thread_group_tile);
        lws[i] = tg_dim * (i == 0 ? gpu_utils::into<size_t>(desc.simd) : 1);
        gws[i] = tg_grid.size(i, tg_dims) * lws[i];
    }
    auto nd_range = compute::nd_range_t(gws, lws);
    kernel_info.set_nd_range(nd_range);
    return status::success;
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
