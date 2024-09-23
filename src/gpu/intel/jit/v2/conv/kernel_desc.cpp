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

std::string align_desc_t::align_t::str() const {
    std::string s = (value == 0 ? "*" : std::to_string(value));
    if (in_bytes) s += "b";
    return s;
}

void align_desc_t::align_t::parse(const std::string &_s) {
    auto s = _s;
    in_bytes = (!s.empty() && s.back() == 'b');
    if (in_bytes) s = s.substr(0, s.length() - 1);
    value = (s == "*") ? 0 : std::stoi(s);
}

std::string align_desc_t::str() const {
    if (is_default()) return "x";
    std::vector<std::string> parts;
    parts.emplace_back(src.str());
    parts.emplace_back(wei.str());
    parts.emplace_back(dst.str());
    if (parts[0] == parts[1] && parts[1] == parts[2]) return parts[0];
    return gpu_utils::join(":", parts);
}

void align_desc_t::parse(std::istream &in) {
    operator=(align_desc_t());
    auto s = jit::parse<std::string>(in);
    if (s == "x") return;
    auto parts = gpu_utils::split(s, ":");
    if (parts.size() == 1) {
        parts.push_back(parts[0]);
        parts.push_back(parts[0]);
    }
    ir_assert(parts.size() == 3);
    src.parse(parts[0]);
    wei.parse(parts[1]);
    dst.parse(parts[2]);
}

void prefetch_desc_t::parse(std::istream &in) {
    operator=(prefetch_desc_t());
    std::string s;
    in >> s;
    auto parts = gpu_utils::split(s, ".");
    ir_assert(utils::one_of((int)parts.size(), 1, 2));
    ir_assert(parts[0].size() >= 2);
    dist = std::stoi(parts[0].substr(1));
    ir_assert(dist >= 0);
    a = (dist > 0);
    b = (dist > 0);
    if (parts.size() == 2 && dist > 0) {
        ir_assert(utils::one_of(parts[1], "a", "b", "ab"));
        a = (parts[1].find("a") != std::string::npos);
        b = (parts[1].find("b") != std::string::npos);
    }
}

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
        ir_assert(c != ' ');
        letter_map[d] = c;
    }
    return layout_desc_t(letter_map);
}

layout_desc_t make_conv_algo_layout_desc(
        prop_kind_t prop, tensor_kind_t tensor_kind) {
    auto desc = make_conv_layout_desc(tensor_kind, /*src_dst_with_group=*/true);
    switch (tensor_kind) {
        case tensor_kind_t::bia:
        case tensor_kind_t::wei: return desc;
        case tensor_kind_t::src:
            if (prop == prop_kind::backward_data) return desc;
            break;
        case tensor_kind_t::dst:
            if (prop != prop_kind::backward_data) return desc;
            break;
        default: ir_error_not_expected();
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
pvar_tile_t min_dims_tile(const problem_t &prb) {
    pvar_tile_t xd;
    xd[pvars::id] = xd[pvars::od] = xd[pvars::kd] = 1;
    xd[pvars::dd] = xd[pvars::pd] = 0;
    xd[pvars::sd] = 1;
    pvar_tile_t xhd = xd;
    xhd[pvars::ih] = xhd[pvars::oh] = xhd[pvars::kh] = 1;
    xhd[pvars::dh] = xhd[pvars::ph] = 0;
    xhd[pvars::sh] = 1;
    for (auto *t : {&xhd, &xd}) {
        bool ok = true;
        for (auto &d : *t) {
            if (prb.shape().at(d) != (*t).at(d)) {
                ok = false;
                break;
            }
        }
        if (ok) return *t;
    }
    return pvar_tile_t();
}

int estimate_grf_usage_bytes(const kernel_desc_t &desc) {
    int a_type_size = desc.a_type().size();
    int b_type_size = desc.b_type().size();
    int c_type_size = desc.c_type().size();
    auto iter = to_gemm(desc.iter_tile, desc.prop);
    int b_iter = iter.at(pvars::b);
    int m_iter = iter.at(pvars::m);
    int n_iter = iter.at(pvars::n);
    int k_iter = iter.at(pvars::k);
    int a_elems = b_iter * m_iter * k_iter;
    int b_elems = b_iter * k_iter * n_iter;
    int c_elems = m_iter * n_iter;
    auto iter_outer_dim
            = (desc.iter_outer_tile.is_empty() ? pvar_t()
                                               : *desc.iter_outer_tile.begin());
    auto bmnk = to_gemm(iter_outer_dim, desc.prop);
    if (bmnk == pvars::m) {
        a_elems = utils::div_up(a_elems, desc.iter_outer_tile.elems());
    } else if (bmnk == pvars::n) {
        b_elems = utils::div_up(b_elems, desc.iter_outer_tile.elems());
    }
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
    auto &iface = parse_iface();
    iface.parse(s, *this);
    set_defaults();
}

void kernel_desc_t::set_defaults() {
    if (loop_desc.is_empty()) {
        switch (prop) {
            case prop_kind::forward_training:
            case prop_kind::forward_inference:
                loop_desc.add(pvars::kw);
                loop_desc.add(pvars::kh);
                loop_desc.add(pvars::kd);
                loop_desc.add(pvars::ic);
                break;
            case prop_kind::backward_data:
                loop_desc.add(pvars::kw);
                loop_desc.add(pvars::kh);
                loop_desc.add(pvars::kd);
                loop_desc.add(pvars::oc);
                break;
            case prop_kind::backward_weights:
                loop_desc.add(pvars::mb);
                loop_desc.add(pvars::ow);
                loop_desc.add(pvars::oh);
                loop_desc.add(pvars::od);
                break;
            default: ir_error_not_expected(); break;
        }
    }
    if (is_dw) {
        reqs.set(pvars::ic, 1);
        reqs.set(pvars::oc, 1);
    }
    if (prop == prop_kind::backward_weights && with_bias) {
        bia_tag = make_conv_layout_tag(tensor_kind_t::bia, "a");
        bia_tag = layout_tag_t(
                bia_tag.desc(), dst_tag.type(), bia_tag.raw_tag());
    }
}

void kernel_desc_t::finalize(const prb_reqs_t &final_reqs) {
    is_finalized = true;
    reqs.add(final_reqs);
}

bool fit_tag(tensor_kind_t kind, const layout_tag_t &desc_tag,
        const layout_tag_t &prb_tag, const pvar_tile_t &shape, bool exact,
        bool adjust) {
    auto &desc_type = desc_tag.type();
    auto &prb_type = prb_tag.type();
    bool type_ok = (desc_tag.type() == prb_tag.type());
    if (!exact) type_ok = (desc_type.size() == prb_type.size());
    ir_check(type_ok && prb_tag.matches(desc_tag, shape, /*check_type=*/false))
            << to_string(kind) << " tag " << prb_tag
            << " does not match kernel descriptor tag " << desc_tag;
    if (desc_tag.type() != prb_tag.type() && adjust) {
        const_cast<layout_tag_t &>(desc_tag)
                = layout_tag_t(desc_tag.desc(), prb_type, desc_tag.raw_tag());
    }
    return true;
}

bool fit_impl(const kernel_desc_t &desc, const problem_t &prb, bool exact,
        bool adjust) {
    ir_check(prb.prop() == desc.prop) << "Propagation kind does not match";
    ir_check(fit_tag(tensor_kind_t::src, desc.src_tag, prb.src_tag(),
            prb.shape(), exact, adjust));
    ir_check(fit_tag(tensor_kind_t::wei, desc.wei_tag, prb.wei_tag(),
            prb.shape(), exact, adjust));
    ir_check(fit_tag(tensor_kind_t::dst, desc.dst_tag, prb.dst_tag(),
            prb.shape(), exact, adjust));
    ir_check(prb.is_depthwise() == desc.is_dw)
            << "Mixing depthwise/non-depthwise descriptor and problem";
    ir_check(prb.with_bias() == desc.with_bias)
            << "Problem and descriptor 'with_bias' field mismatch";
    ir_check(desc.reqs.fits(prb.shape()));
    return true;
}

bool kernel_desc_t::can_fit(const problem_t &prb) const {
    return fit_impl(*this, prb, /*exact=*/false, /*adjust=*/false);
}

void kernel_desc_t::fit_to(const problem_t &prb) {
    fit_impl(*this, prb, /*exact=*/false, /*adjust=*/true);
}

bool kernel_desc_t::matches(const problem_t &prb) const {
    return fit_impl(*this, prb, /*exact=*/true, /*adjust=*/false);
}

std::string kernel_desc_t::cmd_str() const {
    return parse_iface().cmd_str(*this);
}

std::string kernel_desc_t::brief_str() const {
    std::ostringstream oss;
    oss << jit::to_string(prop) << "_";
    oss << "i_" << iter_tile.str();
    oss << "_T_" << thread_group_tile.str();
    oss << "_p_" << prefetch.str();
    return oss.str();
}

std::string kernel_desc_t::str() const {
    std::ostringstream oss;
    oss << "Propagation:            " << jit::to_string(prop) << std::endl;
    oss << "Depthwise:              " << ir_utils::to_string(is_dw)
        << std::endl;
    oss << "With bias:              " << ir_utils::to_string(with_bias)
        << std::endl;
    oss << "Source tag:             " << src_tag << std::endl;
    oss << "Weights tag:            " << wei_tag << std::endl;
    oss << "Destination tag:        " << dst_tag << std::endl;
    oss << "HW:                     " << jit::to_string(hw.to_ngen())
        << std::endl;
    oss << "FMA kind:               " << to_string(fma) << std::endl;
    oss << "SIMD:                   " << simd << std::endl;
    oss << "Registers:              " << regs << std::endl;
    oss << "Iteration tile:         " << iter_tile << std::endl;
    oss << "Iteration outer tile:   " << iter_outer_tile << std::endl;
    oss << "Thread group tile:      " << thread_group_tile << std::endl;
    oss << "Loop desc:              " << loop_desc << std::endl;
    oss << "Use block 2D access:    " << ir_utils::to_string(use_2d_access)
        << std::endl;
    oss << "Align:                  " << align.str() << std::endl;
    oss << "Prefetch:               " << prefetch.str() << std::endl;
    if (reqs) oss << ir_utils::add_tag("Reqs", reqs.str()) << std::endl;
    oss << "Command:                " << cmd_str();
    return ir_utils::add_tag("Desc", oss.str());
}

void kernel_desc_t::init_parse_iface(parse_iface_t<kernel_desc_t> *iface) {
    iface->set_relaxed(true);
#define PACK(member) decltype(kernel_desc_t::member), &kernel_desc_t::member
    iface->add<PACK(hw_desc)>("hw", "Hardware (xehpc).", /*required=*/true);
    iface->add<PACK(prop)>("prop", "Propagation kind (fwd, bwd_d or bwd_w).",
            /*required=*/true);
    iface->add<PACK(is_dw)>(
            "dw", "Whether the problem is a depthwise convolution (0 or 1).");
    iface->add<PACK(with_bias)>(
            "with_bias", "Whether the problem has bias (0 or 1).");
    iface->add<PACK(src_tag)>("src",
            "Source layout tag. Examples: axb:f32, aBx16b:f16).",
            /*required=*/true);
    iface->add<PACK(wei_tag)>("wei", "Weights layout tag (e.g. axcb:f32).",
            /*required=*/true);
    iface->add<PACK(dst_tag)>("dst", "Destination layout tag (e.g. axb:f32).",
            /*required=*/true);
    iface->add<PACK(fma)>("fma", "FMA kind (e.g. mad).", /*required=*/true);
    iface->add<PACK(simd)>("simd", "SIMD size (16 or 32).", /*required=*/true);
    iface->add<PACK(regs)>(
            "regs", "Number of registers (128 or 256).", /*required=*/true);
    iface->add<PACK(iter_tile)>("iter", "Iteration tile (e.g. mb32ic16oc16).",
            /*required=*/true);
    iface->add<PACK(iter_outer_tile)>("iter_outer",
            "Outer iteration tile (e.g. mb2).",
            /*required=*/false);
    iface->add<PACK(thread_group_tile)>(
            "tg", "Threadgroup tile (e.g. ow4oc4).", /*required=*/true);
    iface->add<PACK(loop_desc)>("loop_desc",
            "Loop description, variables ordered from innermost to outermost "
            "(e.g. kw,kh,kd,ic).");
    iface->add<PACK(use_2d_access)>(
            "2d", "Whether to use block 2D messages for access.");
    iface->add<PACK(align)>("align",
            "Alignments in bytes/elements for the innermost dimension in "
            "source, weights and destination. Examples: 8b:8b:8b (in bytes), "
            "2:2:2 (in elements), *:*:* (for optimal values determined during "
            "kernel plan generation).");
    iface->add<PACK(prefetch)>("prefetch",
            "Prefetch description specifying distance and whether A/B are "
            "prefetched. Examples: x3 (distance is 3, both A/B are "
            "prefetched), x2.a (distance is 2, only A is prefetched), x0 (no "
            "prefetch, default).");
    iface->add<PACK(spec_strategy)>("spec_strategy",
            "Specialization strategy for problem dimensions (e.g. min_dims to "
            "eliminate unused spatial dimensions).");
    iface->add<PACK(reqs)>("reqs",
            "Dimension requirements, colon-separated (e.g. kd=1:mb>=16).");
#undef PACK

    iface->set_post_parse_func([](kernel_desc_t &desc) {
        desc.src_tag
                = make_conv_layout_tag(tensor_kind_t::src, desc.src_tag.str());
        desc.wei_tag
                = make_conv_layout_tag(tensor_kind_t::wei, desc.wei_tag.str());
        desc.dst_tag
                = make_conv_layout_tag(tensor_kind_t::dst, desc.dst_tag.str());
    });
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
        kernel_info_t &kernel_info, const pvar_tile_t &tg_dims) {
    for (auto &d : tg_dims) {
        uint32_t size = tg_dims.at(d);
        uint64_t magic = ir_utils::idiv_magicgu_packed(size);
        kernel_info.set_internal_arg(d.str() + "_grid_size", size);
        kernel_info.set_internal_arg(d.str() + "_magic", magic);
    }
}

compute::range_t kernel_desc_t::local_range() const {
    auto thr_grid = create_thread_grid(*this);
    compute::range_t lws = compute::range_t::empty();
    for (size_t i = 0; i < compute::range_t::max_ndims; i++) {
        size_t tg_dim = thr_grid.size(i, thread_group_tile);
        lws[i] = tg_dim * (i == 0 ? gpu_utils::into<size_t>(simd) : 1);
    }
    return lws;
}

status_t kernel_desc_t::init_kernel_info(kernel_info_t &kernel_info) const {
    auto tensor_config = get_tensor_config(prop, with_bias);
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
    return engine.create_kernel(&kernel, &ir_gen);
}

serialized_t kernel_desc_t::serialize() const {
    std::ostringstream oss;
    jit::stringify(oss, *this);
    auto str = oss.str();
    return serialized_t::from_data(
            std::vector<uint8_t>(str.begin(), str.end()));
}

kernel_desc_t kernel_desc_t::deserialize(const serialized_t &s) {
    auto &data = s.get_data();
    std::string str(data.begin(), data.end());
    std::istringstream iss(str);
    auto desc = jit::parse<kernel_desc_t>(iss);
    return desc;
}

const parse_iface_t<kernel_desc_t> &kernel_desc_t::parse_iface() {
    return parse_iface_helper_t<kernel_desc_t>::get();
}

void kernel_desc_t::show_help() {
    parse_iface().print_help();
}

grid_t create_thread_group_grid(const kernel_desc_t &desc) {
    grid_t grid("tg_idx");
    switch (desc.prop) {
        case prop_kind::forward:
            grid.add_mapping(pvars::oc, 0);
            grid.add_mapping(pvars::g, 1);
            grid.add_mapping(pvars::od, 1);
            grid.add_mapping(pvars::oh, 1);
            grid.add_mapping(pvars::ow, 1);
            grid.add_mapping(pvars::mb, 2);
            break;
        case prop_kind::backward_data:
            grid.add_mapping(pvars::ic, 0);
            grid.add_mapping(pvars::g, 1);
            grid.add_mapping(pvars::id, 1);
            grid.add_mapping(pvars::ih, 1);
            grid.add_mapping(pvars::iw, 1);
            grid.add_mapping(pvars::mb, 2);
            break;
        case prop_kind::backward_weights:
            grid.add_mapping(pvars::oc, 0);
            grid.add_mapping(pvars::ic, 1);
            grid.add_mapping(pvars::kd, 1);
            grid.add_mapping(pvars::kh, 1);
            grid.add_mapping(pvars::kw, 1);
            grid.add_mapping(pvars::g, 2);
            break;
        default: ir_error_not_expected();
    }
    return grid;
}

grid_t create_thread_grid(const kernel_desc_t &desc) {
    grid_t grid("thr_idx");
    switch (desc.prop) {
        case prop_kind::forward:
            grid.add_mapping(pvars::oc, 0);
            grid.add_mapping(pvars::mb, 1);
            grid.add_mapping(pvars::ow, 1);
            grid.add_mapping(pvars::ic, 2);
            break;
        case prop_kind::backward_data:
            grid.add_mapping(pvars::ic, 0);
            grid.add_mapping(pvars::mb, 1);
            grid.add_mapping(pvars::iw, 1);
            grid.add_mapping(pvars::oc, 2);
            break;
        case prop_kind::backward_weights:
            grid.add_mapping(pvars::oc, 0);
            grid.add_mapping(pvars::ic, 1);
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
    pvar_tile_t tg_dims;
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
