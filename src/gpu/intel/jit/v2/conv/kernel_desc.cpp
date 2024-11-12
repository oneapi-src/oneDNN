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
#include "gpu/intel/jit/v2/conv/tensor_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

std::string align_desc_t::align_t::str() const {
    std::string s = std::to_string(value);
    if (in_bytes) s += "b";
    return s;
}

void align_desc_t::align_t::parse(const std::string &_s) {
    auto s = _s;
    in_bytes = (!s.empty() && s.back() == 'b');
    if (in_bytes) s = s.substr(0, s.length() - 1);
    value = std::stoi(s);
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

void extensions_t::add(extension_kind_t kind) {
    kinds = static_cast<extension_kind_t>(
            static_cast<uint32_t>(kinds) | static_cast<uint32_t>(kind));
}

bool extensions_t::has(extension_kind_t kind) const {
    return static_cast<uint32_t>(kinds) & static_cast<uint32_t>(kind);
}

std::string extensions_t::str() const {
    if (kinds == extension_kind_t::undef) return "x";
    std::ostringstream oss;
    bool is_first = true;
    for (auto &p : extension_kind_names) {
        if (p.first == extension_kind_t::undef) continue;
        if (has(p.first)) {
            if (!is_first) oss << ",";
            oss << p.second;
            is_first = false;
        }
    }
    return oss.str();
}

void extensions_t::parse(std::istream &in) {
    auto s = jit::parse<std::string>(in);
    auto parts = gpu_utils::split(s, ",");
    kinds = extension_kind_t::undef;
    for (auto &p : parts) {
        add(to_enum<extension_kind_t>(p));
    }
}

extension_kind_t extensions_t::out_size(int size) {
    switch (size) {
        case 1: return extension_kind_t::out_b1;
        case 2: return extension_kind_t::out_b2;
        case 4: return extension_kind_t::out_b4;
        default: ir_error_not_expected();
    }
    return extension_kind_t::undef;
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
    dim_t b_iter = iter.at(pvars::b);
    dim_t m_iter = iter.at(pvars::m);
    dim_t n_iter = iter.at(pvars::n);
    dim_t k_iter = iter.at(pvars::k);
    dim_t a_elems = b_iter * m_iter * k_iter;
    dim_t b_elems = b_iter * k_iter * n_iter;
    dim_t c_elems = m_iter * n_iter;
    auto iter_outer_dim
            = (desc.iter_outer_tile.is_empty() ? pvar_t()
                                               : *desc.iter_outer_tile.begin());
    auto bmnk = to_gemm(iter_outer_dim, desc.prop);
    if (bmnk == pvars::m) {
        a_elems = utils::div_up(a_elems, desc.iter_outer_tile.elems());
    } else if (bmnk == pvars::n) {
        b_elems = utils::div_up(b_elems, desc.iter_outer_tile.elems());
    }
    dim_t a_size = a_elems * a_type_size;
    int a_reorder_size = 0;
    dim_t b_size = b_elems * b_type_size;
    int b_reorder_size = 0;
    dim_t c_size = c_elems * c_type_size;
    dim_t abc_size = 0;
    abc_size += a_size + a_reorder_size;
    abc_size += b_size + b_reorder_size;
    abc_size += c_size;
    return into<int>(abc_size);
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
}

void kernel_desc_t::finalize(const prb_reqs_t &final_reqs) {
    is_finalized = true;
    reqs.add(final_reqs);
}

bool fit_tag(tensor_kind_t abc, const kernel_desc_t &kernel_desc,
        const problem_t &prb, bool exact) {
    auto &desc_tag = kernel_desc.layout_tag(abc);
    auto &prb_tag = prb.layout_tag(abc);
    bool is_out = (abc == tensor_kind_t::c);
    auto &desc_type = desc_tag.type();
    auto &prb_type = prb_tag.type();
    bool type_ok = (desc_tag.type() == prb_tag.type());
    if (!exact) type_ok = (desc_type.size() == prb_type.size());
    if (!exact && is_out
            && kernel_desc.ext.has(extensions_t::out_size(prb_type.size())))
        type_ok = true;
    ir_check(type_ok
            && prb_tag.matches(desc_tag, prb.shape(), /*check_type=*/false))
            << to_string(abc) << " tag " << prb_tag
            << " does not match kernel descriptor tag " << desc_tag;
    return true;
}

bool fit_impl(const kernel_desc_t &desc, const problem_t &prb, bool exact) {
    ir_check(prb.prop() == desc.prop) << "Propagation kind does not match";
    ir_check(fit_tag(tensor_kind_t::a, desc, prb, exact));
    ir_check(fit_tag(tensor_kind_t::b, desc, prb, exact));
    ir_check(fit_tag(tensor_kind_t::c, desc, prb, exact));
    ir_check(prb.is_depthwise() == desc.is_dw)
            << "Mixing depthwise/non-depthwise descriptor and problem";
    if (exact) {
        ir_check(prb.with_bias_bwd_w() == desc.with_bias_bwd_w())
                << "Problem and descriptor bias reduction mismatch";
        ir_check(prb.with_bias_fwd() == desc.with_bias_fwd())
                << "Problem and descriptor bias mismatch";
    }
    if (prb.with_bias_bwd_w() != desc.with_bias_bwd_w()) {
        if (prb.with_bias_bwd_w()) {
            ir_check(desc.ext.has(extension_kind_t::bias))
                    << "Bias is not supported";
        }
    }
    ir_check(desc.reqs.fits(prb.shape() | prb.vars()));
    return true;
}

void fit_tag_to(
        tensor_kind_t abc, kernel_desc_t &kernel_desc, const problem_t &prb) {
    auto &desc_tag = const_cast<layout_tag_t &>(kernel_desc.layout_tag(abc));
    auto &prb_tag = prb.layout_tag(abc);
    if (desc_tag.type() != prb_tag.type()) {
        desc_tag = layout_tag_t(
                desc_tag.desc(), prb_tag.type(), desc_tag.raw_tag());
    }
}

void fit_to_impl(kernel_desc_t &desc, const problem_t &prb) {
    desc.reqs.substitute(prb.vars());
    fit_tag_to(tensor_kind_t::a, desc, prb);
    fit_tag_to(tensor_kind_t::b, desc, prb);
    fit_tag_to(tensor_kind_t::c, desc, prb);
    desc.bias_type = prb.bias_type();
}

bool kernel_desc_t::can_fit(const problem_t &prb) const {
    return fit_impl(*this, prb, /*exact=*/false);
}

void kernel_desc_t::fit_to(const problem_t &prb) {
    fit_to_impl(*this, prb);
    specialize(prb);
}

status_t kernel_desc_t::set_post_ops(
        const post_ops_t &attr_post_ops, const memory_desc_t *out_md) {
    // Adjust post-ops to be expressed in terms of the full layout, including
    // all spatial dimensions.
    int old_ndims = out_md->ndims;
    int new_ndims = 5;
    post_op::ndim_normalizer_t ndim_normalizer {2, new_ndims - old_ndims};
    return gpu_post_ops_t::make(post_ops, attr_post_ops, out_md,
            post_op::specializations_t(), ndim_normalizer);
}

bool kernel_desc_t::matches(const problem_t &prb) const {
    return fit_impl(*this, prb, /*exact=*/true);
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
    if (is_empty()) return "(empty)";
    std::ostringstream oss;
    oss << "Propagation:            " << jit::to_string(prop) << std::endl;
    oss << "Depthwise:              " << ir_utils::to_string(is_dw)
        << std::endl;
    oss << "Bias type:              " << bias_type << std::endl;
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
    oss << "Extensions:             " << ext.str() << std::endl;
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
    iface->add<PACK(bias_type)>("bias", "Bias type.");
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
            "2:2:2 (in elements).");
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
    iface->add<PACK(ext)>("ext",
            "Kernel extensions, comma-separated (e.g. "
            "bias,out1b,out2b,out4b).");

    parse_iface_t<kernel_desc_t>::entry_t po_entry;
    po_entry.name = "post_ops";
    po_entry.help = "Kernel post-ops.";
    po_entry._default = serialize_to_hex(gpu_post_ops_t());
    po_entry.stringify = [](std::ostream &out, const kernel_desc_t &parent) {
        out << serialize_to_hex(parent.post_ops);
    };
    po_entry.parse = [](std::istream &in, kernel_desc_t &parent) {
        auto s_data = stream_parse<std::string>(in);
        deserialize_from_hex(parent.post_ops, s_data);
    };
    iface->add(po_entry);
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
    auto sw_magic = var_t::make(type_t::u64(), "sw_magic");
    kernel_info.register_internal_arg(sw_magic);
}

void init_dispatch_kernel_info_div_magic(
        kernel_info_t &kernel_info, const pvar_tile_t &tg_dims, dim_t sw) {
    for (auto &d : tg_dims) {
        uint32_t size = into<uint32_t>(tg_dims.at(d));
        uint64_t magic = ir_utils::idiv_magicgu_packed(size);
        kernel_info.set_internal_arg(d.str() + "_grid_size", size);
        kernel_info.set_internal_arg(d.str() + "_magic", magic);
    }
    kernel_info.set_internal_arg(
            "sw_magic", ir_utils::idiv_magicgu_packed(into<uint32_t>(sw)));
}

arg_helper_t::arg_helper_t(const kernel_desc_t &desc) : desc_(desc) {}

int arg_helper_t::key(const std::string &name) const {
    if (name == "src") {
        if (is_fwd()) return DNNL_ARG_SRC;
        if (is_bwd_d()) return DNNL_ARG_DIFF_SRC;
        if (is_bwd_w()) return DNNL_ARG_SRC;
    } else if (name == "wei") {
        if (is_fwd()) return DNNL_ARG_WEIGHTS;
        if (is_bwd_d()) return DNNL_ARG_WEIGHTS;
        if (is_bwd_w()) return DNNL_ARG_DIFF_WEIGHTS;
    } else if (name == "dst") {
        if (is_fwd()) return DNNL_ARG_DST;
        if (is_bwd_d()) return DNNL_ARG_DIFF_DST;
        if (is_bwd_w()) return DNNL_ARG_DIFF_DST;
    } else if (name == "bia") {
        if (is_fwd()) return DNNL_ARG_BIAS;
        if (is_bwd_d()) return DNNL_ARG_BIAS;
        if (is_bwd_w()) return DNNL_ARG_DIFF_BIAS;
    }
    ir_error_not_expected();
    return DNNL_ARG_UNDEF;
}

bool arg_helper_t::is_input(const std::string &name) const {
    if (name == "src") return is_fwd() || is_bwd_w();
    if (name == "wei") return is_fwd() || is_bwd_d();
    if (name == "dst") return is_bwd_d() || is_bwd_w();
    if (name == "bia") return desc_.with_bias_fwd();
    ir_error_not_expected();
    return false;
}

bool arg_helper_t::is_output(const std::string &name) const {
    if (name == "src") return is_bwd_d();
    if (name == "wei") return is_bwd_w();
    if (name == "dst") return is_fwd();
    if (name == "bia") return desc_.with_bias_bwd_w();
    ir_error_not_expected();
    return false;
}

std::string arg_helper_t::post_op_name(size_t idx) const {
    ir_assert(idx < desc_.post_ops.len());
    auto &po = desc_.post_ops[idx];
    if (po.is_eltwise() || po.is_sum()) return "";
    if (po.is_binary()) return "binary_" + std::to_string(idx);
    ir_error_not_expected();
    return "";
}

int arg_helper_t::post_op_key(size_t idx) const {
    int _idx = static_cast<int>(idx);
    ir_assert(idx < desc_.post_ops.len());
    auto &po = desc_.post_ops[idx];
    if (po.is_eltwise() || po.is_sum()) return DNNL_ARG_UNDEF;
    if (po.is_binary() && po.as_binary().alg == alg_kind::binary_prelu) {
        return DNNL_ARG_ATTR_MULTIPLE_POST_OP(_idx) | DNNL_ARG_WEIGHTS;
    }
    if (po.is_binary()) {
        return DNNL_ARG_ATTR_MULTIPLE_POST_OP(_idx) | DNNL_ARG_SRC_1;
    }
    ir_error_not_expected();
    return -1;
}

tensor_config_t get_tensor_config(const kernel_desc_t &desc) {
    arg_helper_t h(desc);
    tensor_config_t tensor_cfg;
    for (auto *t : {"src", "wei", "dst", "bia"}) {
        bool is_input = h.is_input(t);
        bool is_output = h.is_output(t);
        if (!is_input && !is_output) continue;
        tensor_cfg.add_tensor(
                t, h.key(t), is_input, is_output, jit::layout_t());
    }
    for (size_t i = 0; i < desc.post_ops.len(); i++) {
        auto name = h.post_op_name(i);
        if (name.empty()) continue;
        tensor_cfg.add_tensor(name, h.post_op_key(i), /*is_input=*/true,
                /*is_output=*/false, jit::layout_t());
    }
    return tensor_cfg;
}

compute::range_t kernel_desc_t::local_range() const {
    auto thr_grid = create_thread_grid(*this);
    compute::range_t lws = compute::range_t::empty();
    for (size_t i = 0; i < compute::range_t::max_ndims; i++) {
        size_t tg_dim = thr_grid.size(i, thread_group_tile);
        lws[i] = tg_dim * (i == 0 ? into<size_t>(simd) : 1);
    }
    return lws;
}

status_t kernel_desc_t::init_kernel_info(kernel_info_t &kernel_info) const {
    auto tensor_config = get_tensor_config(*this);
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
    auto &shape = prb.shape();
    for (auto &d : shape) {
        kernel_info.set_internal_arg(d.str(), shape.at(d));
    }
    pvar_tile_t tg_dims;
    for (auto &d : tg_grid.all_dims()) {
        dim_t tg_size = desc.thread_group_tile.get(d, 1);
        dim_t iter_size = desc.iter_tile.get(d, 1);
        tg_dims[d] = utils::div_up(shape.at(d), tg_size * iter_size);
    }
    init_dispatch_kernel_info_div_magic(
            kernel_info, tg_dims, shape.at(pvars::sw));
    compute::range_t gws = compute::range_t::empty();
    compute::range_t lws = compute::range_t::empty();
    for (size_t i = 0; i < compute::range_t::max_ndims; i++) {
        size_t tg_dim = thr_grid.size(i, desc.thread_group_tile);
        lws[i] = tg_dim * (i == 0 ? into<size_t>(desc.simd) : 1);
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
