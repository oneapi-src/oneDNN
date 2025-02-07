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

#include "gpu/intel/jit/v2/conv/kernel_desc.hpp"

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/jit/codegen/kernel.hpp"
#include "gpu/intel/jit/ir/config.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/utils/utils.hpp"
#include "gpu/intel/jit/v2/conv/bridge.hpp"
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

pvar_tile_t get_dims_tile(const problem_t &prb, specialization_mode_t mode) {
    switch (mode) {
        case specialization_mode_t::min_dims: return min_dims_tile(prb);
        case specialization_mode_t::max: return prb.shape();
        default: gpu_error_not_expected();
    }
    return {};
}

void specialization_t::specialize(const problem_t &prb) {
    auto t = get_dims_tile(prb, mode);
    for (auto &d : t) {
        gpu_assert(!dim_values.has(d) || dim_values[d] == t[d]);
        dim_values[d] = t[d];
    }
    mode = specialization_mode_t::none;
    canonicalize();
}

prb_reqs_t specialization_t::reqs() const {
    gpu_assert(!is_dynamic()) << "Must be specialized before this call";
    prb_reqs_t reqs;
    reqs.add(dim_values);
    for (auto &d : dim_mods) {
        reqs.add(d.var() % dim_mods[d] == 0);
    }
    return reqs;
}

std::string specialization_t::str() const {
    std::vector<std::string> parts;
    std::string s_dims;
    if (!dim_values.is_empty()) s_dims = dim_values.str();
    for (auto &d : dim_mods) {
        s_dims += d.str() + "@" + std::to_string(dim_mods[d]);
    }
    if (!s_dims.empty()) parts.emplace_back(s_dims);
    if (mode != specialization_mode_t::none)
        parts.emplace_back(to_string(mode));
    return gpu_utils::join(":", parts);
}

void specialization_t::parse(std::istream &in) {
    auto s = jit::parse<std::string>(in);
    auto parts = gpu_utils::split(s, ":");
    for (auto &p : parts) {
        bool found = false;
        for (auto &kv : specialization_mode_names) {
            if (p == kv.second) {
                gpu_assert(mode == specialization_mode_t::none);
                mode = kv.first;
                found = true;
                break;
            }
        }
        if (found) continue;
        auto tile = jit::parse<pvar_tile_t>(p);
        for (auto &d : tile) {
            if (d.name().back() == '@') {
                dim_mods[pvar_t(d.name().substr(0, d.name().size() - 1))]
                        = tile[d];
            } else {
                dim_values[d] = tile[d];
            }
        }
    }
    canonicalize();
}

void specialization_t::canonicalize() {
    for (auto &d : dim_values) {
        if (dim_mods.has(d)) {
            gpu_assert(dim_values[d] % dim_mods[d] == 0)
                    << "Incompatible dim_values/dim_mods: " << dim_values.str()
                    << "/" << dim_mods.str();
            dim_mods.unset(d);
        }
    }
}

void prefetch_desc_t::parse(std::istream &in) {
    operator=(prefetch_desc_t());
    std::string s;
    in >> s;
    auto parts = gpu_utils::split(s, ".");
    gpu_assert(utils::one_of((int)parts.size(), 1, 2));
    gpu_assert(parts[0].size() >= 2);
    dist = std::stoi(parts[0].substr(1));
    gpu_assert(dist >= 0);
    a = (dist > 0);
    b = (dist > 0);
    if (parts.size() == 2 && dist > 0) {
        gpu_assert(utils::one_of(parts[1], "a", "b", "ab"));
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
        default: gpu_error_not_expected();
    }
    return extension_kind_t::undef;
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

bool is_tg_size_ok(const kernel_desc_t &desc, const hw_t &hw) {
    int max_tg_size = hw.max_tg_size(desc.regs, desc.simd);
    return desc.thread_group_tile.elems() <= max_tg_size;
}

bool is_grf_usage_ok(const kernel_desc_t &desc) {
    int size = estimate_grf_usage_bytes(desc);
    if (size > desc.hw_desc.grf_size() * desc.regs) { return false; }
    return true;
}

prb_reqs_t kernel_desc_t::reqs() const {
    return generate_2d_reqs(*this);
}

bool kernel_desc_t::is_supported(const hw_t &hw, const problem_t *prb) const {
    gpu_check(prop != prop_kind::undef)
            << "Invalid prop: " << ir_utils::to_string(prop);
    gpu_check(!prb || (hw_desc.hw == prb->hw().to_ngen()))
            << "HW mismatch, desc: " << jit::to_string(hw_desc.hw)
            << ", problem: " << jit::to_string(prb->hw().to_ngen());
    gpu_check(fma != fma_kind_t::undef)
            << "Invalid fma: " << jit::to_string(fma);
    gpu_check(simd != 0) << "Invalid simd: " << simd;
    gpu_check(regs != 0) << "Invalid regs: " << regs;
    gpu_check(is_tg_size_ok(*this, hw))
            << "Invalid thread_group_tile: " << thread_group_tile;
    if (use_stream_k) {
        gpu_check(c_type() == accumulator_type(a_type(), b_type()))
                << "Output/accumulator types must match for Stream-K";
    }
    gpu_check(is_grf_usage_ok(*this)) << "GRF usage exceeded";
    if (prb) gpu_check(matches(*prb)) << "Descriptor does not match problem";
    return true;
}

void kernel_desc_t::set(const std::string &s) {
    operator=(kernel_desc_t());
    if (s.empty()) return;
    auto &iface = parse_iface();
    parse_result_t result;
    iface.parse(s, *this, &result);
    if (!result.is_set("--iter")) {
        gpu_info() << "Error: missing --iter parameter in kernel descriptor.";
        gpu_error_not_expected();
    }
    set_defaults();
}

loop_desc_t default_loop_desc(prop_kind_t prop) {
    loop_desc_t loop_desc;
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
            loop_desc.add(pvars::ow);
            loop_desc.add(pvars::oh);
            loop_desc.add(pvars::od);
            loop_desc.add(pvars::mb);
            break;
        default: gpu_error_not_expected(); break;
    }
    return loop_desc;
}

void kernel_desc_t::set_defaults() {
    src_tag = make_conv_layout_tag(tensor_kind_t::src, src_tag.str());
    wei_tag = make_conv_layout_tag(tensor_kind_t::wei, wei_tag.str());
    dst_tag = make_conv_layout_tag(tensor_kind_t::dst, dst_tag.str());
    if (loop_desc.is_empty()) loop_desc = default_loop_desc(prop);
    if (is_dw) {
        spec.dim_values[pvars::ic] = 1;
        spec.dim_values[pvars::oc] = 1;
    }
    if (prop == prop_kind::backward_data) {
        // XXX: No stride support in backward by data yet.
        spec.dim_values[pvars::sw] = 1;
        spec.dim_values[pvars::sh] = 1;
        spec.dim_values[pvars::sd] = 1;
    }
}

bool is_compatible(tensor_kind_t abc, const kernel_desc_t &kernel_desc,
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
    if (!type_ok && is_out && kernel_desc.use_stream_k) type_ok = true;
    gpu_check(type_ok) << to_string(abc) << " tag " << prb_tag
                       << " does not match kernel descriptor tag " << desc_tag;
    return true;
}

bool is_compatible(const hw_desc_t &hw_desc, const hw_t &hw, bool exact) {
    if (!exact && hw != hw_desc.hw) {
        switch (hw_desc.hw) {
            case ngen::HW::XeHPC:
                return utils::one_of(
                        hw.to_ngen(), ngen::HW::Xe2, ngen::HW::Xe3);
            default: break;
        }
    }
    return hw_desc.hw == hw.to_ngen();
}

bool is_compatible(
        const kernel_desc_t &desc, const problem_t &prb, bool exact) {
    gpu_check(is_compatible(desc.hw_desc, prb.hw(), exact))
            << "HW does not match";
    gpu_check(prb.prop() == desc.prop) << "Propagation kind does not match";
    gpu_check(is_compatible(tensor_kind_t::a, desc, prb, exact));
    gpu_check(is_compatible(tensor_kind_t::b, desc, prb, exact));
    gpu_check(is_compatible(tensor_kind_t::c, desc, prb, exact));
    gpu_check(prb.is_depthwise() == desc.is_dw)
            << "Mixing depthwise/non-depthwise descriptor and problem";
    if (desc.use_stream_k) {
        gpu_check(!prb.with_bias_fwd() && !prb.with_post_ops())
                << "Stream-K is incompatible with post-ops/bias";
        gpu_check(!prb.deterministic())
                << "Stream-K is not supported in deterministic mode";
    }
    if (exact) {
        gpu_check(prb.with_bias_bwd_w() == desc.with_bias_bwd_w())
                << "Problem and descriptor bias reduction mismatch";
        gpu_check(prb.with_bias_fwd() == desc.with_bias_fwd())
                << "Problem and descriptor bias mismatch";
    }
    if (prb.with_bias_bwd_w() != desc.with_bias_bwd_w()) {
        if (prb.with_bias_bwd_w()) {
            gpu_check(desc.ext.has(extension_kind_t::bias))
                    << "Bias is not supported";
        }
    }
    gpu_check(desc.reqs().fits(prb.shape()));
    return true;
}

void fit_tag_to(
        tensor_kind_t abc, kernel_desc_t &kernel_desc, const problem_t &prb) {
    auto &desc_tag = const_cast<layout_tag_t &>(kernel_desc.layout_tag(abc));
    auto &prb_tag = prb.layout_tag(abc);
    bool is_out_stream_k
            = (abc == tensor_kind_t::c) && kernel_desc.use_stream_k;
    if (desc_tag.type() != prb_tag.type() && !is_out_stream_k) {
        desc_tag = layout_tag_t(
                desc_tag.desc(), prb_tag.type(), desc_tag.raw_tag());
    }
}

void fit_to_impl(kernel_desc_t &desc, const problem_t &prb) {
    desc.hw_desc = hw_desc_t(prb.hw().to_ngen());
    fit_tag_to(tensor_kind_t::a, desc, prb);
    fit_tag_to(tensor_kind_t::b, desc, prb);
    fit_tag_to(tensor_kind_t::c, desc, prb);
    if (!prb.bias_type().is_undef()) {
        if (desc.use_stream_k) {
            auto acc_type = accumulator_type(desc.a_type(), desc.b_type());
            desc.bias_type = acc_type;
        } else {
            desc.bias_type = prb.bias_type();
        }
    }
}

bool kernel_desc_t::can_fit(const problem_t &prb) const {
    return is_compatible(*this, prb, /*exact=*/false);
}

void kernel_desc_t::fit_to(const problem_t &prb) {
    fit_to_impl(*this, prb);
    spec.specialize(prb);
}

status_t kernel_desc_t::set_post_ops(const post_ops_t &attr_post_ops,
        const memory_desc_t *out_md, const convolution_pd_t *pd) {
    for (int i = 0; i < attr_post_ops.len(); i++) {
        auto &e = attr_post_ops.entry_[i];
        if (e.is_binary()) {
            auto &md = e.binary.src1_desc;
            gpu_assert(out_md->ndims == md.ndims);
            memory_desc_t axb_md;
            CHECK(memory_desc_init_by_tag(axb_md, md.ndims, md.dims,
                    md.data_type,
                    utils::pick(md.ndims - 3, format_tag::acb, format_tag::acdb,
                            format_tag::acdeb)));
            if (memory_desc_wrapper(md) != memory_desc_wrapper(axb_md))
                return status::unimplemented;
        }
    }

    // Adjust post-ops to be expressed in terms of the full layout, including
    // all spatial dimensions.
    int old_ndims = out_md->ndims;
    int new_ndims = 5;
    post_op::ndim_normalizer_t ndim_normalizer {2, new_ndims - old_ndims};
    return gpu_post_ops_t::make(post_ops, attr_post_ops, out_md,
            post_op::specializations_t(), ndim_normalizer);
}

bool kernel_desc_t::matches(const problem_t &prb) const {
    return is_compatible(*this, prb, /*exact=*/true);
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
    oss << "_sk_" << (use_stream_k ? "1" : "0");
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
    oss << "HW:                     " << jit::to_string(hw_desc.hw)
        << std::endl;
    oss << "FMA kind:               " << to_string(fma) << std::endl;
    oss << "SIMD:                   " << simd << std::endl;
    oss << "Registers:              " << regs << std::endl;
    oss << "Iteration tile:         " << iter_tile << std::endl;
    oss << "Iteration outer tile:   " << iter_outer_tile << std::endl;
    oss << "Thread group tile:      " << thread_group_tile << std::endl;
    oss << "Loop desc:              " << loop_desc << std::endl;
    oss << "Use Stream-K:           " << ir_utils::to_string(use_stream_k)
        << std::endl;
    oss << "Use block 2D access:    " << ir_utils::to_string(use_2d_access)
        << std::endl;
    oss << "Prefetch:               " << prefetch.str() << std::endl;
    if (spec) oss << "Specialization:         " << spec.str() << std::endl;
    oss << "Extensions:             " << ext.str() << std::endl;
    oss << "Command:                " << cmd_str();
    return ir_utils::add_tag("Desc", oss.str());
}

void kernel_desc_t::init_parse_iface(parse_iface_t<kernel_desc_t> *iface) {
    iface->set_relaxed(true);
#define PACK(member) decltype(kernel_desc_t::member), &kernel_desc_t::member
    iface->add<PACK(hw_desc)>(
            "hw", "Hardware (xehpc, xe2 or xe3).", /*required=*/true);
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
            /*required=*/false);
    iface->add<PACK(iter_outer_tile)>("iter_outer",
            "Outer iteration tile (e.g. mb2).",
            /*required=*/false);
    iface->add<PACK(thread_group_tile)>(
            "tg", "Threadgroup tile (e.g. ow4oc4).", /*required=*/false);
    iface->add<PACK(loop_desc)>("loop_desc",
            "Loop description, variables ordered from innermost to outermost "
            "(e.g. kw,kh,kd,ic).",
            /*required=*/false, [](const kernel_desc_t &parent) {
                return default_loop_desc(parent.prop).str();
            });
    iface->add<PACK(use_stream_k)>("stream-k", "Whether to use Stream-K.");
    iface->add<PACK(use_2d_access)>(
            "2d", "Whether to use block 2D messages for access.");
    iface->add<PACK(prefetch)>("prefetch",
            "Prefetch description specifying distance and whether A/B are "
            "prefetched. Examples: x3 (distance is 3, both A/B are "
            "prefetched), x2.a (distance is 2, only A is prefetched), x0 (no "
            "prefetch, default).");
    iface->add<PACK(spec)>("spec",
            "Dimension specialization requirements (e.g. kd1kh1 for fixed "
            "values or oc@64 for divisibility requirements). Special "
            "values max and min_dims can be used for "
            "problem-specific specialization, e.g. mb1:min_dims.");
    iface->add<PACK(ext)>("ext",
            "Kernel extensions, comma-separated (e.g. "
            "bias,out_b1,out_b2,out_b4).");

    parse_iface_t<kernel_desc_t>::entry_t po_entry;
    po_entry.name = "post_ops";
    po_entry.help = "Kernel post-ops.";
    po_entry._default = [](const kernel_desc_t &) {
        return serialize_to_hex(gpu_post_ops_t());
    };
    po_entry.stringify = [](std::ostream &out, const kernel_desc_t &parent) {
        out << serialize_to_hex(parent.post_ops);
    };
    po_entry.parse = [](std::istream &in, kernel_desc_t &parent) {
        auto s_data = stream_parse<std::string>(in);
        deserialize_from_hex(parent.post_ops, s_data);
    };
    iface->add(po_entry);
#undef PACK

    iface->set_post_parse_func(
            [](kernel_desc_t &desc) { desc.set_defaults(); });
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
    } else if (name == "bias") {
        if (is_fwd()) return DNNL_ARG_BIAS;
        if (is_bwd_d()) return DNNL_ARG_BIAS;
        if (is_bwd_w()) return DNNL_ARG_DIFF_BIAS;
    }
    return DNNL_ARG_UNDEF;
}

bool arg_helper_t::is_input(const std::string &name) const {
    if (name == "src") return is_fwd() || is_bwd_w();
    if (name == "wei") return is_fwd() || is_bwd_d();
    if (name == "dst") return is_bwd_d() || is_bwd_w();
    if (name == "bias") return desc_.with_bias_fwd();
    gpu_error_not_expected();
    return false;
}

bool arg_helper_t::is_output(const std::string &name) const {
    if (name == "src") return is_bwd_d();
    if (name == "wei") return is_bwd_w();
    if (name == "dst") return is_fwd();
    if (name == "bias") return desc_.with_bias_bwd_w();
    gpu_error_not_expected();
    return false;
}

std::string arg_helper_t::post_op_name(size_t idx) const {
    gpu_assert(idx < desc_.post_ops.len());
    auto &po = desc_.post_ops[idx];
    if (po.is_eltwise() || po.is_sum()) return "";
    if (po.is_binary()) return "binary_" + std::to_string(idx);
    gpu_error_not_expected();
    return "";
}

int arg_helper_t::post_op_key(size_t idx) const {
    int _idx = static_cast<int>(idx);
    gpu_assert(idx < desc_.post_ops.len());
    auto &po = desc_.post_ops[idx];
    if (po.is_eltwise() || po.is_sum()) return DNNL_ARG_UNDEF;
    if (po.is_binary() && po.as_binary().alg == alg_kind::binary_prelu) {
        return DNNL_ARG_ATTR_MULTIPLE_POST_OP(_idx) | DNNL_ARG_WEIGHTS;
    }
    if (po.is_binary()) {
        return DNNL_ARG_ATTR_MULTIPLE_POST_OP(_idx) | DNNL_ARG_SRC_1;
    }
    gpu_error_not_expected();
    return -1;
}

tensor_config_t get_tensor_config(
        const kernel_desc_t &desc, const convolution_pd_t *pd = nullptr) {
    arg_helper_t h(desc);
    tensor_config_t tensor_cfg;
    for (auto *t : {"src", "wei", "dst", "bias"}) {
        bool is_input = h.is_input(t);
        bool is_output = h.is_output(t);
        if (!is_input && !is_output) continue;
        int key = h.key(t);
        tensor_cfg.add_tensor(t, key, is_input, is_output,
                pd ? jit::layout_t(pd->arg_md(key)) : jit::layout_t());
    }
    for (size_t i = 0; i < desc.post_ops.len(); i++) {
        auto name = h.post_op_name(i);
        if (name.empty()) continue;
        int key = h.post_op_key(i);
        tensor_cfg.add_tensor(name, key, /*is_input=*/true,
                /*is_output=*/false,
                pd ? jit::layout_t(pd->arg_md(key)) : jit::layout_t());
    }
    return tensor_cfg;
}

send_kind_t kernel_desc_t::access_kind(
        send_op_t op, tensor_kind_t tensor) const {
    if (use_2d_access && tensor != tensor_kind_t::undef && !is_atomic(op))
        return send_kind_t::_2d;
    return send_kind_t::undef;
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

void kernel_desc_t::init_kernel_iface(kernel_iface_t &kernel_iface) const {
    auto tensor_config = get_tensor_config(*this);
    for (auto &t : tensor_config.tensors()) {
        kernel_iface.register_arg(t.name, type_t::byte_ptr());
    }
    auto _reqs = reqs();
    auto tg_grid = create_thread_group_grid(*this);
    for (int i = 0; i < grid_t::N; i++) {
        auto &dims = tg_grid.dims(i);
        for (size_t j = 0; j < dims.size(); j++) {
            if (j == dims.size() - 1) continue;
            kernel_iface.register_arg(
                    dims[j].str() + "_grid_size", type_t::u32());
            kernel_iface.register_arg(
                    dims[j].str() + "_grid_size_magic", type_t::u64());
        }
    }
    for (auto &d : conv_dims()) {
        dim_t dummy;
        if (_reqs.get_value(d, dummy)) continue;
        auto var = var_t::make(type_t::s32(), d.str());
        kernel_iface.register_arg(var);
        if (d == pvars::sw)
            kernel_iface.register_arg("sw_magic", type_t::u64());
    }
    if (use_stream_k) {
        kernel_iface.register_arg("sk_iters_per_tile", type_t::s32());
        kernel_iface.register_arg("sk_iters_per_tile_magic", type_t::u64());
        kernel_iface.register_arg("sk_total_iters", type_t::s32());
        kernel_iface.register_arg("sk_iters_per_tg", type_t::s32());
        kernel_iface.register_arg("sk_iters_per_tg_magic", type_t::u64());
        for (auto &e : loop_desc) {
            dim_t dummy;
            if (_reqs.get_value(e.dim, dummy)) continue;
            dim_t iter_size = iter_tile.get(e.dim, 1);
            dim_t tg_size = thread_group_tile.get(e.dim, 1);
            dim_t size = iter_size * tg_size;
            std::string bound_name = e.dim.str();
            if (size != 1) bound_name += "_divup_" + std::to_string(size);
            kernel_iface.register_arg(bound_name + "_magic", type_t::u64());
        }
    }
}

static bool try_parse_internal_arg(std::string s, pvar_t &dim, dim_t &denom,
        const std::string &suffix = {}) {
    size_t pos;
    if (suffix.empty()) {
        pos = s.size();
    } else {
        pos = s.find(suffix);
        if (pos == std::string::npos) return false;
    }
    s = s.substr(0, pos);
    const char *divup_tag = "_divup_";
    size_t divup_pos = s.find(divup_tag);
    denom = 1;
    if (divup_pos != std::string::npos) {
        auto pos = divup_pos + std::strlen(divup_tag);
        denom = std::stoi(s.substr(pos));
        s = s.substr(0, divup_pos);
    }
    dim = pvar_t(s);
    return true;
}

bool try_register_internal_arg(kernel_info_t &kernel_info, const expr_t &var,
        const pvar_tile_t &pvar_map) {
    auto &type = var.type();
    auto &name = var.as<var_t>().name;
    pvar_t dim;
    dim_t denom = 1;
    if (try_parse_internal_arg(name, dim, denom, "_magic")) {
        gpu_assert(var.type().is_u64());
        uint64_t value = ir_utils::idiv_magicgu_packed(
                into<uint32_t>(utils::div_up(pvar_map.at(dim), denom)));
        kernel_info.set_internal_arg(name, value);
        return true;
    }
    if (try_parse_internal_arg(name, dim, denom)) {
        gpu_assert(!dim.is_undef());
        if (type == type_t::s32()) {
            int32_t value
                    = into<int32_t>(utils::div_up(pvar_map.at(dim), denom));
            kernel_info.set_internal_arg(name, value);
        } else if (type == type_t::u32()) {
            uint32_t value
                    = into<uint32_t>(utils::div_up(pvar_map.at(dim), denom));
            kernel_info.set_internal_arg(name, value);
        }
        return true;
    }
    return false;
}

dim_t stream_k_thread_groups(
        dim_t total_iters, dim_t max_thread_groups_per_wave) {
    const dim_t min_iters_per_tg = 2;
    dim_t ref_iters = utils::div_up(total_iters, min_iters_per_tg);
    return std::min(ref_iters, max_thread_groups_per_wave);
}

type_t accumulator_type(const type_t &a_type, const type_t &b_type) {
    gpu_assert(a_type.size() == b_type.size());
    return a_type.is_fp() ? type_t::f32() : type_t::s32();
}

kernel_desc_t to_stream_k(const kernel_desc_t &desc, bool check_ext) {
    if (desc.use_stream_k) return desc;
    if (check_ext && !desc.ext.has(extension_kind_t::stream_k))
        return kernel_desc_t();
    if (desc.with_bias_fwd()) return kernel_desc_t();

    auto sk_desc = desc;
    sk_desc.use_stream_k = true;
    auto out_kind = pick_c(sk_desc.prop, tensor_kind_t::src, tensor_kind_t::wei,
            tensor_kind_t::dst);
    auto acc_type = accumulator_type(sk_desc.a_type(), sk_desc.b_type());
    switch (out_kind) {
        case tensor_kind_t::src:
            sk_desc.src_tag = sk_desc.src_tag.with_type(acc_type);
            break;
        case tensor_kind_t::wei:
            sk_desc.wei_tag = sk_desc.wei_tag.with_type(acc_type);
            break;
        case tensor_kind_t::dst:
            sk_desc.dst_tag = sk_desc.dst_tag.with_type(acc_type);
            break;
        default: gpu_error_not_expected();
    }
    sk_desc.set_defaults();
    return sk_desc;
}

void init_kernel_info(kernel_info_t &kernel_info, const problem_t &prb,
        const kernel_desc_t &desc, const grid_t &tg_grid,
        const pvar_tile_t &grid_dims, dim_t max_tgs, dim_t &stream_k_tgs) {
    auto pvar_map = prb.shape();
    for (auto &d : grid_dims) {
        pvar_map[pvar_t(d.str() + "_grid_size")] = grid_dims.at(d);
    }
    if (desc.use_stream_k) {
        dim_t iters_per_tile = 1;
        for (auto &e : desc.loop_desc) {
            dim_t tg_size = desc.thread_group_tile.get(e.dim, 1);
            dim_t iter_size = desc.iter_tile.get(e.dim, 1);
            dim_t dim_iters_per_tile
                    = utils::div_up(prb.shape().at(e.dim), tg_size * iter_size);
            iters_per_tile *= dim_iters_per_tile;
        }
        dim_t total_iters = iters_per_tile * tg_grid.size(0, grid_dims);
        stream_k_tgs = stream_k_thread_groups(total_iters, max_tgs);
        dim_t iters_per_tg = utils::div_up(total_iters, stream_k_tgs);
        pvar_map[pvar_t("sk_iters_per_tile")] = iters_per_tile;
        pvar_map[pvar_t("sk_total_iters")] = total_iters;
        pvar_map[pvar_t("sk_iters_per_tg")] = iters_per_tg;
    }
    for (int i = 0; i < kernel_info.nargs(); i++) {
        auto &var = kernel_info.arg_var(i);
        if (var.type().is_scalar()) {
            bool ok = try_register_internal_arg(kernel_info, var, pvar_map);
            gpu_assert(ok) << "Cannot handle argument: " << var;
        }
    }
}

void kernel_desc_t::init_kernel_info(kernel_info_t &kernel_info,
        const kernel_params_base_t &params,
        const impl::engine_t *engine) const {
    auto &prb = static_cast<const kernel_params_t &>(params).prb;
    auto tg_grid = create_thread_group_grid(*this);
    auto thr_grid = create_thread_grid(*this);
    auto &shape = prb.shape();
    pvar_tile_t grid_dims;
    for (auto &d : tg_grid.all_dims()) {
        dim_t tg_size = thread_group_tile.get(d, 1);
        dim_t iter_size = iter_tile.get(d, 1);
        grid_dims[d] = utils::div_up(shape.at(d), tg_size * iter_size);
    }
    dim_t max_tgs = prim_config_t::get_max_threadgroups_per_wave(
            exec_cfg(engine), thread_group_tile.elems());
    dim_t stream_k_tgs = 0;
    conv::init_kernel_info(
            kernel_info, prb, *this, tg_grid, grid_dims, max_tgs, stream_k_tgs);
    compute::range_t gws = compute::range_t::empty();
    compute::range_t lws = compute::range_t::empty();
    for (size_t i = 0; i < compute::range_t::max_ndims; i++) {
        size_t tg_dim = thr_grid.size(i, thread_group_tile);
        lws[i] = tg_dim * (i == 0 ? into<size_t>(simd) : 1);
        gws[i] = lws[i];
    }
    if (use_stream_k) {
        gws[0] *= stream_k_tgs;
    } else {
        for (size_t i = 0; i < compute::range_t::max_ndims; i++) {
            gws[i] *= tg_grid.size(i, grid_dims);
        }
    }
    auto nd_range = compute::nd_range_t(gws, lws);
    kernel_info.set_nd_range(nd_range);
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

jit::layout_t get_kernel_layout(const std::string &name,
        const kernel_desc_t &desc, const memory_desc_t &md,
        const convolution_pd_t *pd) {
    layout_tag_t tag;
    if (name == "src") {
        tag = desc.src_tag;
    } else if (name == "wei") {
        tag = desc.wei_tag;
    } else if (name == "dst") {
        tag = desc.dst_tag;
    } else if (name == "bias") {
        tag = make_conv_layout_tag(
                tensor_kind_t::bias, "a:" + desc.bias_type.str());
    } else if (name.find("binary") == 0) {
        auto out_kind = pick_c(desc.prop, tensor_kind_t::src,
                tensor_kind_t::wei, tensor_kind_t::dst);
        tag = make_conv_layout_tag(
                out_kind, "axb:" + type_t(md.data_type).str());
    }
    gpu_assert(!tag.is_empty()) << "Unknown tensor: " << name;
    auto layout = to_conv_layout(tag, md, name == "wei" && !pd->with_groups());
    if (layout.type() != tag.type()) layout = layout.retype(tag.type());
    return layout;
}

status_t kernel_desc_t::init_primitive_plan(primitive_init_plan_t &plan,
        const problem_t &prb, convolution_pd_t *pd) const {
    auto tensor_config = get_tensor_config(*this, pd);
    int scratchpad_key = memory_tracking::names::key_none;
    for (auto &t : tensor_config.tensors()) {
        auto user_name = t.name;
        auto &md = *pd->arg_md(t.arg_key);
        auto compute_layout = get_kernel_layout(t.name, *this, md, pd);
        auto user_layout = jit::layout_t(md, /*do_normalize=*/false);
        bool is_out_stream_k = use_stream_k && t.is_output;
        bool zero_out = is_out_stream_k;
        if (compute_layout != user_layout) {
            user_name += "_user";
            scratchpad_key++;
            pd->scratchpad_registry().registrar().book(
                    into<uint32_t>(scratchpad_key), compute_layout.size(), 1,
                    ocl::OCL_BUFFER_ALIGNMENT);
            plan.add_internal_buffer(t.name, compute_layout, user_name,
                    scratchpad_key, zero_out);
            zero_out = false;
        }
        plan.add_user_buffer(user_name, user_layout, t.is_input, t.is_output,
                t.arg_key, zero_out);
        if (user_name == t.name) {
            gpu_assert(user_layout == compute_layout)
                    << "Incompatible user/kernel layouts. User: "
                    << user_layout.str()
                    << ", kernel: " << compute_layout.str();
        }
    }
    kernel_params_t _params;
    _params.prb = prb;
    auto desc = std::make_shared<kernel_desc_t>(*this);
    auto params = std::make_shared<kernel_params_t>(_params);
    plan.set_regs(regs);
    plan.set_simd(simd);
    plan.set_dpas(fma == fma_kind_t::dpas);
    plan.add_kernel(desc, params);
    return status::success;
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
    grid_t grid(jit::ir_builder_t::tg_idxs());
    auto set = [&](const pvar_t dim, int idx) {
        grid.add_mapping(dim, desc.use_stream_k ? 0 : idx);
    };
    switch (desc.prop) {
        case prop_kind::forward:
            set(pvars::oc, 0);
            set(pvars::g, 1);
            set(pvars::ow, 1);
            set(pvars::oh, 1);
            set(pvars::od, 1);
            set(pvars::mb, 2);
            break;
        case prop_kind::backward_data:
            set(pvars::ic, 0);
            set(pvars::g, 1);
            set(pvars::iw, 1);
            set(pvars::ih, 1);
            set(pvars::id, 1);
            set(pvars::mb, 2);
            break;
        case prop_kind::backward_weights:
            set(pvars::oc, 0);
            set(pvars::ic, 1);
            set(pvars::kw, 1);
            set(pvars::kh, 1);
            set(pvars::kd, 1);
            set(pvars::g, 2);
            break;
        default: gpu_error_not_expected();
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
        default: gpu_error_not_expected();
    }
    for (auto &d : grid.all_dims()) {
        if (!desc.thread_group_tile.has(d)) grid.unset(d);
    }
    return grid;
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
