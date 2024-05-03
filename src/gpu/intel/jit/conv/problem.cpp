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

#include "gpu/intel/jit/conv/problem.hpp"
#include "common/convolution_pd.hpp"
#include "gpu/intel/jit/ir/fma.hpp"
#include "gpu/intel/jit/ir/hw.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

const std::vector<prb_dim_t> &conv_dims() {
    static std::vector<prb_dim_t> _conv_dims = []() {
        std::vector<prb_dim_t> ret;
        for (auto &d : conv_index_dims(prop_kind::forward)) {
            ret.push_back(d);
        }
        ret.push_back(prb_dims::id);
        ret.push_back(prb_dims::ih);
        ret.push_back(prb_dims::iw);
        for (auto &d : conv_stride_dims())
            ret.push_back(d);
        for (auto &d : conv_dilation_dims())
            ret.push_back(d);
        for (auto &d : conv_padding_dims())
            ret.push_back(d);
        return ret;
    }();
    return _conv_dims;
}

const std::vector<prb_dim_t> &conv_index_dims(prop_kind_t prop) {
    auto get_dims = [&](prop_kind_t prop) {
        std::vector<prb_dim_t> ret;
        ret.push_back(prb_dims::mb);
        ret.push_back(prb_dims::g);
        ret.push_back(prb_dims::oc);
        ret.push_back(prb_dims::ic);
        ret.push_back(prb_dims::kd);
        ret.push_back(prb_dims::kh);
        ret.push_back(prb_dims::kw);
        if (prop != prop_kind::backward_data) {
            ret.push_back(prb_dims::od);
            ret.push_back(prb_dims::oh);
            ret.push_back(prb_dims::ow);
        } else {
            ret.push_back(prb_dims::id);
            ret.push_back(prb_dims::ih);
            ret.push_back(prb_dims::iw);
        }
        return ret;
    };
    static std::vector<prb_dim_t> fwd_dims = get_dims(prop_kind::forward);
    static std::vector<prb_dim_t> bwd_d_dims
            = get_dims(prop_kind::backward_data);
    static std::vector<prb_dim_t> bwd_w_dims
            = get_dims(prop_kind::backward_weights);
    switch (prop) {
        case prop_kind::forward: return fwd_dims;
        case prop_kind::backward_data: return bwd_d_dims;
        case prop_kind::backward_weights: return bwd_w_dims;
        default: ir_error_not_expected(); return fwd_dims;
    }
}

bool is_conv_index(const prb_dim_t &dim) {
    for (auto prop : {prop_kind::forward, prop_kind::backward_data,
                 prop_kind::backward_weights})
        if (is_conv_index(dim, prop)) return true;
    return false;
}

bool is_conv_index(const prb_dim_t &dim, prop_kind_t prop) {
    for (auto &d : conv_index_dims(prop))
        if (d == dim) return true;
    return false;
}

const std::vector<prb_dim_t> &conv_layout_dims(
        tensor_kind_t tensor_kind, bool src_dst_with_group) {
    static const std::vector<prb_dim_t> src_dims({prb_dims::mb, prb_dims::ic,
            prb_dims::id, prb_dims::ih, prb_dims::iw});
    static const std::vector<prb_dim_t> src_g_dims({prb_dims::mb, prb_dims::g,
            prb_dims::ic, prb_dims::id, prb_dims::ih, prb_dims::iw});
    static const std::vector<prb_dim_t> wei_dims({prb_dims::g, prb_dims::oc,
            prb_dims::ic, prb_dims::kd, prb_dims::kh, prb_dims::kw});
    static const std::vector<prb_dim_t> dst_dims({prb_dims::mb, prb_dims::oc,
            prb_dims::od, prb_dims::oh, prb_dims::ow});
    static const std::vector<prb_dim_t> dst_g_dims({prb_dims::mb, prb_dims::g,
            prb_dims::oc, prb_dims::od, prb_dims::oh, prb_dims::ow});
    switch (tensor_kind) {
        case tensor_kind_t::src:
            return src_dst_with_group ? src_g_dims : src_dims;
        case tensor_kind_t::wei: return wei_dims;
        case tensor_kind_t::dst:
            return src_dst_with_group ? dst_g_dims : dst_dims;
        default: ir_error_not_expected();
    }
    return src_dims;
}

tensor_kind_t to_abc(prop_kind_t prop, tensor_kind_t tensor) {
    bool is_bwd_d = (prop == prop_kind::backward_data);
    bool is_bwd_w = (prop == prop_kind::backward_weights);
    tensor_kind_t kinds[3]
            = {tensor_kind_t::a, tensor_kind_t::b, tensor_kind_t::c};
    if (is_bwd_d) std::swap(kinds[0], kinds[2]);
    if (is_bwd_w) std::swap(kinds[1], kinds[2]);
    switch (tensor) {
        case tensor_kind_t::src: return kinds[0];
        case tensor_kind_t::wei: return kinds[1];
        case tensor_kind_t::dst: return kinds[2];
        default: ir_error_not_expected();
    }
    return kinds[0];
}

const std::vector<prb_dim_t> &conv_stride_dims() {
    static std::vector<prb_dim_t> _stride_dims = [&]() {
        std::vector<prb_dim_t> ret;
        ret.push_back(prb_dims::sd);
        ret.push_back(prb_dims::sh);
        ret.push_back(prb_dims::sw);
        return ret;
    }();
    return _stride_dims;
}

const std::vector<prb_dim_t> &conv_dilation_dims() {
    static std::vector<prb_dim_t> _dilation_dims = [&]() {
        std::vector<prb_dim_t> ret;
        ret.push_back(prb_dims::dd);
        ret.push_back(prb_dims::dh);
        ret.push_back(prb_dims::dw);
        return ret;
    }();
    return _dilation_dims;
}

const std::vector<prb_dim_t> &conv_padding_dims() {
    static std::vector<prb_dim_t> _padding_dims = [&]() {
        std::vector<prb_dim_t> ret;
        ret.push_back(prb_dims::pd);
        ret.push_back(prb_dims::ph);
        ret.push_back(prb_dims::pw);
        return ret;
    }();
    return _padding_dims;
}

const memory_desc_t &conv_problem_t::a_md() const {
    return *pick_a(conv_pd->invariant_src_md(), conv_pd->invariant_wei_md(),
            conv_pd->invariant_dst_md());
}

const memory_desc_t &conv_problem_t::b_md() const {
    return *pick_b(conv_pd->invariant_src_md(), conv_pd->invariant_wei_md(),
            conv_pd->invariant_dst_md());
}

const memory_desc_t &conv_problem_t::c_md() const {
    return *pick_c(conv_pd->invariant_src_md(), conv_pd->invariant_wei_md(),
            conv_pd->invariant_dst_md());
}

status_t conv_problem_t::init_abc_data_types(const hw_t &hw) {
    a_data_type = pick_a(src_data_type, wei_data_type, dst_data_type);
    b_data_type = pick_b(src_data_type, wei_data_type, dst_data_type);
    // Always use f32 for accumulation/storing in the main kernel.
    c_data_type = is_bwd_w
            ? data_type::f32
            : pick_c(src_data_type, wei_data_type, dst_data_type);

    if (utils::everyone_is(
                data_type::f32, a_data_type, b_data_type, c_data_type)) {

        // TODO: bf16 and f16 currently perform worse than tf32, this is
        // likely due to an extra reorder required on the b buffer.
        bool use_matching_fpmath
                = gpu_utils::dev_getenv("use_matching_fpmath", false);
        if (use_matching_fpmath
                && attr->mayiconvert(data_type::f32, data_type::bf16)
                && get_supported_fma_kind(
                           hw, data_type::bf16, data_type::bf16, data_type::f32)
                        != fma_kind_t::undef) {
            a_data_type = data_type::bf16;
            b_data_type = data_type::bf16;
        } else if (use_matching_fpmath
                && attr->mayiconvert(data_type::f32, data_type::f16)
                && get_supported_fma_kind(
                           hw, data_type::f16, data_type::f16, data_type::f32)
                        != fma_kind_t::undef) {
            a_data_type = data_type::f16;
            b_data_type = data_type::f16;
        } else if (attr->mayiconvert(data_type::f32, data_type::tf32)
                && get_supported_fma_kind(
                           hw, data_type::tf32, data_type::tf32, data_type::f32)
                        != fma_kind_t::undef) {
            a_data_type = data_type::tf32;
            b_data_type = data_type::tf32;
        }
    }

    a_data_type_size = (int)types::data_type_size(a_data_type);
    b_data_type_size = (int)types::data_type_size(b_data_type);
    c_data_type_size = (int)types::data_type_size(c_data_type);
    return status::success;
}

status_t conv_problem_t::init_acc_data_type() {
    auto a = a_data_type;
    auto b = b_data_type;
    acc_data_type = data_type::undef;
    if (utils::one_of(a, data_type::s8, data_type::u8)
            && utils::one_of(b, data_type::s8, data_type::u8)) {
        acc_data_type = data_type::s32;
    } else if (utils::everyone_is(data_type::f16, a, b)
            || utils::everyone_is(data_type::bf16, a, b)
            || utils::everyone_is(data_type::f8_e5m2, a, b)) {
        acc_data_type = data_type::f32;
    } else if (utils::everyone_is(data_type::tf32, a, b)) {
        acc_data_type = data_type::f32;
    } else if (utils::everyone_is(data_type::f32, a, b)) {
        acc_data_type = data_type::f32;
    } else if (utils::everyone_is(data_type::f64, a, b)) {
        acc_data_type = data_type::f64;
    }
    if (acc_data_type == data_type::undef) return status::unimplemented;
    acc_data_type_size = (int)types::data_type_size(acc_data_type);
    return status::success;
}

bool conv_problem_t::with_sum_post_op() const {
    auto &post_ops = attr->post_ops_;
    return post_ops.find(primitive_kind::sum) != -1;
}

void conv_problem_t::init_transpose(const hw_t &hw) {
    using sm = primitive_attr_t::skip_mask_t;
    auto attr_skip_mask = sm::post_ops | sm::sum_dt | sm::scales_runtime;
    bool allow_ab_transpose = gpu_utils::dev_getenv("allow_ab_transpose", true);
    bool any_zp = !attr->has_default_values(attr_skip_mask);
    bool any_f64 = utils::one_of(data_type::f64, src_data_type, dst_data_type);
    if (!allow_ab_transpose || any_zp || any_f64 || with_groups
            || hw <= ngen::HW::Gen9) {
        ab_swap_transpose = gpu_utils::dev_getenv("ab_swap_transpose", false);
        return;
    }
    int max_sp = (hw >= ngen::HW::XeHPC) ? 1240 : 512;
    bool do_ic_swap = ((is_fwd || is_bwd_w) && oc < 6);
    bool do_oc_swap = ((is_bwd_d) && ic < 6);
    bool allow_bwd_w = !is_bwd_w
            || ((src_data_type != data_type::f32
                        || fpmath_mode == dnnl_fpmath_mode_tf32)
                    && osp % 8 == 0);
    bool allow_bwd_d
            = !is_bwd_d || (wei_data_type == data_type::f32 && osp == isp);
    bool allow_fwd = !is_fwd
            || (dst_data_type != data_type::f32
                    && dst_data_type != data_type::f64 && mb <= 8 && ih != iw
                    && iw <= max_sp);
    ab_swap_transpose = allow_fwd && allow_bwd_d && allow_bwd_w
            && (do_oc_swap || do_ic_swap);
    ab_swap_transpose
            = gpu_utils::dev_getenv("ab_swap_transpose", ab_swap_transpose);
}

prb_dim_t to_gemm(const prb_dim_t &d, prop_kind_t prop, bool is_transpose) {
    const bool is_fwd = (prop == prop_kind::forward);
    const bool is_bwd_d = (prop == prop_kind::backward_data);
    const bool is_bwd_w = (prop == prop_kind::backward_weights);
    auto transpose_gemm = [](const prb_dim_t &d) {
        if (d == prb_dims::m) return prb_dims::n;
        if (d == prb_dims::n) return prb_dims::m;
        if (d == prb_dims::k) return prb_dims::k;
        ir_error_not_expected();
        return prb_dim_t();
    };
    auto pick = [&](const prb_dim_t &fwd, const prb_dim_t &bwd_d,
                        const prb_dim_t &bwd_w) {
        if (is_transpose) {
            if (is_fwd) return transpose_gemm(fwd);
            if (is_bwd_d) return transpose_gemm(bwd_d);
            if (is_bwd_w) return transpose_gemm(bwd_w);
        }
        if (is_fwd) return fwd;
        if (is_bwd_d) return bwd_d;
        if (is_bwd_w) return bwd_w;
        ir_error_not_expected();
        return prb_dim_t();
    };
    switch (d.kind()) {
        case prb_dim_kind_t::g: return prb_dims::b;
        case prb_dim_kind_t::mb:
            return pick(prb_dims::m, prb_dims::m, prb_dims::k);
        case prb_dim_kind_t::oc:
            return pick(prb_dims::n, prb_dims::k, prb_dims::n);
        case prb_dim_kind_t::ic:
            return pick(prb_dims::k, prb_dims::n, prb_dims::m);
        case prb_dim_kind_t::kd:
        case prb_dim_kind_t::kh:
        case prb_dim_kind_t::kw:
            return pick(prb_dims::k, prb_dims::k, prb_dims::m);
        case prb_dim_kind_t::od:
        case prb_dim_kind_t::oh:
        case prb_dim_kind_t::ow:
            return pick(prb_dims::m, prb_dim_t(), prb_dims::k);
        case prb_dim_kind_t::id:
        case prb_dim_kind_t::ih:
        case prb_dim_kind_t::iw:
            return pick(prb_dim_t(), prb_dims::m, prb_dim_t());
        default: return prb_dim_t();
    }
}

prb_tile_t to_gemm(const prb_tile_t &t, prop_kind_t prop, bool is_transpose) {
    prb_tile_t ret;
    ret[prb_dims::b] = 1;
    ret[prb_dims::m] = 1;
    ret[prb_dims::n] = 1;
    ret[prb_dims::k] = 1;
    for (auto &d : t) {
        auto gemm_d = to_gemm(d, prop, is_transpose);
        if (gemm_d.is_undef()) continue;
        ret[gemm_d] *= t[d];
    }
    return ret;
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
