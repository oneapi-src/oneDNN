/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include "common/dnnl_thread.hpp"

#include "cpu/x64/jit_uni_reduction.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

static cpu_isa_t get_supported_isa() {
    if (mayiuse(avx512_core_fp16)) return avx512_core_fp16;
    if (mayiuse(avx512_core_bf16)) return avx512_core_bf16;
    if (mayiuse(avx512_core)) return avx512_core;
    if (mayiuse(avx2_vnni_2)) return avx2_vnni_2;
    if (mayiuse(avx2)) return avx2;
    if (mayiuse(avx)) return avx;
    if (mayiuse(sse41)) return sse41;

    return isa_undef;
}

static bool impl_supports_datatype(data_type_t data_type) {
    switch (data_type) {
        case data_type::bf16:
            return mayiuse(avx512_core) || mayiuse(avx2_vnni_2);
        case data_type::f16:
            return mayiuse(avx512_core_fp16) || mayiuse(avx2_vnni_2);
        case data_type::f32:
        case data_type::s32:
        case data_type::s8:
        case data_type::u8: return true;
        default: return false;
    }
}

status_t jit_uni_reduction_t::pd_t::init(engine_t *engine) {
    using namespace alg_kind;
    using namespace data_type;
    using namespace format_tag;
    using sm = primitive_attr_t::skip_mask_t;

    conf_.isa = get_supported_isa();

    conf_.src_type = src_md()->data_type;
    conf_.dst_type = dst_md()->data_type;
    conf_.acc_type
            = types::default_accum_data_type(conf_.src_type, conf_.dst_type);
    conf_.src_dt_size = types::data_type_size(conf_.src_type);
    conf_.dst_dt_size = types::data_type_size(conf_.dst_type);
    conf_.acc_dt_size = types::data_type_size(conf_.acc_type);

    const bool ok = impl_supports_datatype(conf_.src_type)
            && impl_supports_datatype(conf_.dst_type)
            && set_default_params() == status::success
            && attr()->has_default_values(sm::post_ops)
            && attr_.set_default_formats(dst_md(0)) == status::success;
    if (!ok) return status::unimplemented;

    const auto src_mdw = memory_desc_wrapper(src_md());
    const auto dst_mdw = memory_desc_wrapper(dst_md());

    const std::vector<injector::post_op_type> accepted_post_ops
            = {injector::sum, injector::eltwise, injector::binary};
    static constexpr bool sum_at_0_pos_only = false;
    static constexpr bool sum_requires_scale_one = false;
    static constexpr bool sum_requires_zp_zero = true;
    static constexpr bool sum_requires_same_params = false;
    const bcast_set_t accepted_broadcasts
            = {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
                    broadcasting_strategy_t::per_oc_spatial,
                    broadcasting_strategy_t::no_broadcast};
    injector::post_ops_ok_args_t post_ops_args(conf_.isa, accepted_post_ops,
            attr()->post_ops_, &dst_mdw, sum_at_0_pos_only,
            sum_requires_scale_one, sum_requires_zp_zero,
            sum_requires_same_params, accepted_broadcasts);
    if (!post_ops_ok(post_ops_args)) return status::unimplemented;

    conf_.post_ops = attr()->post_ops_;

    static constexpr bool require_scale_one = false;
    conf_.with_eltwise = conf_.with_binary = conf_.with_sum = false;
    for (const auto &entry : conf_.post_ops.entry_) {
        if (entry.is_eltwise()) {
            conf_.with_eltwise = true;
        } else if (entry.is_binary()) {
            conf_.with_binary = true;
        } else if (entry.is_sum(require_scale_one) && entry.sum.scale != 0.f) {
            conf_.with_sum = true;
            conf_.sum_scales.push(entry.sum.scale);
        }
    }
    conf_.with_postops
            = conf_.with_eltwise || conf_.with_binary || conf_.with_sum;

    const format_tag_t src_md_desired_format = memory_desc_matches_one_of_tag(
            *src_md(), x, nc, ncw, nchw, ncdhw);
    const format_tag_t dst_md_desired_format = memory_desc_matches_one_of_tag(
            *dst_md(), x, nc, ncw, nchw, ncdhw);
    if (src_md_desired_format != dst_md_desired_format
            || src_md_desired_format == format_tag::undef)
        return status::unimplemented;

    const int ndims = src_mdw.ndims();
    const auto &src_dims = src_mdw.dims();
    const auto &dst_dims = dst_mdw.dims();

    conf_.is_saturation_needed = utils::one_of(conf_.dst_type, s32, s8, u8);

    int num_of_reduced_dims = 0;
    conf_.idle_size = dst_mdw.nelems();
    conf_.reduce_size = 1;
    for (int d = ndims - 1; d >= 0; --d) {
        if (src_dims[d] != dst_dims[d]) {
            num_of_reduced_dims++;
            conf_.reduce_size *= src_dims[d];
        } else
            break;
    }

    if (num_of_reduced_dims == 0) return status::unimplemented;

    for (int d = 0; d < ndims - num_of_reduced_dims; ++d)
        if (src_dims[d] != dst_dims[d]) return status::unimplemented;

    conf_.alg = desc()->alg_kind;
    if (utils::one_of(conf_.alg, reduction_norm_lp_max, reduction_norm_lp_sum,
                reduction_norm_lp_power_p_max, reduction_norm_lp_power_p_sum))
        return status::unimplemented;

    return status::success;
}

status_t jit_uni_reduction_t::init(engine_t *engine) {
    using namespace format_tag;

    const memory_desc_t *dst_md = pd()->dst_md();
    const jit_reduction_conf_t &conf = pd()->get_conf();

    CHECK(get_proper_kernel(dst_md, conf));
    CHECK(kernel_->create_kernel());

    return status::success;
}

status_t jit_uni_reduction_t::execute(const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const uint8_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(uint8_t *, DNNL_ARG_DST);

    const dim_t idle_size = pd()->get_conf().idle_size;
    const dim_t reduce_size = pd()->get_conf().reduce_size;
    const std::size_t src_dt_size = pd()->get_conf().src_dt_size;
    const std::size_t dst_dt_size = pd()->get_conf().dst_dt_size;
    const auto &post_ops = pd()->attr()->post_ops_;
    const auto &post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(post_ops, ctx);

    parallel_nd(idle_size, [&](dim_t i) {
        const dim_t src_off = i * reduce_size * src_dt_size;
        const dim_t dst_off = i * dst_dt_size;

        jit_reduction_call_s args = jit_reduction_call_s();
        args.src = src + src_off;
        args.dst = dst + dst_off;
        args.dst_orig = dst;
        args.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();

        (*kernel_)(&args);
    });

    return status::success;
}

status_t jit_uni_reduction_t::get_proper_kernel(
        const memory_desc_t *dst_md, const jit_reduction_conf_t &conf) {
    using namespace data_type;

    if (conf.isa == avx512_core_fp16)
        return safe_ptr_assign(kernel_,
                new jit_uni_reduction_kernel_t<avx512_core_fp16>(conf, dst_md));
    if (conf.isa == avx512_core_bf16)
        return safe_ptr_assign(kernel_,
                new jit_uni_reduction_kernel_t<avx512_core_bf16>(conf, dst_md));
    else if (conf.isa == avx512_core)
        return safe_ptr_assign(kernel_,
                new jit_uni_reduction_kernel_t<avx512_core>(conf, dst_md));
    else if (is_superset(conf.isa, avx)) {
        const bool is_src_i8 = utils::one_of(conf.src_type, s8, u8);
        const bool is_dst_i8 = utils::one_of(conf.dst_type, s8, u8);
        if (conf.isa == avx2_vnni_2) {
            if (is_src_i8 || is_dst_i8)
                return safe_ptr_assign(kernel_,
                        new jit_uni_reduction_kernel_t<avx2_vnni_2, Xbyak::Xmm>(
                                conf, dst_md));
            else
                return safe_ptr_assign(kernel_,
                        new jit_uni_reduction_kernel_t<avx2_vnni_2>(
                                conf, dst_md));
        } else if (conf.isa == avx2) {
            if (is_src_i8 || is_dst_i8)
                return safe_ptr_assign(kernel_,
                        new jit_uni_reduction_kernel_t<avx2, Xbyak::Xmm>(
                                conf, dst_md));
            else
                return safe_ptr_assign(kernel_,
                        new jit_uni_reduction_kernel_t<avx2>(conf, dst_md));
        } else {
            if (is_src_i8 || is_dst_i8)
                return safe_ptr_assign(kernel_,
                        new jit_uni_reduction_kernel_t<avx, Xbyak::Xmm>(
                                conf, dst_md));
            else
                return safe_ptr_assign(kernel_,
                        new jit_uni_reduction_kernel_t<avx>(conf, dst_md));
        }
    } else if (conf.isa == sse41)
        return safe_ptr_assign(
                kernel_, new jit_uni_reduction_kernel_t<sse41>(conf, dst_md));
    else
        return status::runtime_error;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
