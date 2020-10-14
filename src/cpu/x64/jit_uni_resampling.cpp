/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <bitset>
#include <cassert>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/resampling_utils.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_resampling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace resampling_utils;

template <cpu_isa_t isa>
status_t jit_uni_resampling_fwd_t<isa>::pd_t::init(engine_t *engine) {
    using namespace format_tag;
    using namespace data_type;

    conf_.data_type = src_md()->data_type;

    const bool ok = mayiuse(isa) && is_fwd() && !has_zero_dim_memory()
            && utils::one_of(conf_.data_type, f32, bf16)
            && IMPLICATION(conf_.data_type == bf16,
                    // extra check for isa is required because
                    // the avx512_common version may reject a
                    // problem because it is blocked by 8
                    // instead of 16.
                    isa >= avx512_common && mayiuse(avx512_core))
            && utils::everyone_is(
                    conf_.data_type, src_md()->data_type, dst_md()->data_type)
            && platform::has_data_type_support(conf_.data_type)
            && set_default_params() == status::success
            && attr()->has_default_values();
    if (!ok) return status::unimplemented;

    if (conf_.data_type == bf16)
        conf_.isa = mayiuse(avx512_core_bf16) ? avx512_core_bf16 : avx512_core;
    else if (isa != avx512_common)
        conf_.isa = mayiuse(avx2) ? avx2 : isa;
    else
        conf_.isa = isa;

    conf_.alg = desc()->alg_kind;
    conf_.od = OD();
    conf_.oh = OH();
    conf_.ow = OW();
    conf_.id = ID();
    conf_.ih = IH();
    conf_.iw = IW();
    conf_.ndims = ndims();

    if (conf_.alg == alg_kind::resampling_linear)
        conf_.number_of_corners = pow(2, conf_.ndims - 2);

    conf_.dt_size = types::data_type_size(conf_.data_type);

    const size_t L3_size = static_cast<size_t>(dnnl_get_max_threads())
            * platform::get_per_core_cache_size(3);
    size_t input_data_size = conf_.dt_size;
    size_t output_data_size = conf_.dt_size;
    for (unsigned i = 0; i < conf_.ndims; ++i) {
        output_data_size *= dst_md()->dims[i];
        input_data_size *= src_md()->dims[i];
    }
    conf_.is_data_size_bigger_than_L3
            = input_data_size + output_data_size > L3_size;

    const memory_desc_wrapper src_d(src_md());
    conf_.inner_stride = src_d.blocking_desc().strides[ndims() - 1];
    conf_.stride_d = IH() * IW() * conf_.inner_stride * conf_.dt_size;
    conf_.stride_h = IW() * conf_.inner_stride * conf_.dt_size;
    conf_.stride_w = conf_.inner_stride * conf_.dt_size;

    conf_.simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);

    const format_tag_t blocked_tag = isa >= avx512_common
            ? utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c)
            : utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);

    const format_tag_t blocked_format
            = memory_desc_matches_tag(*src_md(), blocked_tag)
            ? blocked_tag
            : format_tag::undef;
    const format_tag_t nspc_format
            = memory_desc_matches_one_of_tag(*src_md(), nwc, nhwc, ndhwc);
    const format_tag_t ncsp_format
            = memory_desc_matches_one_of_tag(*src_md(), ncw, nchw, ncdhw);

    if (memory_desc_matches_tag(*dst_md(), blocked_format)) {
        conf_.tag_kind = jit_memory_tag_kind_t::blocked;
        conf_.tail = 0;
    } else if (memory_desc_matches_tag(*dst_md(), nspc_format)) {
        conf_.tag_kind = jit_memory_tag_kind_t::nspc;
        conf_.tail = conf_.inner_stride % conf_.simd_w;
    } else if (memory_desc_matches_tag(*dst_md(), ncsp_format)) {
        conf_.tag_kind = jit_memory_tag_kind_t::ncsp;
        if (conf_.alg == alg_kind::resampling_nearest)
            conf_.tail = conf_.ow % conf_.simd_w;
        else
            conf_.tail = (conf_.od * conf_.oh * conf_.ow) % conf_.simd_w;
    } else
        return status::unimplemented;

    conf_.el_size_of_indices = sizeof(unsigned);

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_resampling_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(
            kernel_, new jit_uni_resampling_kernel<isa>(pd()->get_conf())));

    CHECK(kernel_->create_kernel());

    return fill_data_for_interpolation();
}

template <cpu_isa_t isa>
status_t jit_uni_resampling_fwd_t<isa>::fill_data_for_interpolation() {
    switch (pd()->desc()->alg_kind) {
        case alg_kind::resampling_nearest: return fill_data_for_nearest();
        case alg_kind::resampling_linear: return fill_data_for_linear();
        default:
            assert(!"Invalid resampling algorithm.");
            return status::invalid_arguments;
    }
}

template <cpu_isa_t isa>
status_t jit_uni_resampling_fwd_t<isa>::fill_data_for_nearest() {
    // In kernel is used vmovdqu to get indices. This instruction don't have
    // tail processing possibilities on sse41 and avx. To avoid problems
    // with that, OW is aligned to simd width, because indices for ow
    // are read in the kernel.
    indices_.reserve(pd()->OD() + pd()->OH()
            + utils::rnd_up(pd()->OW(), pd()->get_conf().simd_w));

    for (dim_t od = 0; od < pd()->OD(); od++) {
        const int offset_id = nearest_idx(od, pd()->OD(), pd()->ID())
                * pd()->get_conf().stride_d;
        indices_.push_back(offset_id);
    }
    for (dim_t oh = 0; oh < pd()->OH(); oh++) {
        const int offset_ih = nearest_idx(oh, pd()->OH(), pd()->IH())
                * pd()->get_conf().stride_h;
        indices_.push_back(offset_ih);
    }
    for (dim_t ow = 0; ow < pd()->OW(); ow++) {
        const int offset_iw = nearest_idx(ow, pd()->OW(), pd()->IW())
                * pd()->get_conf().stride_w;
        indices_.push_back(offset_iw);
    }

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_resampling_fwd_t<isa>::fill_data_for_linear() {
    using namespace resampling_utils;

    const unsigned number_of_corners = pd()->get_conf().number_of_corners;
    const unsigned stride_w = pd()->get_conf().stride_w;
    const unsigned stride_h = pd()->get_conf().stride_h;
    const unsigned stride_d = pd()->get_conf().stride_d;

    unsigned num_of_elements = 0;
    if (pd()->get_conf().tag_kind == jit_memory_tag_kind_t::ncsp) {
        // In kernel is used vmovdqu to get indices. This instruction don't have
        // tail processing possibilities on sse41 and avx. To avoid problems
        // with that, number of spatial points is aligned to simd width, because
        // all of them are read in the kernel.
        num_of_elements = number_of_corners
                * utils::rnd_up(pd()->OD() * pd()->OH() * pd()->OW(),
                        pd()->get_conf().simd_w);

        indices_.resize(num_of_elements);
        weights_.resize(num_of_elements);

        const size_t indices_stride = pd()->OW() * pd()->OH() * pd()->OD();
        const size_t weights_stride = pd()->OW() * pd()->OH() * pd()->OD();

        parallel_nd(pd()->OD(), pd()->OH(), [&](dim_t od, dim_t oh) {
            const linear_coeffs_t coeffs_id(od, pd()->OD(), pd()->ID());
            const linear_coeffs_t coeffs_ih(oh, pd()->OH(), pd()->IH());

            for (dim_t ow = 0; ow < pd()->OW(); ow++) {
                const size_t offset
                        = od * pd()->OH() * pd()->OW() + oh * pd()->OW() + ow;

                const linear_coeffs_t coeffs_iw(ow, pd()->OW(), pd()->IW());

                for (unsigned i = 0; i < number_of_corners; i++) {
                    std::bitset<3> corners(i);
                    indices_[i * indices_stride + offset]
                            = coeffs_id.idx[corners.test(2)] * stride_d
                            + coeffs_ih.idx[corners.test(1)] * stride_h
                            + coeffs_iw.idx[corners.test(0)] * stride_w;
                    weights_[i * weights_stride + offset]
                            = coeffs_id.wei[corners.test(2)]
                            * coeffs_ih.wei[corners.test(1)]
                            * coeffs_iw.wei[corners.test(0)];
                }
            }
        });
    } else if (pd()->get_conf().tag_kind == jit_memory_tag_kind_t::nspc
            || pd()->get_conf().tag_kind == jit_memory_tag_kind_t::blocked) {
        num_of_elements = 2 * (pd()->OD() + pd()->OH() + pd()->OW());

        indices_.resize(num_of_elements);
        weights_.resize(num_of_elements);

        unsigned *indices_w = &indices_[0];
        unsigned *indices_h = &indices_[2 * pd()->OW()];
        unsigned *indices_d = &indices_[2 * (pd()->OW() + pd()->OH())];
        float *weights_w = &weights_[0];
        float *weights_h = &weights_[2 * pd()->OW()];
        float *weights_d = &weights_[2 * (pd()->OW() + pd()->OH())];

        for (dim_t ow = 0; ow < pd()->OW(); ow++) {
            const linear_coeffs_t coeffs_iw(ow, pd()->OW(), pd()->IW());

            // The right and left corners are set one after
            // the other because in the kernel these values
            // are read one by one, which makes it easier
            // to read and makes the operation faster.
            weights_w[2 * ow] = coeffs_iw.wei[0];
            weights_w[2 * ow + 1] = coeffs_iw.wei[1];
            indices_w[2 * ow] = coeffs_iw.idx[0] * stride_w;
            indices_w[2 * ow + 1] = coeffs_iw.idx[1] * stride_w;
        }

        for (dim_t oh = 0; oh < pd()->OH(); oh++) {
            const linear_coeffs_t coeffs_ih(oh, pd()->OH(), pd()->IH());

            weights_h[oh] = coeffs_ih.wei[0];
            weights_h[pd()->OH() + oh] = coeffs_ih.wei[1];
            indices_h[oh] = coeffs_ih.idx[0] * stride_h;
            indices_h[pd()->OH() + oh] = coeffs_ih.idx[1] * stride_h;
        }

        for (dim_t od = 0; od < pd()->OD(); od++) {
            const linear_coeffs_t coeffs_id(od, pd()->OD(), pd()->ID());

            weights_d[od] = coeffs_id.wei[0];
            weights_d[pd()->OD() + od] = coeffs_id.wei[1];
            indices_d[od] = coeffs_id.idx[0] * stride_d;
            indices_d[pd()->OD() + od] = coeffs_id.idx[1] * stride_d;
        }
    } else {
        assert(!"Invalid memory format kind.");
        return status::invalid_arguments;
    }

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_resampling_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const uint8_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(uint8_t *, DNNL_ARG_DST);

    switch (pd()->desc()->alg_kind) {
        case alg_kind::resampling_nearest: return interpolate_nearest(src, dst);
        case alg_kind::resampling_linear: return interpolate_linear(src, dst);
        default:
            assert(!"Invalid resampling algorithm.");
            return status::invalid_arguments;
    }
}

template <cpu_isa_t isa>
status_t jit_uni_resampling_fwd_t<isa>::interpolate_nearest(
        const uint8_t *src, uint8_t *dst) const {
    const size_t dt_size = pd()->get_conf().dt_size;
    const size_t inner_stride = pd()->get_conf().inner_stride;

    const dim_t MB = pd()->MB();
    const dim_t C = pd()->C();
    const dim_t CB = utils::div_up(C, inner_stride);
    const dim_t nsp_outer = MB * CB;
    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();

    const unsigned *indices_d = &indices_[0];
    const unsigned *indices_h = &indices_[OD];
    const unsigned *indices_w = &indices_[OD + OH];

    if (pd()->get_conf().tag_kind == jit_memory_tag_kind_t::ncsp) {
        parallel_nd(MB, C, OD, [&](dim_t mb, dim_t c, dim_t od) {
            const dim_t src_off
                    = (mb * C + c) * ID * IH * IW * dt_size + indices_d[od];
            const dim_t dst_off
                    = ((mb * C + c) * OD * OH * OW + od * OH * OW) * dt_size;

            jit_resampling_call_s args = jit_resampling_call_s();
            args.src = src + src_off;
            args.dst = dst + dst_off;
            args.indices = &indices_h[0];

            (*kernel_)(&args);
        });
    } else if (pd()->get_conf().tag_kind == jit_memory_tag_kind_t::nspc
            || pd()->get_conf().tag_kind == jit_memory_tag_kind_t::blocked) {
        parallel_nd(nsp_outer, OD, OH, [&](dim_t nsp, dim_t od, dim_t oh) {
            const dim_t src_off = nsp * ID * IH * IW * inner_stride * dt_size
                    + indices_d[od] + indices_h[oh];
            const dim_t dst_off
                    = ((nsp * OD + od) * OH + oh) * OW * inner_stride * dt_size;

            jit_resampling_call_s args = jit_resampling_call_s();
            args.batch_of_sp_points_to_process = OW;
            args.src = src + src_off;
            args.dst = dst + dst_off;
            args.indices = &indices_w[0];

            (*kernel_)(&args);
        });
    } else {
        assert(!"Invalid memory format kind.");
        return status::invalid_arguments;
    }

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_resampling_fwd_t<isa>::interpolate_linear(
        const uint8_t *src, uint8_t *dst) const {
    const size_t dt_size = pd()->get_conf().dt_size;
    const size_t inner_stride = pd()->get_conf().inner_stride;

    const dim_t MB = pd()->MB();
    const dim_t C = pd()->C();
    const dim_t CB = utils::div_up(C, inner_stride);
    const dim_t nsp_outer = MB * CB;
    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();

    if (pd()->get_conf().tag_kind == jit_memory_tag_kind_t::ncsp) {
        parallel_nd(MB, C, [&](dim_t mb, dim_t c) {
            const dim_t src_off = (mb * C + c) * ID * IH * IW * dt_size;
            const dim_t dst_off = (mb * C + c) * OD * OH * OW * dt_size;

            jit_resampling_call_s args = jit_resampling_call_s();
            args.batch_of_sp_points_to_process = OW * OH * OD;
            args.src = src + src_off;
            args.dst = dst + dst_off;
            args.indices = &indices_[0];
            args.weights = &weights_[0];

            (*kernel_)(&args);
        });
    } else if (pd()->get_conf().tag_kind == jit_memory_tag_kind_t::nspc
            || pd()->get_conf().tag_kind == jit_memory_tag_kind_t::blocked) {
        const unsigned *indices_top = &indices_[2 * OW];
        const unsigned *indices_bottom = &indices_[2 * OW + OH];
        const unsigned *indices_front = &indices_[2 * (OW + OH)];
        const unsigned *indices_back = &indices_[2 * (OW + OH) + OD];
        const float *weights_top = &weights_[2 * OW];
        const float *weights_bottom = &weights_[2 * OW + OH];
        const float *weights_front = &weights_[2 * (OW + OH)];
        const float *weights_back = &weights_[2 * (OW + OH) + OD];

        parallel_nd(nsp_outer, OD, OH, [&](dim_t nsp, dim_t od, dim_t oh) {
            const dim_t src_off = nsp * ID * IH * IW * inner_stride * dt_size;
            const dim_t dst_off = (((nsp * OD + od) * OH + oh) * OW)
                    * inner_stride * dt_size;

            jit_resampling_call_s args = jit_resampling_call_s();
            args.batch_of_sp_points_to_process = OW;
            args.src = src + src_off;
            args.dst = dst + dst_off;
            args.indices = &indices_[0];
            args.weights = &weights_[0];
            args.src_offset_front = indices_front[od];
            args.src_offset_back = indices_back[od];
            args.src_offset_top = indices_top[oh];
            args.src_offset_bottom = indices_bottom[oh];
            args.weight_front = weights_front[od];
            args.weight_back = weights_back[od];
            args.weight_top = weights_top[oh];
            args.weight_bottom = weights_bottom[oh];

            (*kernel_)(&args);
        });
    } else {
        assert(!"Invalid memory format kind.");
        return status::invalid_arguments;
    }

    return status::success;
}

template struct jit_uni_resampling_fwd_t<sse41>;
template struct jit_uni_resampling_fwd_t<avx>;
template struct jit_uni_resampling_fwd_t<avx512_common>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
