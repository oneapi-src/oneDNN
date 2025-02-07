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

#include "gpu/intel/jit/gemm/jit_gemm_pd.hpp"
#include "gpu/intel/jit/gemm/include/generator.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

status_t jit_gemm_pd_t::init_post_ops() {
    using namespace primitive_kind;
    using namespace alg_kind;
    using namespace data_type;

    const auto d = desc();

    // Examine post-ops and remember binary srcs.
    post_ops_ = attr()->post_ops_;
    binary_srcs_.reserve(post_ops_.len() + 4);

    bool ok = true;
    int prelu_count = 0;
    for (int i = 0; i < post_ops_.len(); i++) {
        const auto &e = post_ops_.entry_[i];
        switch (e.kind) {
            case binary:
                ok &= gemm_kernel_generator_t<ngen::HW::Unknown>::
                                supportedBinaryOp(e.binary.alg)
                        && is_md_gemm_compatible_plain_format(
                                &e.binary.src1_desc);
                binary_srcs_.push_back(
                        binary_src_t {binary_src_t::binary, int(i)});
                break;
            case sum:
                ok &= !with_sum_;
                with_sum_ = true;
                sum_at_begin_ = (i == 0);
                binary_srcs_.push_back(binary_src_t {binary_src_t::none, 0});
                beta_ = e.sum.scale;
                break;
            case eltwise:
                ok &= eltwise_injector_f32_is_supported(e.eltwise.alg);
                binary_srcs_.push_back(binary_src_t {binary_src_t::none, 0});
                break;
            case prelu:
                binary_srcs_.push_back(
                        binary_src_t {binary_src_t::prelu, int(i)});
                ok &= get_prelu_md(e.prelu.mask, dst_md()->dims, prelu_wei_md,
                              dst_md()->ndims)
                        == status::success;
                prelu_count++;
                ok &= prelu_count <= 1;
                break;
            default: return status::unimplemented;
        }
    }

    if (!ok) return status::unimplemented;

    // If scales are present, convert them and any bias to binary post-ops.
    //   Exception: 2D scales.
    // Also convert bias to binary post-op if dst zp are present.
    const auto *wei_scales = &attr()->scales_.get(DNNL_ARG_WEIGHTS);
    const auto *src_scales = &attr()->scales_.get(DNNL_ARG_SRC);
    const auto *c_scales = &attr()->scales_.get(DNNL_ARG_DST);

    bias_via_binary_ = (desc()->bias_type() != data_type::undef)
            && (d->bias_desc.ndims >= 1 || !wei_scales->has_default_values()
                    || !src_scales->has_default_values()
                    || !attr()->zero_points_.has_default_values(DNNL_ARG_DST));
    if (bias_via_binary_) {
        auto status = post_ops_.prepend_binary(binary_add, &d->bias_desc);
        if (status != status::success) return status;
        binary_srcs_.insert(
                binary_srcs_.begin(), binary_src_t {binary_src_t::bias, 0});
    }

    if (!wei_scales->has_default_values()) {
        const auto &mask = wei_scales->get_mask();
        bool convert = (mask == 0 || math::is_pow2(mask));
        if (!wei_scales->has_default_groups())
            convert |= (wei_scales->get_group(0) >= d->k());
        if (convert) {
            dim_t dims = {(mask > 0) ? d->m() : 1};
            CHECK(memory_desc_init_by_tag(wei_scales_md, 1, &dims,
                    wei_scales->get_data_type(), format_tag::a));

            CHECK(post_ops_.prepend_binary(binary_mul, &wei_scales_md));

            binary_srcs_.insert(binary_srcs_.begin(),
                    binary_src_t {binary_src_t::scales, DNNL_ARG_WEIGHTS});
        }
    }
    if (!src_scales->has_default_values()) {
        const auto &mask = src_scales->get_mask();
        bool convert = (mask == 0);
        if (!src_scales->has_default_groups()) {
            convert |= (src_scales->get_group(1) >= d->k());
        }
        if (convert) {
            if (mask == 0) {
                dim_t dims = 1;
                CHECK(memory_desc_init_by_tag(src_scales_md, 1, &dims,
                        src_scales->get_data_type(), format_tag::a));
            } else if (!src_scales->has_default_groups()) {
                // TODO: is it inverted?
                int n_group = src_scales->get_group(0);
                int k_group = src_scales->get_group(1);
                dim_t dims[]
                        = {(mask & (d->batch() > 1 ? 2 : 1)) ? d->n() / n_group
                                                             : 1,
                                d->k() / k_group};
                CHECK(memory_desc_init_by_tag(src_scales_md, 2, dims,
                        src_scales->get_data_type(), format_tag::ab));
            } else {
                dim_t dims[] = {d->n(), 1};
                CHECK(memory_desc_init_by_tag(src_scales_md, 2, dims,
                        src_scales->get_data_type(), format_tag::ab));
            }

            CHECK(post_ops_.prepend_binary(binary_mul, &src_scales_md));

            binary_srcs_.insert(binary_srcs_.begin(),
                    binary_src_t {binary_src_t::scales, DNNL_ARG_SRC});
        }
    }
    if (!c_scales->has_default_values()) {
        const auto &mask = c_scales->get_mask();
        bool convert = (mask == 0 || math::is_pow2(mask));
        if (!c_scales->has_default_groups())
            convert |= (c_scales->get_group(0) >= d->m());
        if (convert) {
            ok = ok && (mask == 0 || mask == (1 << (d->c_desc.ndims - 1)));
            dim_t dims = {(mask > 0) ? d->m() : 1};
            CHECK(memory_desc_init_by_tag(c_scales_md, 1, &dims,
                    c_scales->get_data_type(), format_tag::a));

            CHECK(post_ops_.append_binary(binary_div, &c_scales_md));

            binary_srcs_.push_back(
                    binary_src_t {binary_src_t::scales, DNNL_ARG_DST});
        }
    }

    return status::success;
}

dim_t jit_gemm_pd_t::ld_binary(int idx) const {
    switch (binary_srcs_[idx].type) {
        case binary_src_t::binary: {
            const auto &entry = post_ops_.entry_[idx];
            assert(entry.kind == primitive_kind::binary);
            return gemm_desc_t::get_ld(entry.binary.src1_desc);
        }
        case binary_src_t::bias: return desc()->ld_bias();
        case binary_src_t::prelu: {
            return gemm_desc_t::get_ld(prelu_wei_md);
        }

        default: return 1;
    }
}

dim_t jit_gemm_pd_t::stride_binary(int idx, int stride) const {
    switch (binary_srcs_[idx].type) {
        case binary_src_t::binary:
        case binary_src_t::bias: {
            const auto &entry = post_ops_.entry_[idx];
            assert(entry.kind == primitive_kind::binary);
            return gemm_desc_t::get_stride(entry.binary.src1_desc, stride);
        }
        case binary_src_t::prelu: {
            return gemm_desc_t::get_stride(prelu_wei_md, stride);
        }
        default: return 0;
    }
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
