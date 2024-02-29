/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GPU_OCL_GEMM_MATMUL_HPP
#define GPU_OCL_GEMM_MATMUL_HPP

#include "common/gemm_utils.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_matmul_pd.hpp"
#include "gpu/gpu_primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gemm_matmul_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_matmul_pd_t {
        pd_t(const matmul_desc_t *adesc, const primitive_attr_t *attr,
                const matmul_pd_t *hint_pd)
            : gpu_matmul_pd_t(adesc, attr, hint_pd) {}

        pd_t(const pd_t &other) = default;

        DECLARE_COMMON_PD_T(gemm_pd_->name(), gemm_matmul_t);

        status_t init(engine_t *engine) {
            using namespace data_type;

            primitive_attr_t gemm_attr;
            if (!attr()->scales_.has_default_values()) {
                gemm_attr.scales_ = attr()->scales_;
            }

            auto post_ops = attr()->post_ops_;
            auto a_md = src_md(), b_md = weights_md(), c_md = dst_md(),
                 bias_md = weights_md(1);
            const auto acc_dt = desc()->accum_data_type;
            memory_desc_t a_md_reshaped, b_md_reshaped, c_md_reshaped,
                    bia_md_reshaped;
            bool with_bia = bias_md->ndims > 0;
            auto orig_dims = a_md->ndims;

            auto map_gemm_zp = [&](int arg, int gemm_arg, bool reshape = false,
                                       int diff_dims = 0) {
                if (!attr()->zero_points_.has_default_values(arg)) {
                    int mask = 0;
                    CHECK(attr()->zero_points_.get(arg, &mask));
                    if (reshape) mask = mask >> diff_dims;
                    CHECK(gemm_attr.zero_points_.set(arg, mask, 0, nullptr,
                            attr()->zero_points_.get_data_type(arg)));
                }
                return status::success;
            };

            auto adjust_scales_mask
                    = [&](arg_scales_t &scales, int arg, int diff_dims) {
                          int mask = 0;
                          bool is_set = false;
                          CHECK(attr()->scales_.get(arg, &mask, &is_set));
                          mask = mask >> diff_dims;
                          if (is_set) {
                              CHECK(scales.set(arg, mask, 0, nullptr,
                                      attr()->scales_.get(arg).data_type_));
                          }
                          return status::success;
                      };
            if (!attr()->zero_points_.has_default_values()) {
                CHECK(map_gemm_zp(DNNL_ARG_SRC, DNNL_ARG_B));
                CHECK(map_gemm_zp(
                        DNNL_ARG_WEIGHTS, DNNL_ARG_A, false, orig_dims - 2));
                CHECK(map_gemm_zp(DNNL_ARG_DST, DNNL_ARG_C));
            }

            auto maybe_reshape = [&](dims_t &orig_a_dims, dims_t &orig_b_dims,
                                         dims_t &orig_c_dims,
                                         dims_t &orig_bias_dims,
                                         int &orig_dims) {
                int batch_b_dims = 1;
                for (int i = b_md->ndims; i > 2; i--) {
                    batch_b_dims *= b_md->dims[b_md->ndims - i];
                }
                for (int i = 0; i < orig_dims; i++) {
                    orig_a_dims[i] = a_md->dims[i];
                    orig_b_dims[i] = b_md->dims[i];
                    orig_c_dims[i] = c_md->dims[i];
                    orig_bias_dims[i] = bias_md->dims[i];
                }
                //for batch dim can map broadcast to 2d: eg. 4x1x4096:1x4096x16 -> 4x4096:4096x16
                auto reshape_2d = (batch_b_dims == 1 && b_md->ndims > 2);
                auto reshape_3d = a_md->ndims > 3;
                if (reshape_2d || reshape_3d) {
                    auto ndims = a_md->ndims;
                    auto reshape_size = reshape_2d ? 2 : 3;
                    dim_t a_dim = a_md->dims[a_md->ndims - reshape_size],
                          b_dim = b_md->dims[b_md->ndims - 1],
                          bia_dim
                            = bias_md->dims[bias_md->ndims - reshape_size];
                    bool with_bia = bias_md->ndims > 0;
                    for (int i = a_md->ndims; i > reshape_size; i--) {
                        a_dim *= a_md->dims[a_md->ndims - i];
                        bia_dim *= bias_md->dims[bias_md->ndims - i];
                    }
                    if (with_bia) {
                        //bias cannot be applied if applied on only on a subset of batch dims
                        if (bia_dim > 1 && bia_dim != a_dim)
                            return status::unimplemented;
                    }
                    dims_t a_dims, b_dims, c_dims, bia_dims;
                    if (reshape_2d) {
                        a_dims[0] = a_dim;
                        a_dims[1] = a_md->dims[a_md->ndims - 1];
                        b_dims[0] = b_md->dims[b_md->ndims - 2];
                        b_dims[1] = b_dim;
                        c_dims[0] = a_dims[0];
                        c_dims[1] = b_dims[1];
                        bia_dims[0] = bia_dim;
                        bia_dims[1] = with_bia
                                ? bias_md->dims[bias_md->ndims - 1]
                                : 1;
                    } else {
                        a_dims[0] = a_dim;
                        a_dims[1] = a_md->dims[ndims - 2];
                        a_dims[2] = a_md->dims[ndims - 1];
                        b_dims[0] = a_dim;
                        b_dims[1] = b_md->dims[ndims - 2];
                        b_dims[2] = b_md->dims[ndims - 1];
                        c_dims[0] = a_dim;
                        c_dims[1] = a_dims[1];
                        c_dims[2] = b_dims[2];
                        bia_dims[0] = bia_dim;
                        bia_dims[1] = with_bia ? bias_md->dims[ndims - 2] : 1;
                        bia_dims[2] = with_bia ? bias_md->dims[ndims - 1] : 1;
                    }
                    CHECK(memory_desc_reshape(
                            a_md_reshaped, *a_md, reshape_size, a_dims));
                    CHECK(memory_desc_reshape(
                            b_md_reshaped, *b_md, reshape_size, b_dims));
                    CHECK(memory_desc_reshape(
                            c_md_reshaped, *c_md, reshape_size, c_dims));
                    if (with_bia) {
                        CHECK(memory_desc_reshape(bia_md_reshaped, *bias_md,
                                reshape_size, bia_dims));
                    }
                    auto tmp_post_ops = post_ops;
                    auto scales = gemm_attr.scales_;
                    for (int i = 0; i < attr()->post_ops_.len(); i++) {
                        auto &po = post_ops.entry_[i];
                        if (po.is_binary()) {
                            auto &po_desc = po.binary.src1_desc;
                            auto a_dim = po_desc.dims[po_desc.ndims
                                    - reshape_size];
                            for (int i = po_desc.ndims; i > reshape_size; i--) {
                                a_dim *= po_desc.dims[po_desc.ndims - i];
                            }
                            //post ops cannot be applied if applied on only on a subset of batch dims
                            if (a_dim != c_dims[0] && a_dim > 1) {
                                return status::unimplemented;
                            }
                            auto has_dims = po_desc.ndims > 0;
                            dims_t po_dims;
                            if (reshape_2d) {
                                po_dims[0] = a_dim;
                                po_dims[1] = has_dims
                                        ? po_desc.dims[po_desc.ndims - 1]
                                        : 1;
                            } else {
                                po_dims[0] = a_dim;
                                po_dims[1] = has_dims
                                        ? po_desc.dims[po_desc.ndims - 2]
                                        : 1;
                                po_dims[2] = has_dims
                                        ? po_desc.dims[po_desc.ndims - 1]
                                        : 1;
                            }
                            CHECK(memory_desc_reshape(
                                    po_desc, po_desc, reshape_size, po_dims));
                            tmp_post_ops.entry_[i].binary.src1_desc = po_desc;
                        } else if (po.is_prelu()) {
                            auto mask = po.prelu.mask;
                            int new_mask = 0;
                            int batch_idx = reshape_size - 1;
                            int batch_dim = 1;
                            int mask_dim = 1;
                            //get mask for batch dim
                            for (int i = 0; i < c_md->ndims - batch_idx; i++) {
                                if (mask >> i & 1) {
                                    //post ops cannot be applied if applied on only on a subset of batch dims
                                    if (new_mask != 0)
                                        return status::unimplemented;
                                    new_mask |= c_md->dims[i] == 1 ? 0 : 1;
                                    mask_dim *= c_md->dims[i];
                                }
                                batch_dim *= c_md->dims[i];
                            }
                            //post ops cannot be applied if applied on only on a subset of batch dims
                            if (batch_dim != mask_dim)
                                return status::unimplemented;
                            //get non-batch part of mask
                            auto shift = c_md->ndims - batch_idx;
                            auto non_batch_mask = mask >> shift;
                            //due to prelu being in axb format, if a reshape is done it
                            //implies layout is different e.g 1x30x20 -> 30 is innermost dimension
                            //but 30x20 -> 20 is innermost. Hence reshape does  not work if mask
                            //is applied across more than one dimension.
                            if (non_batch_mask > 2
                                    || (non_batch_mask > 0 && new_mask > 0))
                                return status::unimplemented;
                            new_mask |= non_batch_mask << 1;
                            tmp_post_ops.entry_[i].prelu.mask = new_mask;
                        }
                    }

                    if (!attr()->scales_.has_default_values())
                        CHECK(adjust_scales_mask(scales, DNNL_ARG_WEIGHTS,
                                orig_dims - reshape_size));
                    if (!attr()->zero_points_.has_default_values())
                        CHECK(map_gemm_zp(DNNL_ARG_WEIGHTS, DNNL_ARG_A, true,
                                orig_dims - reshape_size));
                    post_ops = tmp_post_ops;
                    gemm_attr.scales_ = scales;
                    a_md = &a_md_reshaped;
                    b_md = &b_md_reshaped;
                    c_md = &c_md_reshaped;
                    if (with_bia) bias_md = &bia_md_reshaped;
                }
                return (!reshape_2d && !reshape_3d) ? status::unimplemented
                                                    : status::success;
            };

            CHECK(gemm_attr.set_fpmath_mode(
                    attr()->fpmath_.mode_, attr()->fpmath_.apply_to_int_));
            gemm_attr.deterministic_ = attr()->deterministic_;

            dims_t orig_a_dims, orig_b_dims, orig_c_dims, orig_bias_dims;
            bool reshape = maybe_reshape(orig_a_dims, orig_b_dims, orig_c_dims,
                                   orig_bias_dims, orig_dims)
                    == status::success;

            if (!attr()->post_ops_.has_default_values()) {
                gemm_attr.post_ops_ = post_ops;
            }

            // We create a gemm_pd and resolve 'any' desc by querying gemm_pd
            VDISPATCH_MATMUL(
                    is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL_SC(create_gemm_pd(gemm_pd_, engine, a_md, b_md,
                                        c_md, bias_md, acc_dt, &gemm_attr),
                    VERBOSE_PRIMITIVE_CREATION_FAIL, "gemm");
            VDISPATCH_MATMUL_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);

            if (reshape) {
                CHECK(memory_desc_reshape(
                        src_md_, src_md_, orig_dims, orig_a_dims));
                CHECK(memory_desc_reshape(
                        weights_md_, weights_md_, orig_dims, orig_b_dims));
                CHECK(memory_desc_reshape(
                        dst_md_, dst_md_, orig_dims, orig_c_dims));
                if (with_bia)
                    CHECK(memory_desc_reshape(
                            bias_md_, bias_md_, orig_dims, orig_bias_dims));
            }
            init_scratchpad();

            return status::success;
        }

        std::shared_ptr<primitive_desc_t> gemm_pd_;

    private:
        status_t set_default_params() {
            src_md_ = *gemm_pd_->arg_md(DNNL_ARG_SRC_0);
            weights_md_ = *gemm_pd_->arg_md(DNNL_ARG_SRC_1);
            bias_md_ = *gemm_pd_->arg_md(DNNL_ARG_BIAS);
            dst_md_ = *gemm_pd_->arg_md(DNNL_ARG_DST);
            return status::success;
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    gemm_pd_->scratchpad_registry());
        }
    };

    status_t init(engine_t *engine) override {
        return create_nested_primitive(gemm_, pd()->gemm_pd_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> gemm_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
