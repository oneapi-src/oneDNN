/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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
            memory_desc_t a_md_2d, b_md_2d, c_md_2d, bia_md_2d;
            bool with_bia = bias_md->ndims > 0;

            auto map_gemm_zp = [&](int arg, int gemm_arg, bool reshape = false,
                                       int diff_dims = 0) {
                if (!attr()->zero_points_.has_default_values(arg)) {
                    int mask = 0;
                    CHECK(attr()->zero_points_.get(arg, &mask));
                    if (reshape) mask = mask >> diff_dims;
                    CHECK(gemm_attr.zero_points_.set(gemm_arg, mask));
                }
                return status::success;
            };

            auto adjust_scales_mask = [&](int arg, int diff_dims) {
                int mask = 0;
                bool is_set = false;
                CHECK(attr()->scales_.get(arg, &mask, &is_set));
                mask = mask >> diff_dims;
                if (is_set) { CHECK(gemm_attr.scales_.set(arg, mask)); }
                return status::success;
            };

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
                auto reshape = batch_b_dims == 1 && b_md->ndims > 2;
                if (reshape) {
                    dim_t a_dim = a_md->dims[a_md->ndims - 2],
                          b_dim = b_md->dims[b_md->ndims - 1],
                          bia_dim = bias_md->dims[bias_md->ndims - 2];
                    bool with_bia = bias_md->ndims > 0;
                    for (int i = a_md->ndims; i > 2; i--) {
                        a_dim *= a_md->dims[a_md->ndims - i];
                        bia_dim *= bias_md->dims[bias_md->ndims - i];
                    }
                    dims_t a_dims = {a_dim, a_md->dims[a_md->ndims - 1]};
                    dims_t b_dims = {b_md->dims[b_md->ndims - 2], b_dim};
                    dims_t c_dims = {a_dims[0], b_dims[1]};
                    dims_t bia_dims = {bia_dim,
                            with_bia ? bias_md->dims[bias_md->ndims - 1] : 1};
                    CHECK(memory_desc_reshape(a_md_2d, *a_md, 2, a_dims));
                    CHECK(memory_desc_reshape(b_md_2d, *b_md, 2, b_dims));
                    CHECK(memory_desc_reshape(c_md_2d, *c_md, 2, c_dims));
                    if (with_bia) {
                        CHECK(memory_desc_reshape(
                                bia_md_2d, *bias_md, 2, bia_dims));
                    }
                    for (int i = 0; i < attr()->post_ops_.len(); i++) {
                        auto &po = post_ops.entry_[i];
                        if (po.is_binary()) {
                            auto &po_desc = po.binary.src1_desc;
                            auto a_dim = po_desc.dims[po_desc.ndims - 2];
                            for (int i = po_desc.ndims; i > 2; i--) {
                                a_dim *= po_desc.dims[po_desc.ndims - i];
                            }
                            if (a_dim != c_dims[0] && a_dim != 1) {
                                return status::unimplemented;
                            }
                            dims_t po_dims = {a_dim,
                                    po_desc.ndims > 0
                                            ? po_desc.dims[po_desc.ndims - 1]
                                            : 1};
                            CHECK(memory_desc_reshape(
                                    po_desc, po_desc, 2, po_dims));
                            post_ops.entry_[i].binary.src1_desc = po_desc;
                        }
                    }

                    a_md = &a_md_2d;
                    b_md = &b_md_2d;
                    c_md = &c_md_2d;
                    if (with_bia) bias_md = &bia_md_2d;
                }
                return (!reshape) ? status::unimplemented : status::success;
            };

            CHECK(gemm_attr.set_fpmath_mode(attr()->fpmath_mode_));
            auto orig_dims = a_md->ndims;
            dims_t orig_a_dims, orig_b_dims, orig_c_dims, orig_bias_dims;
            bool reshape = maybe_reshape(orig_a_dims, orig_b_dims, orig_c_dims,
                                   orig_bias_dims, orig_dims)
                    == status::success;

            if (!attr()->zero_points_.has_default_values()) {
                CHECK(map_gemm_zp(DNNL_ARG_SRC, DNNL_ARG_B));
                CHECK(map_gemm_zp(
                        DNNL_ARG_WEIGHTS, DNNL_ARG_A, reshape, orig_dims - 2));
                CHECK(map_gemm_zp(DNNL_ARG_DST, DNNL_ARG_C));
            }

            if (!attr()->scales_.has_default_values() && reshape) {
                CHECK(adjust_scales_mask(DNNL_ARG_WEIGHTS, orig_dims - 2));
            }
            if (!attr()->post_ops_.has_default_values()) {
                gemm_attr.post_ops_ = post_ops;
            }

            // We create a gemm_pd and resolve 'any' desc by querying gemm_pd
            bool ok = is_dense_format_kind()
                    && status::success
                            == create_gemm_pd(gemm_pd_, engine, a_md, b_md,
                                    c_md, bias_md, acc_dt, &gemm_attr)
                    && status::success == set_default_params()
                    && attr_.set_default_formats(dst_md(0)) == status::success;
            if (!ok) return status::unimplemented;
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
