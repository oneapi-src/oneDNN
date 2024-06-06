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

#ifndef GPU_INTEL_OCL_MULTI_PO_REORDER_BINARY_HPP
#define GPU_INTEL_OCL_MULTI_PO_REORDER_BINARY_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/reorder.hpp"
#include "common/reorder_pd.hpp"
#include "common/stream.hpp"
#include "gpu/gpu_binary_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct multi_po_reorder_binary : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_binary_pd_t {
        using gpu_binary_pd_t::gpu_binary_pd_t;

        DECLARE_COMMON_PD_T("multi_po_reorder_binary", multi_po_reorder_binary);

        status_t init(impl::engine_t *engine) {
            if (attr()->scales_.get(DNNL_ARG_SRC_0).is_set_
                    || attr()->scales_.get(DNNL_ARG_SRC_1).is_set_
                    || attr()->post_ops_.len() >= 1) {
                VDISPATCH_BINARY(false, VERBOSE_UNSUPPORTED_ATTR);
            }

            // Assumption: src_mds have different layouts with dst mem
            // descriptor matching only with one of the src mem descriptors
            // or, all of them have the same memory layout
            bool need_output_reorder
                    = (!dnnl_memory_desc_equal(src_md(0), src_md(1))
                              && (dnnl_memory_desc_equal(src_md(0), dst_md())
                                      || dnnl_memory_desc_equal(
                                              src_md(1), dst_md())))
                    || (dnnl_memory_desc_equal(src_md(0), src_md(1))
                            && dnnl_memory_desc_equal(src_md(0), dst_md()));

            // Assumption: src_mds have different layouts with dst_md matching one of src_md
            need_input_reorder = dnnl_memory_desc_equal(src_md(0), src_md(1))
                    && !dnnl_memory_desc_equal(src_md(0), dst_md());

            if ((!need_output_reorder && !need_input_reorder)
                    || is_broadcast(src_md(0), src_md(1))) {
                VDISPATCH_BINARY(false, VERBOSE_INCONSISTENT_NDIMS,
                        "src(0) src(1)", "dst");
            }

            alg_kind_t binary_alg = desc()->alg_kind;
            primitive_attr_t attr;

            reorder_pd_list.emplace_back(nullptr);

            if (need_input_reorder) {
                if (binary_alg != dnnl::impl::alg_kind::binary_add) {
                    VDISPATCH_BINARY(false, VERBOSE_BAD_ALGORITHM);
                }

                VDISPATCH_BINARY_SC(
                        reorder_primitive_desc_create(reorder_pd_list.back(),
                                engine, src_md(0), dst_md()),
                        VERBOSE_PRIMITIVE_CREATION_FAIL, "reorder");

                reorder_pd_list.emplace_back(nullptr);

                VDISPATCH_BINARY_SC(attr.post_ops_.append_sum(1.f),
                        VERBOSE_UNSUPPORTED_ATTR);

                VDISPATCH_BINARY_SC(
                        reorder_primitive_desc_create(reorder_pd_list.back(),
                                engine, src_md(1), dst_md(), &attr),
                        VERBOSE_PRIMITIVE_CREATION_FAIL, "reorder");
            } else { //need_output_reorder else-block
                switch (binary_alg) {
                    case alg_kind::binary_add:
                    case alg_kind::binary_mul:
                    case alg_kind::binary_min:
                    case alg_kind::binary_max: break;
                    default: VDISPATCH_BINARY(false, VERBOSE_BAD_ALGORITHM);
                };

                // Check for memory descriptor layours for SRC and DST
                // Matching layouts will not cause a reorder
                src_index = dnnl_memory_desc_equal(src_md(0), dst_md()) ? 0 : 1;

                VDISPATCH_BINARY_SC(attr.post_ops_.append_binary(
                                            binary_alg, src_md(src_index)),
                        VERBOSE_UNSUPPORTED_ATTR);

                VDISPATCH_BINARY_SC(
                        reorder_primitive_desc_create(reorder_pd_list.back(),
                                engine, src_md(!src_index), dst_md(), &attr),
                        VERBOSE_PRIMITIVE_CREATION_FAIL, "reorder");
            }

            return status::success;
        }

        std::vector<std::shared_ptr<primitive_desc_t>> reorder_pd_list;
        bool need_input_reorder = false;
        int src_index = -1;

    private:
        bool is_broadcast(
                const memory_desc_t *src0_md, const memory_desc_t *src1_md) {
            bool res = false;
            for (int i = 0; i < src0_md->ndims; i++) {
                if (src0_md->dims[i] != src1_md->dims[i]) { return res = true; }
            }
            return res;
        }
    };

    status_t init(impl::engine_t *engine) override {
        int size = pd()->need_input_reorder ? 2 : 1;
        reorder_primitive_list.resize(size);
        for (int i = 0; i < size; ++i) {
            CHECK(create_nested_primitive(reorder_primitive_list[i],
                    pd()->reorder_pd_list[i], engine));
        }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        for (int id = 0; id < int(reorder_primitive_list.size()); id++) {
            exec_args_t r_args;
            auto dst = ctx.args().at(DNNL_ARG_DST);
            r_args[DNNL_ARG_DST] = dst;

            if (pd()->need_input_reorder) {
                r_args[DNNL_ARG_SRC] = id == 0 ? ctx.args().at(DNNL_ARG_SRC_0)
                                               : ctx.args().at(DNNL_ARG_SRC_1);
            } else {
                r_args[DNNL_ARG_SRC] = !pd()->src_index
                        ? ctx.args().at(DNNL_ARG_SRC_1)
                        : ctx.args().at(DNNL_ARG_SRC_0);

                r_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1]
                        = pd()->src_index ? ctx.args().at(DNNL_ARG_SRC_1)
                                          : ctx.args().at(DNNL_ARG_SRC_0);
            }
            exec_ctx_t r_ctx(ctx, std::move(r_args));
            CHECK(reorder_primitive_list[id]->execute(r_ctx));
        }
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<std::shared_ptr<impl::primitive_t>> reorder_primitive_list;
};
} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
