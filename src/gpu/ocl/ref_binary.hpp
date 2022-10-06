/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#ifndef GPU_OCL_REF_BINARY_HPP
#define GPU_OCL_REF_BINARY_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_binary_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ref_binary_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_binary_pd_t {
        using gpu_binary_pd_t::gpu_binary_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_binary_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            const auto attr_skip_mask = sm::post_ops | sm::scales_runtime;

            bool ok = set_default_params() == status::success
                    && ((utils::everyone_is(bf16, src_md(0)->data_type,
                                 src_md(1)->data_type)
                                && utils::one_of(dst_md()->data_type, bf16, u8))
                            || (utils::one_of(
                                        src_md(0)->data_type, f16, f32, s8, u8)
                                    && utils::one_of(src_md(1)->data_type, f16,
                                            f32, s8, u8)
                                    && utils::one_of(dst_md()->data_type, f16,
                                            f32, s8, u8)))
                    && !memory_desc_ndims_ok(src_md(0), src_md(1), dst_md())
                    && IMPLICATION(!attr()->scales_.has_default_values(),
                            check_scales_mask())
                    && attr()->has_default_values(attr_skip_mask)
                    && post_ops_with_binary_ok(
                            attr(), dst_md()->data_type, MAX_NDIMS)
                    && attr_.set_default_formats(dst_md(0)) == status::success
                    && !(attr()->post_ops_.len() > 0
                            && src_md(0)->data_type == bf16
                            && src_md(1)->data_type == bf16
                            && dst_md()->data_type == u8);

            if (!ok) return status::unimplemented;

            status_t status = init_scales_md();
            if (status != status::success) return status;
            return init_conf(engine);
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        bool with_scales(int position) const {
            return !attr()->scales_.get(position).has_default_values();
        }

        bool with_scales() const {
            return with_scales(DNNL_ARG_SRC_0) || with_scales(DNNL_ARG_SRC_1);
        }

        float get_scale(int position) const {
            return *attr()->scales_.get(position).scales_;
        }

        bool with_eltwise(int position) const {
            return attr()->post_ops_.contain(primitive_kind::eltwise, position);
        }

        bool with_sum() const {
            return attr()->post_ops_.find(primitive_kind::sum) != -1;
        }

        float eltwise_alpha() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alpha
                    : 1.0f;
        }

        float eltwise_beta() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.beta
                    : 0.0f;
        }

        float eltwise_scale() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.scale
                    : 1.0f;
        }

        float sum_scale() const {
            const int sum_idx = attr()->post_ops_.find(primitive_kind::sum);
            return sum_idx != -1 ? attr()->post_ops_.entry_[sum_idx].sum.scale
                                 : 0.0f;
        }

        alg_kind_t eltwise_alg_kind() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alg
                    : dnnl_alg_kind_undef;
        }

        binary_conf_t conf;

        const memory_desc_t *src0_scale_md() const { return &src0_scale_md_; }
        const memory_desc_t *src1_scale_md() const { return &src1_scale_md_; }

    private:
        bool check_scales_mask() const {
            for (const auto &s : attr()->scales_.scales_) {
                if (s.second.mask_ != 0) return false;
            }
            return true;
        }

        memory_desc_t src0_scale_md_ = memory_desc_t();
        memory_desc_t src1_scale_md_ = memory_desc_t();

        status_t init_scales_md() {
            src0_scale_md_.data_type = data_type::f32;
            src1_scale_md_.data_type = data_type::f32;
            src0_scale_md_.ndims = 1;
            src1_scale_md_.ndims = 1;
            src0_scale_md_.dims[0] = 1;
            src1_scale_md_.dims[0] = 1;
            auto status
                    = memory_desc_init_by_tag(src0_scale_md_, format_tag::x);
            if (status != status::success) return status;
            return (memory_desc_init_by_tag(src1_scale_md_, format_tag::x));
        }
    };

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        create_kernel(engine, &kernel_, "ref_binary", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

    status_t handle_runtime_value(engine_t *engine, int idx,
            const memory_desc_t *md,
            std::unique_ptr<memory_storage_t> &mem_storage) const {
        const primitive_attr_t &attr = *pd()->attr();
        void *p;
        memory_desc_wrapper mdw(*md);
        size_t sz = sizeof(float);
        memory_storage_t *mem_s_ptr;
        status_t status
                = engine->create_memory_storage(&mem_s_ptr, mdw.nelems() * sz);
        if (status != status::success) {
            mem_storage.reset();
            return status;
        }
        mem_storage.reset(mem_s_ptr);
        assert(sizeof(float) == sizeof(int));
        status = mem_storage->map_data(
                &p, nullptr, sizeof(float) * mdw.nelems());
        if (status != status::success) return status;
        if (attr.scales_.has_default_values()) {
            utils::array_set((float *)p, (float)1, mdw.nelems());
        } else {
            switch (idx) {
                case SRC0_SCALE_:
                    utils::array_copy((float *)p,
                            (float *)attr.scales_.get(DNNL_ARG_SRC_0).scales_,
                            mdw.nelems());
                    break;
                case SRC1_SCALE_:
                    utils::array_copy((float *)p,
                            (float *)attr.scales_.get(DNNL_ARG_SRC_1).scales_,
                            mdw.nelems());
                    break;
            }
        }
        status = mem_storage->unmap_data(p, nullptr);
        return status;
    }

protected:
    status_t init_res_storage(
            engine_t *engine, gpu_resource_t *r) const override {
        std::unique_ptr<memory_storage_t> tmp_mem_storage;

        CHECK(handle_runtime_value(
                engine, SRC0_SCALE_, pd()->src0_scale_md(), tmp_mem_storage));
        r->add_memory_storage(SRC0_SCALE_, std::move(tmp_mem_storage));

        CHECK(handle_runtime_value(
                engine, SRC1_SCALE_, pd()->src1_scale_md(), tmp_mem_storage));
        r->add_memory_storage(SRC1_SCALE_, std::move(tmp_mem_storage));
        return status::success;
    }

private:
    status_t execute_ref(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
    enum { SRC0_SCALE_ = 0, SRC1_SCALE_ = 1 };
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
