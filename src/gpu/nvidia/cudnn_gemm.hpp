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

#ifndef GPU_NVIDIA_GEMM_HPP
#define GPU_NVIDIA_GEMM_HPP

#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/gpu_gemm_pd.hpp"
#include "gpu/intel/gemm/gpu_gemm.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_gemm_t : public intel::gpu_gemm_t {
    using intel::gpu_gemm_t::gpu_gemm_t;
    struct pd_t : public gpu_gemm_pd_t {
        using gpu_gemm_pd_t::gpu_gemm_pd_t;

        DECLARE_COMMON_PD_T("nvidia:cudnn:any", cudnn_gemm_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            auto mm_desc = matmul_desc_t();
            auto src_desc = src_md(0);
            auto wei_desc = src_md(1);
            auto bias_desc = src_md(2);
            auto dst_desc = dst_md();


            auto has_rt_dims = [&](const memory_desc_t * md){
                return memory_desc_wrapper(md).has_runtime_dims_or_strides();
            };            


            auto has_runtime_params = has_rt_dims(src_desc)
                || has_rt_dims(dst_desc)
                || has_rt_dims(wei_desc);
            
            if (has_runtime_params){
                return status::unimplemented;
            }

            mm_desc.src_desc = *src_desc;
            mm_desc.weights_desc = *wei_desc;
            mm_desc.bias_desc = *bias_desc;
            mm_desc.dst_desc = *dst_desc;

            mm_desc.primitive_kind = primitive_kind::matmul; 

            primitive_attr_t mm_attr = *attr();

            if (!attr()->post_ops_.has_default_values()) {
                mm_attr.post_ops_ = attr()->post_ops_;
            }
            mm_attr.scales_= attr()->scales_;

            CHECK(mm_attr.set_fpmath_mode(
                    attr()->fpmath_.mode_, attr()->fpmath_.apply_to_int_));

            mm_attr.deterministic_ = attr()->deterministic_;


            primitive_desc_iterator_t it(
                engine, (op_desc_t *)&mm_desc, &mm_attr, nullptr);

            mm_pd_ = *(++it);
            if (!mm_pd_) return status::unimplemented;

            return status::success;
        }


        std::shared_ptr<primitive_desc_t> mm_pd_;
    };

    status_t init(impl::engine_t *engine) override {
        return create_nested_primitive(matmul_, pd()->mm_pd_, engine);
    }

    status_t execute(const intel::gemm_exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> matmul_;
};

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
