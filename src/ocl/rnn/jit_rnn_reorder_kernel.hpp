/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef JIT_RNN_REORDER_KERNEL_HPP
#define JIT_RNN_REORDER_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/reorder_pd.hpp"
#include "compute/compute.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

struct jit_rnn_reorder_kernel {

    jit_rnn_reorder_kernel(const jit_rnn_reorder_conf_t &ajrp) : jrp(ajrp) {}

    ~jit_rnn_reorder_kernel() {}

    static status_t init_conf(
            jit_rnn_reorder_conf_t &jrp, const reorder_pd_t *pd) {

        const memory_desc_wrapper input_md(pd->src_md());
        const memory_desc_wrapper output_md(pd->dst_md());

        status_t status = status::success;

        const auto &dims = output_md.padded_dims();
        jrp.with_sum_ab = (pd->alpha() != 1.f || pd->beta() != 0.f);
        jrp.with_sum_a = jrp.with_sum_ab && pd->beta() == 0.f;
        jrp.do_reorder = input_md != output_md;
        jrp.has_padding = !input_md.is_dense() || !output_md.is_dense();
        jrp.ndims = input_md.ndims();
        jrp.nelems = utils::array_product(dims, jrp.ndims);
        jrp.lws_d[0] = 1;
        jrp.lws_d[1] = 1;
        jrp.lws_d[2] = 1;

        jrp.use_ref_impl = 1;
        jrp.with_group = 0;
        jrp.sub_group_size = 1;

        // only for LDIGO
        jrp.gws_d[0] = dims[0] * dims[1];
        jrp.gws_d[1] = dims[3] * dims[4];
        jrp.gws_d[2] = 1;

        jrp.mask = pd->attr()->rnn_weights_qparams_.mask_;
        const auto &input_dims = input_md.dims();
        jrp.scales_count = jrp.mask ? input_dims[3] * input_dims[4] : 1;

        return status;
    };

    static status_t init_const_def(compute::kernel_ctx_t &kernel_ctx,
            const jit_rnn_reorder_conf_t &jrp,
            const memory_desc_wrapper &input_md,
            const memory_desc_wrapper &output_md) {

        kernel_ctx.define_int("NDIMS", jrp.ndims);
        if (jrp.with_sum_a)
            kernel_ctx.define_int("WITH_SUM_A", 1);
        else if (jrp.with_sum_ab)
            kernel_ctx.define_int("WITH_SUM_AB", 1);
        kernel_ctx.define_int("WITH_GROUP", jrp.with_group);

        kernel_ctx.define_int("LWS_0", jrp.lws_d[0]);
        kernel_ctx.define_int("LWS_1", jrp.lws_d[1]);
        kernel_ctx.define_int("LWS_2", jrp.lws_d[2]);

        auto input_type = input_md.data_type();
        auto output_type = output_md.data_type();

        switch (input_type) {
            case dnnl_u8: kernel_ctx.define_int("IN_TYPE_U8", 1); break;
            case dnnl_s8: kernel_ctx.define_int("IN_TYPE_S8", 1); break;
            case dnnl_f16: kernel_ctx.define_int("IN_TYPE_F16", 1); break;
            case dnnl_s32: kernel_ctx.define_int("IN_TYPE_S32", 1); break;
            case dnnl_f32: kernel_ctx.define_int("IN_TYPE_F32", 1); break;
            case dnnl_bf16: kernel_ctx.define_int("IN_TYPE_BF16", 1); break;
            default: return status::invalid_arguments;
        }
        switch (output_type) {
            case dnnl_u8: kernel_ctx.define_int("OUT_TYPE_U8", 1); break;
            case dnnl_s8: kernel_ctx.define_int("OUT_TYPE_S8", 1); break;
            case dnnl_f16: kernel_ctx.define_int("OUT_TYPE_F16", 1); break;
            case dnnl_s32: kernel_ctx.define_int("OUT_TYPE_S32", 1); break;
            case dnnl_f32: kernel_ctx.define_int("OUT_TYPE_F32", 1); break;
            case dnnl_bf16: kernel_ctx.define_int("OUT_TYPE_BF16", 1); break;
            default: return status::invalid_arguments;
        }

        kernel_ctx.define_int("REF_REORDER", jrp.use_ref_impl);
        kernel_ctx.define_int("SUB_GROUP_SIZE", jrp.sub_group_size);

        set_offsets(kernel_ctx, input_md, "SRC");
        set_offsets(kernel_ctx, output_md, "DST");

        const auto &in_dims = input_md.dims();
        const auto &out_dims = output_md.padded_dims();

        kernel_ctx.define_int("PAD_FILL_ZERO", jrp.has_padding);
        for (int d = 0; d < MAX_NDIMS; ++d)
            kernel_ctx.define_int(utils::format("SRC_D%d", d),
                    (d < input_md.ndims()) ? in_dims[d] : 1);
        for (int d = 0; d < MAX_NDIMS; ++d)
            kernel_ctx.define_int(utils::format("DST_D%d", d),
                    (d < output_md.ndims()) ? out_dims[d] : 1);
        kernel_ctx.define_int("MASK", jrp.mask);
        return status::success;
    }

    jit_rnn_reorder_conf_t jrp;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif
