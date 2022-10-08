/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#include "dnnl_graph_common.hpp"
#include "tests/test_thread.hpp"

#include "mlp/mlp.hpp"
#include "utils/parser.hpp"

namespace mlp {

#define BIAS_MASK_DEFAULT 2

void compute_ref_mlp(
        const mlp_graph_spec_t *spec, const std::vector<args_t> &args) {
    for (size_t i = 0; i < spec->activation_func.size(); i++) {
        vdims_t strides {vdims_t(STRIDES_SIZE)};
        std::vector<::matmul::dims_mask_t> rt_dims_masks {};
        int bias_mask = BIAS_MASK_DEFAULT;
        prb_vdims_t prb_vdims;
        std::string dims_str = dims2str(spec->layer_dims[i]);
        dims_str = dims_str + ":" + dims2str(spec->weight_dims[i]);
        //pick only the activation function for that layer.
        attr_t attr;
        attr.post_ops.entry.push_back(spec->attr.post_ops.entry[i]);
        attr.insert(spec->attr.oscale);
        attr.insert(spec->attr.zero_points);
        float *scales = (float *)zmalloc(sizeof(float), 4);
        SAFE_V(scales != nullptr ? OK : FAIL);
        scales[0] = spec->attr.oscale.scale;

        ::parser::parse_prb_vdims(prb_vdims, dims_str);
        std::vector<dnnl_data_type_t> dt_vec;
        handle_legacy_cfg(dt_vec, spec->cfg);
        ::matmul::prb_t matmul_prb(prb_vdims, dt_vec, spec->raw_data_tag,
                spec->raw_wei_tag, spec->raw_data_tag, strides,
                benchdnnext::convert_dt(spec->mlp_bias_dt), bias_mask,
                rt_dims_masks, attr, default_thr_ctx, default_thr_ctx);
        matmul_prb.scales = scales;

        ::matmul::compute_ref(&matmul_prb, args[i], NULL);

        const dnn_mem_t &dst_m = args[i].find(DNNL_ARG_DST);
        assert(dst_m.nelems() != 0);
        const auto dt = benchdnnext::convert_dt(spec->mlp_src_dt);
        dnnl::impl::parallel_nd(dst_m.nelems(), [&](int64_t i) {
            //below adjustment needed for input for the next layer.
            ((float *)dst_m)[i]
                    = round_to_nearest_representable(dt, dst_m.get_elem(i));
        });
    }
}

} // namespace mlp
