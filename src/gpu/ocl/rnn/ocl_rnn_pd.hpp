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

#ifndef GPU_OCL_RNN_OCL_RNN_PD_HPP
#define GPU_OCL_RNN_OCL_RNN_PD_HPP

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/rnn_pd.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ocl_rnn_fwd_pd_t : public rnn_fwd_pd_t {
    using rnn_fwd_pd_t::rnn_fwd_pd_t;
    virtual status_t set_default_params() {
        using namespace format_tag;
        if (src_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(src_layer_md_, tnc));
        if (dst_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(dst_layer_md_, tnc));

        // Optional parameters
        if ((!types::is_zero_md(&src_iter_md_))
                && (src_iter_md_.format_kind == format_kind::any))
            CHECK(memory_desc_init_by_tag(src_iter_md_, ldnc));
        if ((!types::is_zero_md(&src_iter_c_md_))
                && (src_iter_c_md_.format_kind == format_kind::any))
            CHECK(memory_desc_init_by_tag(src_iter_c_md_, ldnc));
        if ((!types::is_zero_md(&bias_md_))
                && (bias_md_.format_kind == format_kind::any))
            CHECK(memory_desc_init_by_tag(bias_md_, ldgo));
        if ((!types::is_zero_md(&dst_iter_md_))
                && (dst_iter_md_.format_kind == format_kind::any))
            CHECK(memory_desc_init_by_tag(dst_iter_md_, ldnc));
        if ((!types::is_zero_md(&dst_iter_c_md_))
                && (dst_iter_c_md_.format_kind == format_kind::any))
            CHECK(memory_desc_init_by_tag(dst_iter_c_md_, ldnc));

        return status::success;
    }
};

struct ocl_rnn_bwd_pd_t : public rnn_bwd_pd_t {
    using rnn_bwd_pd_t::rnn_bwd_pd_t;

protected:
    virtual status_t set_default_params() {
        using namespace format_tag;
        if (src_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(src_layer_md_, tnc));
        if (weights_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(weights_layer_md_, ldgoi));
        if (dst_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(dst_layer_md_, tnc));

        if (weights_iter_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(weights_iter_md_, ldgoi));

        if (diff_src_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_src_layer_md_, tnc));
        if (diff_weights_layer_md_.format_kind == format_kind::any) {
            CHECK(memory_desc_init_by_tag(diff_weights_layer_md_, ldigo));
            CHECK(rnn_utils::set_good_strides(diff_weights_layer_md_, ldigo));
        }
        if (diff_weights_iter_md_.format_kind == format_kind::any) {
            CHECK(memory_desc_init_by_tag(diff_weights_iter_md_, ldigo));
            CHECK(rnn_utils::set_good_strides(diff_weights_iter_md_, ldigo));
        }
        if (diff_dst_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_dst_layer_md_, tnc));

        // Optional parameters
        if ((!types::is_zero_md(&src_iter_md_))
                && (src_iter_md_.format_kind == format_kind::any))
            CHECK(memory_desc_init_by_tag(src_iter_md_, ldnc));
        if ((!types::is_zero_md(&src_iter_c_md_))
                && (src_iter_c_md_.format_kind == format_kind::any))
            CHECK(memory_desc_init_by_tag(src_iter_c_md_, ldnc));
        if ((!types::is_zero_md(&bias_md_))
                && (bias_md_.format_kind == format_kind::any))
            CHECK(memory_desc_init_by_tag(bias_md_, ldgo));
        if ((!types::is_zero_md(&dst_iter_md_))
                && (dst_iter_md_.format_kind == format_kind::any))
            CHECK(memory_desc_init_by_tag(dst_iter_md_, ldnc));
        if ((!types::is_zero_md(&dst_iter_c_md_))
                && (dst_iter_c_md_.format_kind == format_kind::any))
            CHECK(memory_desc_init_by_tag(dst_iter_c_md_, ldnc));

        if ((!types::is_zero_md(&diff_src_iter_md_))
                && (diff_src_iter_md_.format_kind == format_kind::any))
            CHECK(memory_desc_init_by_tag(diff_src_iter_md_, ldnc));
        if ((!types::is_zero_md(&diff_src_iter_c_md_))
                && (diff_src_iter_c_md_.format_kind == format_kind::any))
            CHECK(memory_desc_init_by_tag(diff_src_iter_c_md_, ldnc));
        if ((!types::is_zero_md(&diff_bias_md_))
                && (diff_bias_md_.format_kind == format_kind::any))
            CHECK(memory_desc_init_by_tag(diff_bias_md_, ldgo));
        if ((!types::is_zero_md(&diff_dst_iter_md_))
                && (diff_dst_iter_md_.format_kind == format_kind::any))
            CHECK(memory_desc_init_by_tag(diff_dst_iter_md_, ldnc));
        if ((!types::is_zero_md(&diff_dst_iter_c_md_))
                && (diff_dst_iter_c_md_.format_kind == format_kind::any))
            CHECK(memory_desc_init_by_tag(diff_dst_iter_c_md_, ldnc));

        return status::success;
    }
};
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
