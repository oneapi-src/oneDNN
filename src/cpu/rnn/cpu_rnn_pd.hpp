/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef CPU_RNN_PD_HPP
#define CPU_RNN_PD_HPP

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "rnn_pd.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "rnn_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_rnn_fwd_pd_t : public rnn_fwd_pd_t {
    using rnn_fwd_pd_t::rnn_fwd_pd_t;

protected:
    status_t set_default_params() {
        using namespace memory_format;
        if (src_layer_md_.format == any)
            CHECK(types::set_default_format(src_layer_md_, tnc));
        if (dst_layer_md_.format == any)
            CHECK(types::set_default_format(dst_layer_md_, tnc));

        // Optional parameters
        if (with_src_iter() && src_iter_md_.format == any)
            CHECK(types::set_default_format(src_iter_md_, ldsnc));
        if (with_bias() && bias_md_.format == any)
            CHECK(types::set_default_format(bias_md_, ldgo));
        if (with_dst_iter() && dst_iter_md_.format == any)
            CHECK(types::set_default_format(dst_iter_md_, ldsnc));

        return status::success;
    }

    status_t check_layout_consistency() {
        using namespace memory_format;
        using namespace data_type;
        using namespace types;
        bool ok = true;
        ok = ok && src_layer_md_.format == tnc
                && dst_layer_md_.format == tnc;
        ok = ok && IMPLICATION(!is_zero_md(&src_iter_md_),
                           src_iter_md_.format == ldsnc)
                && IMPLICATION(!is_zero_md(&dst_iter_md_),
                           dst_iter_md_.format == ldsnc);

        ok = ok && utils::one_of(weights_layer_md_.format, ldigo, rnn_packed)
                && utils::one_of(weights_iter_md_.format, ldigo, rnn_packed);
        ok = ok && IMPLICATION(weights_iter_md_.format == rnn_packed,
                           weights_iter_md_.layout_desc.rnn_packed_desc.format
                                   == mkldnn_ldigo_p);
        ok = ok && IMPLICATION(weights_layer_md_.format == rnn_packed,
                           weights_layer_md_.layout_desc.rnn_packed_desc.format
                                   == mkldnn_ldigo_p);

        ok = ok && IMPLICATION(!is_zero_md(&bias_md_), bias_md_.format == ldgo);

        /* Int8 is supported only for packed weights */
        data_type_t weights_iter_dt = weights_iter_md_.data_type;
        data_type_t weights_layer_dt = weights_layer_md_.data_type;
        ok = ok && IMPLICATION(weights_iter_dt == s8,
                           weights_iter_md_.format == rnn_packed);
        ok = ok && IMPLICATION(weights_layer_dt == s8,
                           weights_layer_md_.format == rnn_packed);

        return ok ? status::success : status::unimplemented;
    }
};

struct cpu_rnn_bwd_pd_t : public rnn_bwd_pd_t {
    using rnn_bwd_pd_t::rnn_bwd_pd_t;

protected:
    status_t set_default_params() {
        using namespace memory_format;
        if (src_layer_md_.format == any)
            CHECK(types::set_default_format(src_layer_md_, tnc));
        if (diff_src_layer_md_.format == any)
            CHECK(types::set_default_format(diff_src_layer_md_, tnc));
        if (diff_weights_layer_md_.format == any) {
            CHECK(types::set_default_format(diff_weights_layer_md_, ldigo));
            CHECK(rnn_utils::set_good_strides(diff_weights_layer_md_));
        }
        if (diff_weights_iter_md_.format == any) {
            CHECK(types::set_default_format(diff_weights_iter_md_, ldigo));
            CHECK(rnn_utils::set_good_strides(diff_weights_iter_md_));
        }
        if (dst_layer_md_.format == any)
            CHECK(types::set_default_format(dst_layer_md_, tnc));
        if (diff_dst_layer_md_.format == any)
            CHECK(types::set_default_format(diff_dst_layer_md_, tnc));

        // Optional parameters
        if (with_src_iter() && src_iter_md_.format == any)
            CHECK(types::set_default_format(src_iter_md_, ldsnc));
        if (with_src_iter() && diff_src_iter_md_.format == any)
            CHECK(types::set_default_format(diff_src_iter_md_, ldsnc));
        if (with_bias() && bias_md_.format == any)
            CHECK(types::set_default_format(bias_md_, ldgo));
        if (with_bias() && diff_bias_md_.format == any)
            CHECK(types::set_default_format(diff_bias_md_, ldgo));
        if (with_dst_iter() && dst_iter_md_.format == any)
            CHECK(types::set_default_format(dst_iter_md_, ldsnc));
        if (with_dst_iter() && diff_dst_iter_md_.format == any)
            CHECK(types::set_default_format(diff_dst_iter_md_, ldsnc));

        return status::success;
    }

    status_t check_layout_consistency() {
        using namespace memory_format;
        using namespace types;
        bool ok = true;
        ok = ok && src_layer_md_.format == tnc
                && dst_layer_md_.format == tnc;
        ok = ok && IMPLICATION(!is_zero_md(&src_iter_md_),
                           src_iter_md_.format == ldsnc)
                && IMPLICATION(!is_zero_md(&dst_iter_md_),
                           dst_iter_md_.format == ldsnc);

        ok = ok && utils::one_of(weights_layer_md_.format, ldgoi, rnn_packed)
                && utils::one_of(weights_iter_md_.format, ldgoi, rnn_packed);
        ok = ok && IMPLICATION(weights_iter_md_.format == rnn_packed,
                           weights_iter_md_.layout_desc.rnn_packed_desc.format
                                   == mkldnn_ldgoi_p);
        ok = ok && IMPLICATION(weights_layer_md_.format == rnn_packed,
                           weights_layer_md_.layout_desc.rnn_packed_desc.format
                                   == mkldnn_ldgoi_p);

        ok = ok && IMPLICATION(!is_zero_md(&bias_md_), bias_md_.format == ldgo);

        ok = ok && diff_src_layer_md_.format == tnc
                && diff_dst_layer_md_.format == tnc;
        ok = ok && IMPLICATION(!is_zero_md(&diff_src_iter_md_),
                           diff_src_iter_md_.format == ldsnc)
                && IMPLICATION(!is_zero_md(&diff_dst_iter_md_),
                           diff_dst_iter_md_.format == ldsnc);
        ok = ok && diff_weights_layer_md_.format == ldigo
                && diff_weights_iter_md_.format == ldigo;
        ok = ok && IMPLICATION(!is_zero_md(&diff_bias_md_),
                           diff_bias_md_.format == ldgo);

        return ok ? status::success : status::unimplemented;
    }
};
}
}
}

#endif
