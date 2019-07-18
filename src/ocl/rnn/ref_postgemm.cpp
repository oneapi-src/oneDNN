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

#include "ocl/rnn/ref_rnn.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
elemwise_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::rnn_elemwise)) {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());
    auto nd_range = compute::nd_range_t({ batch, dic });
    const compute::kernel_t &kernel = (aprop == prop_kind::forward)
            ? elemwise_fwd_kernel_
            : elemwise_bwd_kernel_;
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, dir);
    arg_list.set(1, lay);
    arg_list.set(2, iter);
    arg_list.set(3, workspace);
    arg_list.set(4, bias);
    compute_stream->parallel_for(nd_range, kernel, arg_list);
}
template elemwise_sig(ref_rnn_fwd_f16_t::rnn_elemwise);
template elemwise_sig(ref_rnn_fwd_f32_t::rnn_elemwise);
template elemwise_sig(ref_rnn_bwd_f32_t::rnn_elemwise);


template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
elemwise_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::lstm_elemwise)) {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());
    auto nd_range = compute::nd_range_t({ batch, dic });
    const compute::kernel_t &kernel = (aprop == prop_kind::forward)
            ? elemwise_fwd_kernel_
            : elemwise_bwd_kernel_;
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, dir);
    arg_list.set(1, lay);
    arg_list.set(2, iter);
    arg_list.set(3, workspace);
    arg_list.set(4, bias);
    compute_stream->parallel_for(nd_range, kernel, arg_list);
}
template elemwise_sig(ref_rnn_fwd_f16_t::lstm_elemwise);
template elemwise_sig(ref_rnn_fwd_f32_t::lstm_elemwise);
template elemwise_sig(ref_rnn_bwd_f32_t::lstm_elemwise);

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
elemwise_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::gru_lbr_elemwise))
{
    assert(!"unimplemented");
}
template elemwise_sig(ref_rnn_fwd_f16_t::gru_lbr_elemwise);
template elemwise_sig(ref_rnn_fwd_f32_t::gru_lbr_elemwise);
template elemwise_sig(ref_rnn_bwd_f32_t::gru_lbr_elemwise);

}
}
}
