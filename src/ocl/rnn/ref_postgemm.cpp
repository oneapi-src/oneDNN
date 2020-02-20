/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

namespace dnnl {
namespace impl {
namespace ocl {

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
elemwise_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::rnn_elemwise)) {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());
    auto nd_range = compute::nd_range_t({dic, batch});
    const compute::kernel_t &kernel = (aprop == prop_kind::forward)
            ? elemwise_fwd_kernel_
            : elemwise_bwd_kernel_;
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, dir);
    arg_list.set(1, lay);
    arg_list.set(2, iter);
    arg_list.set(3, workspace);
    arg_list.set(4, bias);
    arg_list.set(5, pd()->desc()->alpha);
    // for test mode
    arg_list.set(6,
            pd()->rnn_conf_.is_testmode ? tm_scales
                                        : memory_storage_t::empty_storage());
    arg_list.set(
            7, pd()->rnn_conf_.is_testmode ? pd()->rnn_conf_.tm_cscale : 0.0f);
    compute_stream->parallel_for(nd_range, kernel, arg_list);
}
template elemwise_sig(ref_rnn_fwd_u8s8_t::rnn_elemwise);
template elemwise_sig(ref_rnn_fwd_f16_t::rnn_elemwise);
template elemwise_sig(ref_rnn_fwd_f32_t::rnn_elemwise);
template elemwise_sig(ref_rnn_bwd_f32_t::rnn_elemwise);
template elemwise_sig(ref_rnn_fwd_bf16_t::rnn_elemwise);
template elemwise_sig(ref_rnn_bwd_bf16_t::rnn_elemwise);

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
elemwise_sig(
        (_ref_rnn_common_t<aprop, src_type, weights_type>::lstm_elemwise)) {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());
    auto nd_range = compute::nd_range_t({dic, batch});
    const compute::kernel_t &kernel = (aprop == prop_kind::forward)
            ? elemwise_fwd_kernel_
            : elemwise_bwd_kernel_;
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, dir);
    arg_list.set(1, lay);
    arg_list.set(2, iter);
    arg_list.set(3, workspace);
    arg_list.set(4, bias);
    arg_list.set(5, pd()->desc()->alpha);
    // for test mode
    arg_list.set(6,
            pd()->rnn_conf_.is_testmode ? tm_scales
                                        : memory_storage_t::empty_storage());
    arg_list.set(
            7, pd()->rnn_conf_.is_testmode ? pd()->rnn_conf_.tm_cscale : 0.0f);
    compute_stream->parallel_for(nd_range, kernel, arg_list);
}
template elemwise_sig(ref_rnn_fwd_f16_t::lstm_elemwise);
template elemwise_sig(ref_rnn_fwd_f32_t::lstm_elemwise);
template elemwise_sig(ref_rnn_bwd_f32_t::lstm_elemwise);
template elemwise_sig(ref_rnn_fwd_bf16_t::lstm_elemwise);
template elemwise_sig(ref_rnn_bwd_bf16_t::lstm_elemwise);

template <>
elemwise_sig(ref_rnn_fwd_u8s8_t::lstm_elemwise) {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());
    auto nd_range = compute::nd_range_t({dic, batch});
    const compute::kernel_t &kernel = elemwise_fwd_kernel_;

    float data_shift = pd()->attr()->rnn_data_qparams_.shift_;
    float data_scale = pd()->attr()->rnn_data_qparams_.scale_;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, dir);
    arg_list.set(1, lay);
    arg_list.set(2, iter);
    arg_list.set(3, workspace);
    arg_list.set(4, scales);
    arg_list.set(5, bias);
    arg_list.set(6, pd()->desc()->alpha);
    arg_list.set(7, data_shift);
    arg_list.set(8, data_scale);
    // for test mode
    arg_list.set(9,
            pd()->rnn_conf_.is_testmode ? tm_scales
                                        : memory_storage_t::empty_storage());
    arg_list.set(
            10, pd()->rnn_conf_.is_testmode ? pd()->rnn_conf_.tm_cscale : 0.0f);
    compute_stream->parallel_for(nd_range, kernel, arg_list);
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
elemwise_sig(
        (_ref_rnn_common_t<aprop, src_type, weights_type>::gru_lbr_elemwise)) {
    assert(!"unimplemented");
}
template elemwise_sig(ref_rnn_fwd_u8s8_t::gru_lbr_elemwise);
template elemwise_sig(ref_rnn_fwd_f16_t::gru_lbr_elemwise);
template elemwise_sig(ref_rnn_fwd_f32_t::gru_lbr_elemwise);
template elemwise_sig(ref_rnn_bwd_f32_t::gru_lbr_elemwise);
template elemwise_sig(ref_rnn_fwd_bf16_t::gru_lbr_elemwise);
template elemwise_sig(ref_rnn_bwd_bf16_t::gru_lbr_elemwise);

} // namespace ocl
} // namespace impl
} // namespace dnnl
