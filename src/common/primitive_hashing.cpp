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

#include "primitive_desc.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "pooling_pd.hpp"
#include "shuffle_pd.hpp"

#include "primitive_hashing.hpp"

namespace dnnl {
namespace impl {
namespace primitive_hashing {

key_t::key_t(const primitive_desc_t *pd, int impl_nthr)
    : primitive_kind_(pd->kind())
    , op_desc_(pd->op_desc())
    , attr_(pd->attr())
    , impl_id_(pd->impl_id())
    , impl_nthr_(impl_nthr) {
    init_mds(pd);
}

void key_t::init_mds(const primitive_desc_t *pd) {
    // Put only **relevant** memory descriptors to the list that might
    // affect the equality. The current cases are:
    // - Backward pooling and shuffle (rationale: implementation might depend
    //   on the fwd_hint_pd).
    //
    // Later this list can be extended. For instance, currently we don't store
    // convolution mds, because nthrs + op_desc (even with format=`any`) +
    // attributes fully define particular implementation.
    //
    // XXX: There is too much knowledge about in the internals...

    switch (primitive_kind_) {
        case primitive_kind::batch_normalization: {
            break;
        }
        case primitive_kind::binary: {
            break;
        }
        case primitive_kind::concat: {
            break;
        }
        case primitive_kind::convolution: {
            break;
        }
        case primitive_kind::deconvolution: {
            break;
        }
        case primitive_kind::eltwise: {
            break;
        }
        case primitive_kind::gemm: {
            break;
        }
        case primitive_kind::inner_product: {
            break;
        }
        case primitive_kind::layer_normalization: {
            break;
        }
        case primitive_kind::logsoftmax: {
            break;
        }
        case primitive_kind::lrn: {
            break;
        }
        case primitive_kind::matmul: {
            break;
        }
        case primitive_kind::pooling: {
            auto typed_pd = utils::downcast<const pooling_pd_t *>(pd);
            if (!typed_pd->is_fwd()) {
                mds.push_back(*typed_pd->diff_dst_md(0));
                mds.push_back(*typed_pd->diff_src_md(0));
            }
            break;
        }
        case primitive_kind::reorder: {
            break;
        }
        case primitive_kind::rnn: {
            break;
        }
        case primitive_kind::shuffle: {
            auto typed_pd = utils::downcast<const shuffle_pd_t *>(pd);
            if (!typed_pd->is_fwd()) {
                mds.push_back(*typed_pd->diff_dst_md(0));
                mds.push_back(*typed_pd->diff_src_md(0));
            }
            break;
        }
        case primitive_kind::softmax: {
            break;
        }
        case primitive_kind::sum: {
            break;
        }
        default: assert(!"unknown primitive_kind");
    }
}

bool key_t::operator==(const key_t &rhs) const {
    DNNL_SHORT_CIRCUIT_SELF_COMPARISON(rhs);

    bool ret = true && primitive_kind_ == rhs.primitive_kind_
            && impl_id_ == rhs.impl_id_ && impl_nthr_ == rhs.impl_nthr_
            && mds.size() == rhs.mds.size() && *attr_ == *rhs.attr_;

    if (!ret) return false;

    switch (primitive_kind_) {
        // NOTE: make sure that op_descs for all primitives are compared below
        case primitive_kind::batch_normalization:
            ret = cast_and_compare<batch_normalization_desc_t>(
                    op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::binary:
            ret = cast_and_compare<binary_desc_t>(op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::concat:
            ret = cast_and_compare<concat_desc_t>(op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::convolution:
            ret = cast_and_compare<convolution_desc_t>(op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::deconvolution:
            ret = cast_and_compare<deconvolution_desc_t>(
                    op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::eltwise:
            ret = cast_and_compare<eltwise_desc_t>(op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::gemm:
            ret = cast_and_compare<gemm_desc_t>(op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::inner_product:
            ret = cast_and_compare<inner_product_desc_t>(
                    op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::layer_normalization:
            ret = cast_and_compare<layer_normalization_desc_t>(
                    op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::logsoftmax:
            ret = cast_and_compare<logsoftmax_desc_t>(op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::lrn:
            ret = cast_and_compare<lrn_desc_t>(op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::matmul:
            ret = cast_and_compare<matmul_desc_t>(op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::pooling:
            ret = cast_and_compare<pooling_desc_t>(op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::reorder:
            ret = cast_and_compare<reorder_desc_t>(op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::rnn:
            ret = cast_and_compare<rnn_desc_t>(op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::shuffle:
            ret = cast_and_compare<shuffle_desc_t>(op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::softmax:
            ret = cast_and_compare<softmax_desc_t>(op_desc_, rhs.op_desc_);
            break;
        case primitive_kind::sum:
            ret = cast_and_compare<sum_desc_t>(op_desc_, rhs.op_desc_);
            break;
        default: assert(!"unknown primitive_kind");
    }

    if (!ret) return false;

    for (size_t i = 0; i < mds.size(); ++i)
        if (mds[i] != rhs.mds[i]) return false;

    return true;
}

} // namespace primitive_hashing
} // namespace impl
} // namespace dnnl
