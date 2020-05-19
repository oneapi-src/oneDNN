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

#include "primitive_desc.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "pooling_pd.hpp"
#include "shuffle_pd.hpp"

#include "engine.hpp"
#include "primitive_hashing.hpp"

namespace dnnl {
namespace impl {
namespace primitive_hashing {

key_t::key_t(const primitive_desc_t *pd, const engine_t *engine, int impl_nthr)
    : primitive_kind_(pd->kind())
    , op_desc_(primitive_kind_, pd->op_desc())
    , attr_(*pd->attr())
    , impl_id_(pd->impl_id())
    , impl_nthr_(impl_nthr)
    , kind_(engine ? engine->kind() : engine_kind::any_engine)
    , runtime_kind_(engine ? engine->runtime_kind() : runtime_kind::none)
    , device_id_(engine ? engine->device_id() : 0) {
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
        case primitive_kind::resampling: {
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
            && mds.size() == rhs.mds.size() && attr_ == rhs.attr_
            && kind_ == rhs.kind_ && runtime_kind_ == rhs.runtime_kind_
            && device_id_ == rhs.device_id_ && op_desc_ == rhs.op_desc_;

    if (!ret) return false;

    for (size_t i = 0; i < mds.size(); ++i)
        if (mds[i] != rhs.mds[i]) return false;

    return true;
}

} // namespace primitive_hashing
} // namespace impl
} // namespace dnnl
