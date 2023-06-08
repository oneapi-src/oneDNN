/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include <memory>

#include "graph/interface/partition.hpp"
#include "graph/interface/partition_hashing.hpp"

#include "common/dnnl_thread.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace partition_hashing {

key_t::key_t(size_t partition_id, const impl::engine_t *engine,
        const std::vector<std::shared_ptr<op_t>> &ops,
        const std::vector<const logical_tensor_t *> &ins,
        const std::vector<const logical_tensor_t *> &outs)
    : partition_id_(partition_id)
    , ops_(get_raw_ptrs(ops))
    , nthread_(dnnl_get_max_threads())
    , engine_id_(engine->engine_id())
    , thread_id_(std::this_thread::get_id()) {
    ins_.reserve(ins.size());
    outs_.reserve(outs.size());
    for (auto &in : ins) {
        ins_.emplace_back(*in);
    }
    for (auto &out : outs) {
        outs_.emplace_back(*out);
    }
}

key_t::key_t(const partition_t *partition, const impl::engine_t *engine,
        const std::vector<const logical_tensor_t *> &ins,
        const std::vector<const logical_tensor_t *> &outs)
    : key_t(partition->id(), engine, partition->get_ops(), ins, outs) {}

bool key_t::operator==(const key_t &rhs) const {
    if (this == &rhs) return true;

    const size_t lhs_num_ops = ops_.size();
    const size_t rhs_num_ops = rhs.ops_.size();
    const size_t lhs_num_ins = ins_.size();
    const size_t rhs_num_ins = rhs.ins_.size();
    const size_t lhs_num_outs = outs_.size();
    const size_t rhs_num_outs = rhs.outs_.size();

    bool ret = true && lhs_num_ops == rhs_num_ops && lhs_num_ins == rhs_num_ins
            && lhs_num_outs == rhs_num_outs
            && partition_id_ == rhs.partition_id_ && nthread_ == rhs.nthread_
            && engine_id_ == rhs.engine_id_;
    if (!ret) return false;

    for (size_t i = 0; i < lhs_num_ops; ++i) {
        const op_t *op = ops_[i];
        if (std::find_if(rhs.ops_.begin(), rhs.ops_.end(),
                    [op](const op_t *rhs_op) { return *op == *rhs_op; })
                == rhs.ops_.end())
            return false;
    }

    for (size_t i = 0; i < lhs_num_ins; ++i) {
        const logical_tensor_wrapper_t lhs_lt {ins_[i]};
        if (std::find_if(rhs.ins_.begin(), rhs.ins_.end(),
                    [&lhs_lt](const logical_tensor_t &rhs_lt) {
                        return logical_tensor_wrapper_t(rhs_lt) == lhs_lt;
                    })
                == rhs.ins_.end())
            return false;
    }

    for (size_t i = 0; i < lhs_num_outs; ++i) {
        const logical_tensor_wrapper_t lhs_lt {outs_[i]};
        if (std::find_if(rhs.outs_.begin(), rhs.outs_.end(),
                    [&lhs_lt](const logical_tensor_t &rhs_lt) {
                        return logical_tensor_wrapper_t(rhs_lt) == lhs_lt;
                    })
                == rhs.outs_.end())
            return false;
    }

    return true;
}

size_t get_op_hash(const op_t &op) {
    size_t seed = 0;
    // TODO(zixuanwe): `id` might be sufficient to cover all attributes.
    seed = hash_combine(seed, op.get_id());
    seed = hash_combine(seed, static_cast<size_t>(op.get_kind()));
    return seed;
}

} // namespace partition_hashing
} // namespace graph
} // namespace impl
} // namespace dnnl
