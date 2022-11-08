/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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
#include <utility>

#include "interface/partition.hpp"
#include "interface/partition_hashing.hpp"
#include "interface/thread.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace partition_hashing {

key_t::key_t(size_t partition_id, engine_kind_t engine_kind,
        const std::vector<std::shared_ptr<op_t>> &ops,
        const std::vector<const logical_tensor_t *> &ins,
        const std::vector<const logical_tensor_t *> &outs)
    : partition_id_(partition_id)
    , ops_(get_raw_ptrs(ops))
    , nthread_(dnnl_graph_get_max_threads())
    , engine_kind_(engine_kind)
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

key_t::key_t(const partition_t *partition,
        const std::vector<const logical_tensor_t *> &ins,
        const std::vector<const logical_tensor_t *> &outs, const context_t *ctx)
    : key_t(partition->id(), partition->get_engine_kind(), partition->get_ops(),
            ins, outs) {
    using ltw = logical_tensor_wrapper_t;
    const auto &input_index_having_context
            = partition->get_input_index_having_context();
    const bool may_hash_context = !input_index_having_context.empty() && ctx
            && !ctx->get_ids().empty();
    if (may_hash_context) {
        for (const auto idx : input_index_having_context) {
            assertm(idx < ins.size(),
                    "compiled partition cache key: idx in "
                    "input_index_having_context exceeds the size of ins.");
            if (ctx->get_ids().count(ins[idx]->id) > 0) {
                size_t tid = ins[idx]->id;
                size_t dims_prod = impl::utils::prod(ltw(ins[idx]).vdims());
                if (ltw(ins[idx]).ndims() == 0) {
                    // a scalar
                    dims_prod = 1;
                }
                impl::data_type_t lt_type = ltw(ins[idx]).data_type();
                size_t total_size = dims_prod * impl::utils::size_of(lt_type);
                PARTITION_HASHING_SWITCH_TYPE(lt_type, dtype, {
                    std::vector<dtype> m;
                    m.reserve(dims_prod);
                    assertm(ctx->get_tensor_data_handle(tid),
                            "tensor data handle cannot be nullptr.");
                    std::memcpy(m.data(), ctx->get_tensor_data_handle(tid),
                            total_size);
                    context_content_pairs_.emplace_back(
                            std::pair<size_t, impl::utils::any_t> {tid, m});
                });
            }
        }
    }
}

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
            && engine_kind_ == rhs.engine_kind_;
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

    if (context_content_pairs_.size() != rhs.context_content_pairs_.size())
        return false;

    for (auto &&pair : context_content_pairs_) {
        auto rhs_matched = std::find_if(rhs.context_content_pairs_.begin(),
                rhs.context_content_pairs_.end(),
                [&pair](const std::pair<size_t, impl::utils::any_t> &p) {
                    return p.first == pair.first;
                });
        if (rhs_matched == rhs.context_content_pairs_.end()) return false;

        auto lhs_matched = std::find_if(ins_.begin(), ins_.end(),
                [&pair](const impl::logical_tensor_t &lt) {
                    return lt.id == pair.first;
                });
        PARTITION_HASHING_SWITCH_TYPE(
                logical_tensor_wrapper_t(*lhs_matched).data_type(), dtype, {
                    const auto &a
                            = impl::utils::any_cast<const std::vector<dtype> &>(
                                    pair.second);
                    const auto &b
                            = impl::utils::any_cast<const std::vector<dtype> &>(
                                    *rhs_matched);
                    if (a != b) return false;
                });
    }

    return true;
}

size_t get_op_hash(const op_t &op) {
    size_t seed = 0;
    // TODO(zixuanwe): `id` might be sufficient to cover all attributes.
    seed = utils::hash_combine(seed, op.get_id());
    seed = utils::hash_combine(seed, static_cast<size_t>(op.get_kind()));
    return seed;
}

} // namespace partition_hashing
} // namespace impl
} // namespace graph
} // namespace dnnl
