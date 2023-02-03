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

#include "graph/interface/value.hpp"
#include "graph/interface/op.hpp"

using namespace dnnl::impl::graph;

utils::optional_t<size_t> value_t::find_consumer(const size_t start_index,
        const op_kind_t kind, const size_t expected_input_offset,
        bool ignore_expected_input_offset) {
    if (start_index >= consumers_.size()) return utils::nullopt;
    for (size_t i = start_index; i < consumers_.size(); i++) {
        const op_t &op1 = consumers_[i].get_op();
        const size_t input_offset = consumers_[i].get_offset();
        if ((op1.get_kind() == kind)
                && (ignore_expected_input_offset
                        || input_offset == expected_input_offset)) {
            return i;
        }
    }
    return utils::nullopt;
}
