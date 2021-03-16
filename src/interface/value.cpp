/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "value.hpp"
#include "op.hpp"

using namespace dnnl::graph::impl;

bool value_t::find_consumer(const op_kind_t &kind, size_t &offset) {
    for (size_t i = 0; i < consumers_.size(); i++) {
        const impl::op_t &op1 = consumers_[i].get_op();
        if (op1.get_kind() == kind) {
            offset = i;
            return true;
        }
    }
    return false;
}
