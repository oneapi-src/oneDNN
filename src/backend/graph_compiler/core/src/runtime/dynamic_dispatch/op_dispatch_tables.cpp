/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#include "op_dispatch_tables.hpp"
#include <assert.h>
#include <string.h>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {

void op_dispatch_tables_t::set_format_table_keys(uint64_t *keys,
        uint64_t num_keys, uint64_t *values, uint64_t num_values) {
    assert(format_table_);
    if (void *v = format_table_->get(keys, num_keys)) {
        memcpy(v, values, num_values * sizeof(uint64_t));
    } else {
        std::unique_ptr<uint64_t[]> new_values(new uint64_t[num_values]);
        memcpy(new_values.get(), values, num_values * sizeof(uint64_t));
        format_table_->set(keys, num_keys, new_values.get());
        format_values_.emplace_back(std::move(new_values));
    }
}

void op_dispatch_tables_t::set_impl_kind_table_keys(
        uint64_t *keys, uint64_t num_keys, int value) {
    assert(impl_kind_table_);
    if (void *v = impl_kind_table_->get(keys, num_keys)) {
        memcpy(v, &value, sizeof(int));
    } else {
        impl_kind_values_.emplace_back(new int);
        *impl_kind_values_.back() = value;
        impl_kind_table_->set(keys, num_keys, impl_kind_values_.back());
    }
}

op_dispatch_tables_t::~op_dispatch_tables_t() {
    kernel_dispatch_func_ = nullptr;
    for (auto &addr : impl_kind_values_) {
        delete addr;
    }
}
} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
