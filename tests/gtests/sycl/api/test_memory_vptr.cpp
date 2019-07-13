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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"

#include <CL/sycl.hpp>
#include <cstdint>

using namespace cl::sycl;

namespace mkldnn {

#if MKLDNN_SYCL_MEMORY_API == MKLDNN_SYCL_MEMORY_API_VPTR

TEST(sycl_memory_vptr_test, Service) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
            "GPU devices not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dims tz = { 4, 4, 4, 4 };
    size_t sz = size_t(tz[0]) * tz[1] * tz[2] * tz[3];

    void *ptr = sycl_malloc(sz);

    ASSERT_TRUE(is_sycl_vptr(ptr));
    ASSERT_TRUE(is_sycl_vptr(static_cast<uint8_t *>(ptr) + sz - 1));
    ASSERT_FALSE(is_sycl_vptr(static_cast<uint8_t *>(ptr) + sz));

    sycl_free(ptr);
}

TEST(sycl_memory_vptr_test, Constructor) {
    engine eng(engine::kind::cpu, 0);
    memory::dim n = 100;
    memory::desc mem_d({ n }, memory::data_type::f32, memory::format_tag::x);

    void *ptr = sycl_malloc(sizeof(float) * n);
    memory mem(mem_d, eng, ptr);

    ASSERT_EQ(ptr, mem.get_data_handle());

    {
        auto buf_u8 = get_sycl_buffer(ptr);
        auto range = cl::sycl::range<1>(buf_u8.get_size() / sizeof(float));
        auto buf = buf_u8.reinterpret<float>(range);
        auto acc = buf.get_access<access::mode::write>();
        for (int i = 0; i < n; i++) {
            acc[i] = float(i);
        }
    }

    {
        auto buf_u8 = get_sycl_buffer(mem.get_data_handle());
        auto range = cl::sycl::range<1>(buf_u8.get_size() / sizeof(float));
        auto buf = buf_u8.reinterpret<float>(range);
        auto acc = buf.get_access<access::mode::read>();
        for (int i = 0; i < n; i++) {
            ASSERT_EQ(acc[i], float(i));
        }
    }

    sycl_free(ptr);
}

#endif

} // namespace mkldnn
