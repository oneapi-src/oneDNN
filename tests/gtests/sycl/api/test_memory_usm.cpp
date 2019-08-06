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

#include <cstdint>
#include <CL/sycl.hpp>

using namespace cl::sycl;

namespace mkldnn {

#ifdef MKLDNN_SYCL_INTEL

// TODO: enable after a USM bug related to mixed CPU/GPU execution is fixed.
#if 0
TEST(sycl_memory_usm_test, Constructor) {
    engine eng(engine::kind::cpu, 0);
    memory::dim n = 100;
    memory::desc mem_d({ n }, memory::data_type::f32, memory::format_tag::x);

    void *ptr = cl::sycl::malloc_shared(
            sizeof(float) * n, eng.get_sycl_device(), eng.get_sycl_context());
    memory mem(mem_d, eng, ptr);

    ASSERT_EQ(ptr, mem.get_data_handle());

    {
        for (int i = 0; i < n; i++) {
            ((float *)ptr)[i] = float(i);
        }
    }

    {
        float *ptr_f32 = (float *)mem.get_data_handle();
        for (int i = 0; i < n; i++) {
            ASSERT_EQ(ptr_f32[i], float(i));
        }
    }

    cl::sycl::free(ptr, eng.get_sycl_context());
}
#endif

#endif

} // namespace mkldnn
