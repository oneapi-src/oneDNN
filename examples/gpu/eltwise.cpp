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

#include <iostream>

#include "mkldnn.hpp"

using namespace mkldnn;

using namespace std;

size_t product(const memory::dims adims) {
    size_t n_elems = 1;
    for (size_t d = 0; d < adims.size(); ++d) {
        n_elems *= (size_t)adims[d];
    }
    return n_elems;
}

void fill(const memory &mem, const memory::dims adims) {
    float *array = mem.map_data<float>();

    for (size_t e = 0; e < adims.size(); ++e) {
        array[e] = e % 7 ? 1.0f : -1.0f;
    }

    mem.unmap_data(array);
}

int find_negative(const memory &mem, const memory::dims adims) {
    int negs = 0;

    float *array = mem.map_data<float>();

    for (size_t e = 0; e < adims.size(); ++e) {
        negs += array[e] < 0.0f;
    }

    mem.unmap_data(array);
    return negs;
}

void doit() {
    auto cpu_engine = engine(engine::cpu, 0);
    auto gpu_engine = engine(engine::gpu, 0);

    const auto tz = memory::dims{ 2, 16, 1, 1 };

    auto m_cpu = memory(
            { { tz }, memory::data_type::f32, memory::format_tag::nchw },
            cpu_engine);
    auto m_gpu = memory(
            { { tz }, memory::data_type::f32, memory::format_tag::nchw },
            gpu_engine);

    fill(m_cpu, tz);

    /* reorder cpu -> gpu */
    auto r1 = reorder(m_cpu, m_gpu);

    /* relu gpu */
    auto relu_d = eltwise_forward::desc(prop_kind::forward,
            algorithm::eltwise_relu, m_gpu.get_desc(), 0.0f);
    auto relu_pd = eltwise_forward::primitive_desc(relu_d, gpu_engine);
    auto relu = eltwise_forward(relu_pd);

    /* reorder gpu -> cpu */
    auto r2 = reorder(m_gpu, m_cpu);

    auto stream_gpu = stream(gpu_engine);
    r1.execute(stream_gpu, m_cpu, m_gpu);
    relu.execute(stream_gpu,
            { { MKLDNN_ARG_SRC, m_gpu }, { MKLDNN_ARG_DST, m_gpu } });
    r2.execute(stream_gpu, m_gpu, m_cpu);

    stream_gpu.wait();

    if (find_negative(m_cpu, tz) != 0)
        printf("failed\n");
    else
        printf("passed\n");
}

int main(int argc, char **argv) {
    try {
        doit();
    } catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
        return 1;
    }
    return 0;
}
