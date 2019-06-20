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

/// @example gpu_getting_started.c
/// @copybrief gpu_getting_started_c
/// > Annotated version: @ref gpu_getting_started_c

/// @page gpu_getting_started_c Getting started on GPU
/// This C API example demonstrates programming for Intel(R) Processor
/// Graphics with Intel MKL-DNN.
///
/// > Example code: @ref gpu_getting_started.c
///
/// - How to create Intel MKL-DNN memory objects.
///   - How to get data from user's buffer into an Intel MKL-DNN
///     memory object.
///   - How tensor's logical dimensions and memory object formats relate.
/// - How to create Intel MKL-DNN primitives.
/// - How to execute the primitives.
///
/// @include gpu_getting_started.c
/// @page gpu_getting_started_c

#include <stdio.h>
#include <stdlib.h>

#include "mkldnn.h"

#define CHECK(f)                                                             \
    do {                                                                     \
        mkldnn_status_t s = f;                                               \
        if (s != mkldnn_success) {                                           \
            printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, \
                    s);                                                      \
            exit(2);                                                         \
        }                                                                    \
    } while (0)

size_t product(int n_dims, const mkldnn_dim_t dims[]) {
    size_t n_elems = 1;
    for (int d = 0; d < n_dims; ++d) {
        n_elems *= (size_t)dims[d];
    }
    return n_elems;
}

void fill(mkldnn_memory_t mem, int n_dims, const mkldnn_dim_t dims[]) {
    float *array;
    CHECK(mkldnn_memory_map_data(mem, (void **)&array));

    const size_t n_elems = product(n_dims, dims);
    for (size_t e = 0; e < n_elems; ++e) {
        array[e] = e % 7 ? 1.0f : -1.0f;
    }

    CHECK(mkldnn_memory_unmap_data(mem, array));
}

int find_negative(mkldnn_memory_t mem, int n_dims, const mkldnn_dim_t dims[]) {
    int negs = 0;

    float *array;
    CHECK(mkldnn_memory_map_data(mem, (void **)&array));

    const size_t n_elems = product(n_dims, dims);
    for (size_t e = 0; e < n_elems; ++e) {
        negs += array[e] < 0.0f;
    }

    CHECK(mkldnn_memory_unmap_data(mem, array));

    return negs;
}

int doit() {
    mkldnn_engine_t engine_cpu, engine_gpu;
    CHECK(mkldnn_engine_create(&engine_cpu, mkldnn_cpu, 0));
    CHECK(mkldnn_engine_create(&engine_gpu, mkldnn_gpu, 0));

    mkldnn_dim_t tz[4] = { 2, 16, 1, 1 };

    mkldnn_memory_desc_t m_cpu_md, m_gpu_md;
    CHECK(mkldnn_memory_desc_init_by_tag(
            &m_cpu_md, 4, tz, mkldnn_f32, mkldnn_nchw));
    CHECK(mkldnn_memory_desc_init_by_tag(
            &m_gpu_md, 4, tz, mkldnn_f32, mkldnn_nchw));

    mkldnn_memory_t m_cpu, m_gpu;
    CHECK(mkldnn_memory_create(
            &m_cpu, &m_cpu_md, engine_cpu, MKLDNN_MEMORY_ALLOCATE));
    CHECK(mkldnn_memory_create(
            &m_gpu, &m_gpu_md, engine_gpu, MKLDNN_MEMORY_ALLOCATE));

    fill(m_cpu, 4, tz);
    if (find_negative(m_cpu, 4, tz) == 0) {
        printf("Please fix filling of data\n");
        exit(2);
    }

    /* reorder cpu -> gpu */
    mkldnn_primitive_desc_t r1_pd;
    CHECK(mkldnn_reorder_primitive_desc_create(
            &r1_pd, &m_cpu_md, engine_cpu, &m_gpu_md, engine_gpu, NULL));
    mkldnn_primitive_t r1;
    CHECK(mkldnn_primitive_create(&r1, r1_pd));

    /* relu gpu */
    mkldnn_eltwise_desc_t relu_d;
    CHECK(mkldnn_eltwise_forward_desc_init(&relu_d, mkldnn_forward,
            mkldnn_eltwise_relu, &m_gpu_md, 0.0f, 0.0f));

    mkldnn_primitive_desc_t relu_pd;
    CHECK(mkldnn_primitive_desc_create(
            &relu_pd, &relu_d, NULL, engine_gpu, NULL));

    mkldnn_primitive_t relu;
    CHECK(mkldnn_primitive_create(&relu, relu_pd));

    /* reorder gpu -> cpu */
    mkldnn_primitive_desc_t r2_pd;
    CHECK(mkldnn_reorder_primitive_desc_create(
            &r2_pd, &m_gpu_md, engine_gpu, &m_cpu_md, engine_cpu, NULL));
    mkldnn_primitive_t r2;
    CHECK(mkldnn_primitive_create(&r2, r2_pd));

    mkldnn_stream_t stream_gpu;
    CHECK(mkldnn_stream_create(
            &stream_gpu, engine_gpu, mkldnn_stream_default_flags));

    mkldnn_exec_arg_t r1_args[]
            = { { MKLDNN_ARG_FROM, m_cpu }, { MKLDNN_ARG_TO, m_gpu } };
    CHECK(mkldnn_primitive_execute(r1, stream_gpu, 2, r1_args));

    mkldnn_exec_arg_t relu_args[]
            = { { MKLDNN_ARG_SRC, m_gpu }, { MKLDNN_ARG_DST, m_gpu } };
    CHECK(mkldnn_primitive_execute(relu, stream_gpu, 2, relu_args));

    mkldnn_exec_arg_t r2_args[]
            = { { MKLDNN_ARG_FROM, m_gpu }, { MKLDNN_ARG_TO, m_cpu } };
    CHECK(mkldnn_primitive_execute(r2, stream_gpu, 2, r2_args));

    CHECK(mkldnn_stream_wait(stream_gpu));

    if (find_negative(m_cpu, 4, tz) != 0)
        return 2;

    /* clean up */
    mkldnn_primitive_desc_destroy(relu_pd);
    mkldnn_primitive_desc_destroy(r1_pd);
    mkldnn_primitive_desc_destroy(r2_pd);

    mkldnn_primitive_destroy(relu);
    mkldnn_primitive_destroy(r1);
    mkldnn_primitive_destroy(r2);
    mkldnn_memory_destroy(m_cpu);
    mkldnn_memory_destroy(m_gpu);

    mkldnn_stream_destroy(stream_gpu);

    mkldnn_engine_destroy(engine_cpu);
    mkldnn_engine_destroy(engine_gpu);

    return 0;
}

int main() {
    int result = doit();
    if (result)
        printf("failed\n");
    else
        printf("passed\n");

    return result;
}
