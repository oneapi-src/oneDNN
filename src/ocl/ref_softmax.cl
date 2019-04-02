/*******************************************************************************
* Copyright 2019 intel corporation
*
* licensed under the apache license, version 2.0 (the "license");
* you may not use this file except in compliance with the license.
* you may obtain a copy of the license at
*
*     http://www.apache.org/licenses/license-2.0
*
* unless required by applicable law or agreed to in writing, software
* distributed under the license is distributed on an "as is" basis,
* without warranties or conditions of any kind, either express or implied.
* see the license for the specific language governing permissions and
* limitations under the license.
*******************************************************************************/

#define CONCAt2(a, b) a##b
#define CONCAT2(a, b) CONCAt2(a, b)

#define OFF(dim, idx)                                    \
    (dim % CONCAT2(DATA_B, idx)) * CONCAT2(DATA_SB, idx) \
            + (dim / CONCAT2(DATA_B, idx)) * CONCAT2(DATA_S, idx)

#if SOFTMAX_AXIS_IDX == 0
#    define DATA_OFF(dim0, dim1, dim2, softmax_dim) \
        OFF(softmax_dim, 0) + OFF(dim0, 1) + OFF(dim1, 2) + OFF(dim2, 3)
#elif SOFTMAX_AXIS_IDX == 1
#    define DATA_OFF(dim0, dim1, dim2, softmax_dim) \
        OFF(dim0, 0) + OFF(softmax_dim, 1) + OFF(dim1, 2) + OFF(dim2, 3)
#elif SOFTMAX_AXIS_IDX == 2
#    define DATA_OFF(dim0, dim1, dim2, softmax_dim) \
        OFF(dim0, 0) + OFF(dim1, 1) + OFF(softmax_dim, 2) + OFF(dim2, 3)
#elif SOFTMAX_AXIS_IDX == 3
#    define DATA_OFF(dim0, dim1, dim2, softmax_dim) \
        OFF(dim0, 0) + OFF(dim1, 1) + OFF(dim2, 2) + OFF(softmax_dim, 3)
#else
#    error unsupported softmax dimension
#endif

__kernel void ref_softmax_fwd_generic(
        __global float *src, __global float *dst) {
    const int dim[] = { get_global_id(0), get_global_id(1), get_global_id(2) };

    float temp_data[SOFTMAX_AXIS];

    float max = temp_data[0] = src[DATA_OFF(dim[0], dim[1], dim[2], 0)];
    for (int i = 1; i < SOFTMAX_AXIS; ++i) {
        size_t data_off = DATA_OFF(dim[0], dim[1], dim[2], i);
        temp_data[i] = src[data_off];
        max = temp_data[i] > max ? temp_data[i] : max;
    }

    float denom = 0.0f;
    for (int i = 0; i < SOFTMAX_AXIS; ++i) {
        denom += temp_data[i] = exp(temp_data[i] - max);
    }

    for (int i = 0; i < SOFTMAX_AXIS; ++i) {
        size_t data_off = DATA_OFF(dim[0], dim[1], dim[2], i);
        dst[data_off] = temp_data[i] / denom;
    }
}
