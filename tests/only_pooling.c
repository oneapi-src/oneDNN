#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>

#include "mkl_dnn.h"

typedef float real_t;

#define CHECK(f) do { \
    mkl_dnn_status_t s = f; \
    if (s != mkl_dnn_success) { \
        printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, s); \
        exit(2); \
    } \
} while(0)

static size_t product(uint32_t *arr, size_t size) {
    size_t prod = 1;
    for (size_t i = 0; i < size; ++i) prod *= arr[i];
    return prod;
}

void init_input(uint32_t dim[4], real_t *x)
{
    uint32_t N = dim[0], C = dim[1], H = dim[2], W = dim[3];
    for (uint32_t n = 0; n < N; n += 1)
    for (uint32_t c = 0; c < C; c += 1)
    for (uint32_t h = 2; h+2 <= H; h += 2)
    for (uint32_t w = 2; w+2 <= W; w += 2)
        x[w + W*h + c*W*H + n*W*H*C] = c*n;
}

int doit() {
    /* AlexNet: p1
     * {16, 96, 55, 55} -> {16, 96, 27, 27}
     * pad: {0, 0}
     * strides: {2, 2}
     * kernel: {3, 3}
     */

    uint32_t p1_input_sizes[4] = {16, 96, 55, 55};
    uint32_t p1_indices_sizes[4] = {16, 96, 27, 27};
    uint32_t p1_output_sizes[4] = {16, 96, 27, 27};
    uint32_t strides[] = {2, 2};
    uint32_t kernel[] = { 3, 3 };
    int32_t  padding[] = { 0, 0 }; // set proper values

    real_t *input = (real_t*)calloc(product(p1_input_sizes, 4), sizeof(real_t));
    real_t *indices = (real_t*)calloc(product(p1_indices_sizes, 4), sizeof(real_t));
    real_t *output = (real_t*)calloc(product(p1_output_sizes, 4), sizeof(real_t));

    mkl_dnn_engine_t engine;
    CHECK(mkl_dnn_engine_create(&engine, mkl_dnn_cpu, 0 /* idx */));

    /* first describe user data and create data descriptors for future
     * pooling w/ the specified format -- we do not want to do a reorder */
    mkl_dnn_tensor_desc_t p1_input_tz, p1_indices_tz, p1_output_tz;
    mkl_dnn_memory_desc_t p1_input_md, p1_indices_md, p1_output_md;
    mkl_dnn_memory_primitive_desc_t p1_input_pd, p1_indices_pd, p1_output_pd;
    mkl_dnn_primitive_t p1_input, p1_indices, p1_output;

    CHECK(mkl_dnn_tensor_desc_init(&p1_input_tz, 1, 1, 2, p1_input_sizes));
    CHECK(mkl_dnn_memory_desc_init(&p1_input_md, &p1_input_tz, mkl_dnn_f32, mkl_dnn_nchw));
    CHECK(mkl_dnn_memory_primitive_desc_init(&p1_input_pd, &p1_input_md, engine));
    CHECK(mkl_dnn_memory_create(&p1_input, &p1_input_pd, 0 ? NULL : input));

    CHECK(mkl_dnn_tensor_desc_init(&p1_indices_tz, 1, 1, 2, p1_indices_sizes));
    CHECK(mkl_dnn_memory_desc_init(&p1_indices_md, &p1_indices_tz, mkl_dnn_f32, mkl_dnn_nchw));
    CHECK(mkl_dnn_memory_primitive_desc_init(&p1_indices_pd, &p1_indices_md, engine));
    CHECK(mkl_dnn_memory_create(&p1_indices, &p1_indices_pd, indices));

    CHECK(mkl_dnn_tensor_desc_init(&p1_output_tz, 1, 1, 2, p1_output_sizes));
    CHECK(mkl_dnn_memory_desc_init(&p1_output_md, &p1_output_tz, mkl_dnn_f32, mkl_dnn_nchw));
    CHECK(mkl_dnn_memory_primitive_desc_init(&p1_output_pd, &p1_output_md, engine));
    CHECK(mkl_dnn_memory_create(&p1_output, &p1_output_pd, output));

    mkl_dnn_primitive_at_t p1_inputs[] = {
        mkl_dnn_primitive_at(p1_input, 0),
        mkl_dnn_primitive_at(p1_indices, 0)
    };

    mkl_dnn_primitive_t p1_outputs[] = { p1_output };

    /* create a pooling */
    mkl_dnn_pooling_desc_t p1_desc;
    mkl_dnn_pooling_primitive_desc_t p1_pd;
    mkl_dnn_primitive_t p1;

    CHECK(mkl_dnn_pooling_desc_init(&p1_desc, mkl_dnn_forward, mkl_dnn_pooling_max,
                &p1_input_md, &p1_indices_md, &p1_output_md,
                strides, kernel, padding, mkl_dnn_padding_zero));
    CHECK(mkl_dnn_pooling_primitive_desc_init(&p1_pd, &p1_desc, engine));
    CHECK(mkl_dnn_primitive_create(&p1, &p1_pd, p1_inputs, p1_outputs));

    assert(mkl_dnn_memory_primitive_desc_equal(&p1_pd.input_primitive_desc, &p1_input_pd));
    assert(mkl_dnn_memory_primitive_desc_equal(&p1_pd.indices_primitive_desc, &p1_indices_pd));
    assert(mkl_dnn_memory_primitive_desc_equal(&p1_pd.output_primitive_desc, &p1_output_pd));

    init_input(p1_input_sizes, input);

    /* let us build a net */
    mkl_dnn_stream_t stream;
    CHECK(mkl_dnn_stream_create(&stream));
    CHECK(mkl_dnn_stream_submit(stream, 1, &p1, NULL));
    CHECK(mkl_dnn_stream_wait(stream, 1, NULL));

    /* clean-up */
    CHECK(mkl_dnn_stream_destroy(stream));
    mkl_dnn_primitive_destroy(p1);
    mkl_dnn_primitive_destroy(p1_input);
    mkl_dnn_primitive_destroy(p1_indices);
    mkl_dnn_primitive_destroy(p1_output);
    mkl_dnn_engine_destroy(engine);

    int n_errors = 0;
    const uint32_t N = p1_output_sizes[0], C = p1_output_sizes[1],
          H = p1_output_sizes[2], W = p1_output_sizes[3];
    for (uint32_t n = 0; n < N; ++n)
    for (uint32_t c = 0; c < C; ++c)
    for (uint32_t h = 0; h < H; ++h)
    for (uint32_t w = 0; w < W; ++w)
    {
        if (output[w + W*h + c*W*H + n*W*H*C] != c*n) n_errors += 1;
    }

    free(input);
    free(indices);
    free(output);

    return n_errors;
}

int main(int argc, char **argv) {
    int rc = doit();
    printf("%s\n", rc ? "failed" : "passed");
    return rc;
}
