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

int doit() {
    /* AlexNet: c3
     * {2, 256, 13, 13} (x) {384, 256, 3, 3} -> {2, 384, 13, 13}
     * pad: {1, 1}
     * strides: {1, 1}
     */

    const uint32_t mb = 2;
    uint32_t c3_input_sizes[4] = {mb, 256, 13, 13};
    uint32_t c3_weights_sizes[4] = {384, 256, 3, 3};
    uint32_t c3_bias_sizes[1] = {384};
    uint32_t padding[] = {0, 0}; // set proper values
    uint32_t strides[] = {1, 1};
    uint32_t c3_output_sizes[4] = {mb, 384,
        (c3_input_sizes[2] + 2*padding[0] - c3_weights_sizes[2])/strides[0] + 1,
        (c3_input_sizes[3] + 2*padding[1] - c3_weights_sizes[3])/strides[1] + 1
    };

    real_t *input = (real_t*)malloc(sizeof(real_t)*product(c3_input_sizes, 4));
    real_t *weights = (real_t*)malloc(sizeof(real_t)*product(c3_weights_sizes, 4));
    real_t *bias = (real_t*)malloc(sizeof(real_t)*product(c3_bias_sizes, 1));
    real_t *output = (real_t*)malloc(sizeof(real_t)*product(c3_output_sizes, 4));

    mkl_dnn_engine_t engine;
    CHECK(mkl_dnn_engine_create(&engine, mkl_dnn_cpu, 0 /* idx */));

    /* first describe user data and create data descriptors for future
     * convolution w/ the specified format -- we do not want to do a reorder */
    mkl_dnn_tensor_desc_t c3_input_tz, c3_weights_tz, c3_bias_tz, c3_output_tz;
    mkl_dnn_memory_desc_t c3_input_md, c3_weights_md, c3_bias_md, c3_output_md;
    mkl_dnn_memory_primitive_desc_t c3_input_pd, c3_weights_pd, c3_bias_pd, c3_output_pd;
    mkl_dnn_primitive_t c3_input, c3_weights, c3_bias, c3_output;

    CHECK(mkl_dnn_tensor_desc_init(&c3_input_tz, 1, 1, 2, c3_input_sizes));
    CHECK(mkl_dnn_memory_desc_init(&c3_input_md, &c3_input_tz, mkl_dnn_nchw_f32));
    CHECK(mkl_dnn_memory_primitive_desc_init(&c3_input_pd, &c3_input_md, engine));
    CHECK(mkl_dnn_memory_create(&c3_input, &c3_input_pd, NULL /*input*/));

    CHECK(mkl_dnn_tensor_desc_init(&c3_weights_tz, 0, 2, 2, c3_weights_sizes));
    CHECK(mkl_dnn_memory_desc_init(&c3_weights_md, &c3_weights_tz, mkl_dnn_oihw_f32));
    CHECK(mkl_dnn_memory_primitive_desc_init(&c3_weights_pd, &c3_weights_md, engine));
    CHECK(mkl_dnn_memory_create(&c3_weights, &c3_weights_pd, weights));

    CHECK(mkl_dnn_tensor_desc_init(&c3_bias_tz, 0, 0, 1, c3_bias_sizes));
    CHECK(mkl_dnn_memory_desc_init(&c3_bias_md, &c3_bias_tz, mkl_dnn_n_f32));
    CHECK(mkl_dnn_memory_primitive_desc_init(&c3_bias_pd, &c3_bias_md, engine));
    CHECK(mkl_dnn_memory_create(&c3_bias, &c3_bias_pd, bias));

    CHECK(mkl_dnn_tensor_desc_init(&c3_output_tz, 1, 1, 2, c3_output_sizes));
    CHECK(mkl_dnn_memory_desc_init(&c3_output_md, &c3_output_tz, mkl_dnn_nchw_f32));
    CHECK(mkl_dnn_memory_primitive_desc_init(&c3_output_pd, &c3_output_md, engine));

    mkl_dnn_primitive_at_t c3_inputs[] = {
        mkl_dnn_primitive_at(c3_input, 0),
        mkl_dnn_primitive_at(c3_weights, 0),
        mkl_dnn_primitive_at(c3_bias, 0)
    };

    mkl_dnn_primitive_t c3_outputs[1];
	CHECK(mkl_dnn_memory_create(&c3_output, &c3_output_pd, output));
	c3_outputs[0] = c3_output;

    /* create a convolution */
    mkl_dnn_convolution_desc_t c3_desc;
    mkl_dnn_convolution_primitive_desc_t c3_pd;
    mkl_dnn_primitive_t c3;

    CHECK(mkl_dnn_convolution_desc_init(&c3_desc, mkl_dnn_forward, mkl_dnn_convolution_direct,
                &c3_input_md, &c3_weights_md, &c3_bias_md, &c3_output_md,
                strides, padding, mkl_dnn_padding_zero));
    CHECK(mkl_dnn_convolution_primitive_desc_init(&c3_pd, &c3_desc, engine));
    CHECK(mkl_dnn_primitive_create(&c3, &c3_pd, c3_inputs, c3_outputs));

    assert(mkl_dnn_memory_primitive_desc_equal(&c3_pd.input_primitive_desc, &c3_input_pd));
    assert(mkl_dnn_memory_primitive_desc_equal(&c3_pd.weights_primitive_desc, &c3_weights_pd));
    assert(mkl_dnn_memory_primitive_desc_equal(&c3_pd.bias_primitive_desc, &c3_bias_pd));
    assert(mkl_dnn_memory_primitive_desc_equal(&c3_pd.output_primitive_desc, &c3_output_pd));

    /* let us build a net */
    mkl_dnn_stream_t stream;
    CHECK(mkl_dnn_stream_create(&stream));
    CHECK(mkl_dnn_stream_submit(stream, 1, &c3, NULL));
    CHECK(mkl_dnn_stream_wait(stream, 1, NULL));

    /* clean-up */
    CHECK(mkl_dnn_stream_destroy(stream));
    mkl_dnn_primitive_destroy(c3);
    mkl_dnn_primitive_destroy(c3_input);
    mkl_dnn_primitive_destroy(c3_weights);
    mkl_dnn_primitive_destroy(c3_bias);
    mkl_dnn_primitive_destroy(c3_output);
    mkl_dnn_engine_destroy(engine);

    free(input);
    free(weights);
    free(bias);
    free(output);

    return 0;
}

int main(int argc, char **argv) {
    int rc = doit();
    printf("%s\n", rc ? "failed" : "passed");
    return rc;
}
