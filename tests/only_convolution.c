#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>

#include "mkl_dnn.h"

typedef float real_t;

#define CHECK(f) do { \
    status_t s = f; \
    if (s != success) { \
        printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, s); \
        exit(2); \
    } \
} while(0)

int doit() {
    uint32_t enough = 128*1024*1024;
    real_t *input = (real_t*)malloc(sizeof(real_t)*enough);
    real_t *weights = (real_t*)malloc(sizeof(real_t)*enough);
    real_t *bias = (real_t*)malloc(sizeof(real_t)*1024);
    real_t *output = (real_t*)malloc(sizeof(real_t)*enough);

    /* AlexNet: c3
     * {256, 256, 13, 13} (x) {384, 256, 3, 3} -> {256, 384, 13, 13}
     * pad: {1, 1}
     * strides: {1, 1}
     */

    uint32_t c3_input_sizes[4] = {256, 256, 13, 13};
    uint32_t c3_weights_sizes[4] = {384, 256, 3, 3};
    uint32_t c3_bias_sizes[1] = {384};
    int32_t padding[] = {0, 0}; // set proper values
    uint32_t strides[] = {1, 1};
    uint32_t c3_output_sizes[4] = {256, 384,
        13 + 2*padding[1] - 2,
        13 + 2*padding[0] - 2};

    dnn_engine_t engine;
    CHECK(engine_create(&engine, engine_kind_cpu, 0 /* idx */));

    /* first describe user data and create data descriptors for future
     * convolution w/ the specified format -- we do not want to do a reorder */
    tensor_desc_t c3_input_tz, c3_weights_tz, c3_bias_tz, c3_output_tz;
    memory_desc_t c3_input_md, c3_weights_md, c3_bias_md, c3_output_md;
    memory_primitive_desc_t c3_input_pd, c3_weights_pd, c3_bias_pd, c3_output_pd;
    dnn_primitive_t c3_input, c3_weights, c3_bias, c3_output;

    CHECK(tensor_desc_init(&c3_input_tz, 1, 1, 2, c3_input_sizes));
    CHECK(memory_desc_init(&c3_input_md, &c3_input_tz, memory_format_nchw_f32));
    CHECK(memory_primitive_desc_init(&c3_input_pd, &c3_input_md, engine));
    CHECK(memory_create(&c3_input, &c3_input_pd, NULL/*input*/));

    CHECK(tensor_desc_init(&c3_weights_tz, 0, 2, 2, c3_weights_sizes));
    CHECK(memory_desc_init(&c3_weights_md, &c3_weights_tz, memory_format_oihw_f32));
    CHECK(memory_primitive_desc_init(&c3_weights_pd, &c3_weights_md, engine));
    CHECK(memory_create(&c3_weights, &c3_weights_pd, weights));

    CHECK(tensor_desc_init(&c3_bias_tz, 0, 0, 1, c3_bias_sizes));
    CHECK(memory_desc_init(&c3_bias_md, &c3_bias_tz, memory_format_n_f32));
    CHECK(memory_primitive_desc_init(&c3_bias_pd, &c3_bias_md, engine));
    CHECK(memory_create(&c3_bias, &c3_bias_pd, bias));

    CHECK(tensor_desc_init(&c3_output_tz, 1, 1, 2, c3_output_sizes));
    CHECK(memory_desc_init(&c3_output_md, &c3_output_tz, memory_format_nchw_f32));
    CHECK(memory_primitive_desc_init(&c3_output_pd, &c3_output_md, engine));

    dnn_primitive_at_t c3_inputs[] = {
        primitive_at(c3_input, 0),
        primitive_at(c3_weights, 0),
        primitive_at(c3_bias, 0)
    };
    const_dnn_primitive_t c3_outputs[1];

    const int conv_with_own_memory = 1;
    if (conv_with_own_memory) { /* primitive has its own memory */
        c3_outputs[0] = primitive_self;
        /* XXX: not finished yet: what to do with this `output` then?
         * TODO: add a reorder from c3 -> user outut */
    } else { /* stand-alone user-defined memory */
        CHECK(memory_create(&c3_output, &c3_output_pd, output));
        c3_outputs[0] = c3_output;
    }

    /* create a convolution */
    convolution_desc_t c3_desc;
    convolution_primitive_desc_t c3_pd;
    dnn_primitive_t c3;

    CHECK(convolution_desc_init(&c3_desc, forward, convolution_direct,
                &c3_input_md, &c3_weights_md, &c3_bias_md, &c3_output_md,
                strides, padding, padding_kind_zero));
    CHECK(convolution_primitive_desc_init(&c3_pd, &c3_desc, engine));
    CHECK(primitive_create(&c3, &c3_pd, c3_inputs, c3_outputs));

    assert(memory_primitive_desc_equal(&c3_pd.input_primitive_desc, &c3_input_pd));
    assert(memory_primitive_desc_equal(&c3_pd.weights_primitive_desc, &c3_weights_pd));
    assert(memory_primitive_desc_equal(&c3_pd.bias_primitive_desc, &c3_bias_pd));
    assert(memory_primitive_desc_equal(&c3_pd.output_primitive_desc, &c3_output_pd));

    /* let us build a net */
    dnn_stream_t stream;
    CHECK(stream_create(&stream));
    CHECK(stream_submit(stream, 1, &c3, NULL));
    CHECK(stream_wait(stream, 1, NULL));

    /* clean-up */
    CHECK(stream_destroy(stream));
    primitive_destroy(c3);
    primitive_destroy(c3_input);
    primitive_destroy(c3_weights);
    primitive_destroy(c3_bias);
    primitive_destroy(c3_output);
    engine_destroy(engine);

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
