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

int doit(int lazy) {
    /* AlexNet: c3
     * {2, 256, 13, 13} (x) {384, 256, 3, 3} -> {2, 384, 13, 13}
     * pad: {1, 1}
     * strides: {1, 1}
     */

    const uint32_t mb = 2;
    const uint32_t groups = 2;
    uint32_t c3_src_sizes[4] = {mb, 256, 13, 13};
    uint32_t c3_weights_sizes[] = {groups, 384/groups, 256/groups, 3, 3};
    uint32_t c3_bias_sizes[1] = {384};
    uint32_t strides[] = {1, 1};
    int32_t  padding[] = {0, 0}; // set proper values
    uint32_t c3_dst_sizes[4] = {mb, 384,
        (c3_src_sizes[2] + 2*padding[0] - c3_weights_sizes[3])/strides[0] + 1,
        (c3_src_sizes[3] + 2*padding[1] - c3_weights_sizes[4])/strides[1] + 1
    };

    real_t *src = (real_t*)calloc(product(c3_src_sizes, 4), sizeof(real_t));
    real_t *weights = (real_t*)calloc(product(c3_weights_sizes, 5), sizeof(real_t));
    real_t *bias = (real_t*)calloc(product(c3_bias_sizes, 1), sizeof(real_t));
    real_t *dst = (real_t*)calloc(product(c3_dst_sizes, 4), sizeof(real_t));

    for (uint32_t i = 0; i < c3_bias_sizes[0]; ++i) bias[i] = i;

    mkl_dnn_engine_t engine;
    CHECK(mkl_dnn_engine_create(&engine, lazy ? mkl_dnn_cpu_lazy : mkl_dnn_cpu, 0 /* idx */));

    /* first describe user data and create data descriptors for future
     * convolution w/ the specified format -- we do not want to do a reorder */
    mkl_dnn_tensor_desc_t c3_src_tz, c3_weights_tz, c3_bias_tz, c3_dst_tz;
    mkl_dnn_memory_desc_t c3_src_md, c3_weights_md, c3_bias_md, c3_dst_md;
    mkl_dnn_memory_primitive_desc_t c3_src_pd, c3_weights_pd, c3_bias_pd, c3_dst_pd;
    mkl_dnn_primitive_t c3_src, c3_weights, c3_bias, c3_dst;

    CHECK(mkl_dnn_tensor_desc_init(&c3_src_tz, 4, c3_src_sizes));
    CHECK(mkl_dnn_memory_desc_init(&c3_src_md, &c3_src_tz, mkl_dnn_f32, mkl_dnn_nchw));
    CHECK(mkl_dnn_memory_primitive_desc_init(&c3_src_pd, &c3_src_md, engine));
    CHECK(mkl_dnn_memory_create(&c3_src, &c3_src_pd, 0 ? NULL : src));

    if (groups == 1) {
        CHECK(mkl_dnn_tensor_desc_init(&c3_weights_tz, 4, c3_weights_sizes + 1));
        CHECK(mkl_dnn_memory_desc_init(&c3_weights_md, &c3_weights_tz, mkl_dnn_f32, mkl_dnn_oihw));
    } else {
        CHECK(mkl_dnn_tensor_desc_init(&c3_weights_tz, 5, c3_weights_sizes));
        CHECK(mkl_dnn_memory_desc_init(&c3_weights_md, &c3_weights_tz, mkl_dnn_f32, mkl_dnn_goihw));
    }
    CHECK(mkl_dnn_memory_primitive_desc_init(&c3_weights_pd, &c3_weights_md, engine));
    CHECK(mkl_dnn_memory_create(&c3_weights, &c3_weights_pd, weights));

    CHECK(mkl_dnn_tensor_desc_init(&c3_bias_tz, 1, c3_bias_sizes));
    CHECK(mkl_dnn_memory_desc_init(&c3_bias_md, &c3_bias_tz, mkl_dnn_f32, mkl_dnn_x));
    CHECK(mkl_dnn_memory_primitive_desc_init(&c3_bias_pd, &c3_bias_md, engine));
    CHECK(mkl_dnn_memory_create(&c3_bias, &c3_bias_pd, bias));

    CHECK(mkl_dnn_tensor_desc_init(&c3_dst_tz, 4, c3_dst_sizes));
    CHECK(mkl_dnn_memory_desc_init(&c3_dst_md, &c3_dst_tz, mkl_dnn_f32, mkl_dnn_nchw));
    CHECK(mkl_dnn_memory_primitive_desc_init(&c3_dst_pd, &c3_dst_md, engine));

    mkl_dnn_primitive_at_t c3_srcs[] = {
        mkl_dnn_primitive_at(c3_src, 0),
        mkl_dnn_primitive_at(c3_weights, 0),
        mkl_dnn_primitive_at(c3_bias, 0)
    };

    const_mkl_dnn_primitive_t c3_dsts[1];
	CHECK(mkl_dnn_memory_create(&c3_dst, &c3_dst_pd, dst));
	c3_dsts[0] = c3_dst;

    /* create a convolution */
    mkl_dnn_convolution_desc_t c3_desc;
    mkl_dnn_convolution_primitive_desc_t c3_pd;
    mkl_dnn_primitive_t c3;

    CHECK(mkl_dnn_convolution_desc_init(&c3_desc, mkl_dnn_forward, mkl_dnn_convolution_direct,
                &c3_src_md, &c3_weights_md, &c3_bias_md, &c3_dst_md,
                strides, padding, mkl_dnn_padding_zero));
    CHECK(mkl_dnn_convolution_primitive_desc_init(&c3_pd, &c3_desc, engine));
    CHECK(mkl_dnn_primitive_create(&c3, &c3_pd, c3_srcs, c3_dsts));

    assert(mkl_dnn_memory_primitive_desc_equal(&c3_pd.src_primitive_desc, &c3_src_pd));
    assert(mkl_dnn_memory_primitive_desc_equal(&c3_pd.weights_primitive_desc, &c3_weights_pd));
    assert(mkl_dnn_memory_primitive_desc_equal(&c3_pd.bias_primitive_desc, &c3_bias_pd));
    assert(mkl_dnn_memory_primitive_desc_equal(&c3_pd.dst_primitive_desc, &c3_dst_pd));

    /* let us build a net */
    mkl_dnn_stream_t stream;
    CHECK(mkl_dnn_stream_create(&stream));
    CHECK(mkl_dnn_stream_submit(stream, 1, &c3, NULL));
    CHECK(mkl_dnn_stream_wait(stream, 1, NULL));

    /* clean-up */
    CHECK(mkl_dnn_stream_destroy(stream));
    mkl_dnn_primitive_destroy(c3);
    mkl_dnn_primitive_destroy(c3_src);
    mkl_dnn_primitive_destroy(c3_weights);
    mkl_dnn_primitive_destroy(c3_bias);
    mkl_dnn_primitive_destroy(c3_dst);
    mkl_dnn_engine_destroy(engine);

    int rc = 0;
    const uint32_t N = c3_dst_sizes[0], C = c3_dst_sizes[1],
          H = c3_dst_sizes[2], W = c3_dst_sizes[3];
    for (uint32_t n = 0; n < N; ++n)
    for (uint32_t c = 0; c < C; ++c)
    for (uint32_t h = 0; h < H; ++h)
    for (uint32_t w = 0; w < W; ++w)
    {
        size_t off = ((n*C + c)*H + h)*W + w;
        if (dst[off] != bias[c]) rc = 1;
    }

    free(src);
    free(weights);
    free(bias);
    free(dst);

    return rc;
}

int main(int argc, char **argv) {
    int rc = doit(0);
    printf("eager: %s\n", rc ? "failed" : "passed");
    rc = doit(1);
    printf("lazy:  %s\n", rc ? "failed" : "passed");
    return rc;
}
