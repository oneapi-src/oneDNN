#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>

#include <vector>

#include "mkl_dnn.hpp"

typedef float real_t;

struct conv_params {
    uint32_t mb, ic, oc;
    uint32_t ih, iw, kh, kw;
    uint32_t ksh, ksw;
    uint32_t padh, padw;
    mkl_dnn::memory::format ifmt, wfmt, bfmt, ofmt;
    conv_params(std::vector<uint32_t> p) {
        assert(p.count() == 11);
        mb = p[0]; ic = p[1]; oc = p[2];
        ih = p[3]; iw = p[4]; kh = p[5]; kw = p[6];
        ksh = p[7]; ksw = p[8]; padh = p[9]; padw = p[10];
        ifmt = ofmt = mkl_dnn::memory::format::nchw_f32;
        wfmt = mkl_dnn::memory::format::oihw_f32;
        bfmt = mkl_dnn::memory::format::nhw_f32;
    }
    uint32_t oh() { return (ih + 2*padh - kh)/ksh + 1; }
    uint32_t ow() { return (iw + 2*padw - kw)/ksw + 1; }
};

int doit(const conv_params& c) {
    using namespace mkl_dnn;

    /* AlexNet: c3
     * {256, 256, 13, 13} (x) {384, 256, 3, 3} -> {256, 384, 13, 13}
     * pad: {1, 1}
     * strides: {1, 1}
     */

    printf("There are %zu CPU engines\n", engine::get_count(engine::cpu));
    auto cpu_engine = engine(engine::cpu, 0);

    // TODO: make tensor desc optional and default to N C X1 .. XN

    auto c3_input_tensor_desc = memory::tensor_desc(1, 1, 2, {256, 256, 13, 13});
    auto c3_input_memory_desc = memory::desc({1, 1, 2, {256, 256, 13, 13}}, memory::format::nchw_f32);
    auto c3_input_primitive_desc = memory::primitive_desc({{1, 1, 2, {256, 256, 13, 13}}, memory::format::nchw_f32}, cpu_engine);

    auto c3_input = memory({{{1, 1, 2, {256, 256, 13, 13}}, memory::format::nchw_f32}, cpu_engine});
    auto c3_weights = memory({{{0, 2, 2, {384, 256, 3, 3}}, memory::format::oihw_f32}, cpu_engine});
    auto c3_bias = memory({{{0, 0, 1, {384}}, memory::format::n_f32}, cpu_engine});
    auto c3_output = memory({{{1, 1, 2, {256, 384, 13, 13}}, memory::format::nchw_f32}, cpu_engine});

    // auto c3 = convolution::create({propagation::forward, algorithm::direct, {0, 0, 1, 1}, {0, 0, 1, 1}, padding::zero}, {input, weights, bias}, {output});

    // stream::create().submit({c3}).wait();

    return 0;
}

#define RUN(doit) do { \
    int rc = doit; \
    if (rc) { \
        printf("[%s:%d] FAILED: %s\n", __FILE__, __LINE__, #doit); \
        exit(2); \
    } \
} while (0)

int main(int argc, char **argv) {
    RUN(doit({2/*mb*/, 256/*ic*/, 384/*oc*/, 13/*ih*/, 13/*iw*/,
                3/*kh*/, 3/*kw*/, 1/*ksh*/, 1/*ksw*/, 1/*padh*/, 1/*padw*/ }));

    printf("passed");
    return 0;
}
