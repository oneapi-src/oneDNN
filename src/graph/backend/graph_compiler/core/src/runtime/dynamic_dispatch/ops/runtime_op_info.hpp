/*******************************************************************************
 * Copyright 2023-2024 Intel Corporation
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
#include <stdint.h>

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_OPS_RUNTIME_OP_INFO_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_OPS_RUNTIME_OP_INFO_HPP

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

struct dyn_conv_fwd_runtime_info_t {
    int stride_d = 1;
    int stride_h = 1;
    int stride_w = 1;

    int pads_begin_d = 0;
    int pads_begin_h = 0;
    int pads_begin_w = 0;

    int pads_end_d = 0;
    int pads_end_h = 0;
    int pads_end_w = 0;

    int dilation_d = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    dyn_conv_fwd_runtime_info_t() = default;
    dyn_conv_fwd_runtime_info_t(int stride_d, int stride_h, int stride_w,
            int pads_begin_d, int pads_begin_h, int pads_begin_w,
            int pads_end_d, int pads_end_h, int pads_end_w, int dilation_d,
            int dilation_h, int dilation_w)
        : stride_d(stride_d)
        , stride_h(stride_h)
        , stride_w(stride_w)
        , pads_begin_d(pads_begin_d)
        , pads_begin_h(pads_begin_h)
        , pads_begin_w(pads_begin_w)
        , pads_end_d(pads_end_d)
        , pads_end_h(pads_end_h)
        , pads_end_w(pads_end_w)
        , dilation_d(dilation_d)
        , dilation_h(dilation_h)
        , dilation_w(dilation_w) {}

    dyn_conv_fwd_runtime_info_t(int stride_h, int stride_w, int pads_begin_h,
            int pads_begin_w, int pads_end_h, int pads_end_w, int dilation_h,
            int dilation_w)
        : stride_h(stride_h)
        , stride_w(stride_w)
        , pads_begin_h(pads_begin_h)
        , pads_begin_w(pads_begin_w)
        , pads_end_h(pads_end_h)
        , pads_end_w(pads_end_w)
        , dilation_h(dilation_h)
        , dilation_w(dilation_w) {}
};

struct dyn_padding_runtime_info_t {
    int pads_begin_d = 0;
    int pads_begin_h = 0;
    int pads_begin_w = 0;

    int pads_end_d = 0;
    int pads_end_h = 0;
    int pads_end_w = 0;

    dyn_padding_runtime_info_t() = default;

    dyn_padding_runtime_info_t(int pads_begin_d, int pads_begin_h,
            int pads_begin_w, int pads_end_d, int pads_end_h, int pads_end_w)
        : pads_begin_d(pads_begin_d)
        , pads_begin_h(pads_begin_h)
        , pads_begin_w(pads_begin_w)
        , pads_end_d(pads_end_d)
        , pads_end_h(pads_end_h)
        , pads_end_w(pads_end_w) {}

    dyn_padding_runtime_info_t(
            int pads_begin_h, int pads_begin_w, int pads_end_h, int pads_end_w)
        : pads_begin_h(pads_begin_h)
        , pads_begin_w(pads_begin_w)
        , pads_end_h(pads_end_h)
        , pads_end_w(pads_end_w) {}
};

struct dyn_pooling_runtime_info_t {
    int stride_d = 1;
    int stride_h = 1;
    int stride_w = 1;

    int pads_begin_d = 0;
    int pads_begin_h = 0;
    int pads_begin_w = 0;

    int pads_end_d = 0;
    int pads_end_h = 0;
    int pads_end_w = 0;

    int kernel_d = 0;
    int kernel_h = 0;
    int kernel_w = 0;

    bool rounding_type_floor = true;
    bool auto_pads_same = false;

    dyn_pooling_runtime_info_t() = default;

    dyn_pooling_runtime_info_t(int stride_d, int stride_h, int stride_w,
            int pads_begin_d, int pads_begin_h, int pads_begin_w,
            int pads_end_d, int pads_end_h, int pads_end_w, int kernel_d,
            int kernel_h, int kernel_w, bool rounding_type_floor,
            bool auto_pads_same)
        : stride_d(stride_d)
        , stride_h(stride_h)
        , stride_w(stride_w)
        , pads_begin_d(pads_begin_d)
        , pads_begin_h(pads_begin_h)
        , pads_begin_w(pads_begin_w)
        , pads_end_d(pads_end_d)
        , pads_end_h(pads_end_h)
        , pads_end_w(pads_end_w)
        , kernel_d(kernel_d)
        , kernel_h(kernel_h)
        , kernel_w(kernel_w)
        , rounding_type_floor(rounding_type_floor)
        , auto_pads_same(auto_pads_same) {}

    dyn_pooling_runtime_info_t(int stride_h, int stride_w, int pads_begin_h,
            int pads_begin_w, int pads_end_h, int pads_end_w, int kernel_h,
            int kernel_w, bool rounding_type_floor, bool auto_pads_same)
        : stride_h(stride_h)
        , stride_w(stride_w)
        , pads_begin_h(pads_begin_h)
        , pads_begin_w(pads_begin_w)
        , pads_end_h(pads_end_h)
        , pads_end_w(pads_end_w)
        , kernel_h(kernel_h)
        , kernel_w(kernel_w)
        , rounding_type_floor(rounding_type_floor)
        , auto_pads_same(auto_pads_same) {}
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
