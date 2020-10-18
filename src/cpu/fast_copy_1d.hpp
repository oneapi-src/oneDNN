/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#pragma once

#include <assert.h>
#include "cpu/platform.hpp"

#if DNNL_X64
#include "cpu/x64/jit_copy_1d.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {

template <typename T>
void fast_copy_1d(
        T *dst, const T *src, int start_off, int src_width, int nelem) {
    DNNL_X64_ONLY(return x64::spmd::copy_1d<T>(
            dst, src, start_off, src_width, nelem));
    for (int i = 0; i < nelem; ++i) {
        auto _off = i + start_off;
        dst[i] = (_off >= 0 && _off < src_width) ? src[_off] : (T)0;
    }
}

template <typename T>
void fast_copy_1d_batch_4(T *dst, int d_stride, const T *src, int s_stride,
        int start_off, int src_width, int nelem) {
    DNNL_X64_ONLY(return x64::spmd::copy_1d_batch_4<T>(
            dst, d_stride, src, s_stride, start_off, src_width, nelem));
    for (int i = 0; i < nelem; ++i) {
        auto p_off = i + start_off;
        for (int batch = 0; batch < 4; ++batch) {
            (dst + batch * d_stride)[i] = (p_off >= 0 && p_off < src_width)
                    ? (src + batch * s_stride)[p_off]
                    : (T)0;
        }
    }
}

template <typename T>
void fast_copy_1d_batch_8(T *dst, int d_stride, const T *src, int s_stride,
        int start_off, int src_width, int nelem) {
    DNNL_X64_ONLY(return x64::spmd::copy_1d_batch_8<T>(
            dst, d_stride, src, s_stride, start_off, src_width, nelem));
    for (int i = 0; i < nelem; ++i) {
        auto p_off = i + start_off;
        for (int batch = 0; batch < 8; ++batch) {
            (dst + batch * d_stride)[i] = (p_off >= 0 && p_off < src_width)
                    ? (src + batch * s_stride)[p_off]
                    : (T)0;
        }
    }
}

template <typename T>
void fast_zero_1d(T *dst, int nelem) {
    DNNL_X64_ONLY(return x64::spmd::zero_1d<T>(dst, nelem));
    for (int pid = 0; pid < nelem; ++pid) {
        dst[pid] = (T)0;
    }
}

template <typename T>
void fast_zero_1d_batch_4(T *dst, int stride, int nelem) {
    DNNL_X64_ONLY(return x64::spmd::zero_1d_batch_4<T>(dst, stride, nelem));
    for (int pid = 0; pid < nelem; ++pid) {
        for (int batch = 0; batch < 4; ++batch) {
            (dst + batch * stride)[pid] = (T)0;
        }
    }
}

template <typename T>
void fast_zero_1d_batch_8(T *dst, int stride, int nelem) {
    DNNL_X64_ONLY(return x64::spmd::zero_1d_batch_8<T>(dst, stride, nelem));
    for (int pid = 0; pid < nelem; ++pid) {
        for (int batch = 0; batch < 8; ++batch) {
            (dst + batch * stride)[pid] = (T)0;
        }
    }
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
