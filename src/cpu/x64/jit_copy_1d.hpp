/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace spmd {

template <typename T>
void copy_1d(T *dst, const T *src, int start_off, int src_width, int nelem);
template <typename T>
void copy_1d_batch_4(T *dst, int d_stride, const T *src, int s_stride,
        int start_off, int src_width, int nelem);
template <typename T>
void copy_1d_batch_8(T *dst, int d_stride, const T *src, int s_stride,
        int start_off, int src_width, int nelem);

template <typename T>
void zero_1d(T *dst, int nelem);
template <typename T>
void zero_1d_batch_4(T *dst, int stride, int nelem);
template <typename T>
void zero_1d_batch_8(T *dst, int stride, int nelem);

} // namespace spmd
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
