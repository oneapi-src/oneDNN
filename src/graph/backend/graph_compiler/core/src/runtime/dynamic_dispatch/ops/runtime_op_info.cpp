/*******************************************************************************
 * Copyright 2023 Intel Corporation
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
#include "runtime_op_info.hpp"
#include <util/reflection.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// clang-format off
SC_CLASS(dyn_conv_fwd_runtime_info_t)
  SC_FIELD(stride_d)
  SC_FIELD(stride_h)
  SC_FIELD(stride_w)
  SC_FIELD(pads_begin_d)
  SC_FIELD(pads_begin_h)
  SC_FIELD(pads_begin_w)
  SC_FIELD(pads_end_d)
  SC_FIELD(pads_end_h)
  SC_FIELD(pads_end_w)
SC_CLASS_END();

SC_CLASS(dyn_padding_runtime_info_t)
  SC_FIELD(pads_begin_d)
  SC_FIELD(pads_begin_h)
  SC_FIELD(pads_begin_w)
  SC_FIELD(pads_end_d)
  SC_FIELD(pads_end_h)
  SC_FIELD(pads_end_w)
SC_CLASS_END();
// clang-format on

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
