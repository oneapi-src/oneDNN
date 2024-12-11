/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_KERNELS_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_KERNELS_HPP

#include "graph/backend/dnnl/kernels/decomp_kernel/mqa_decomp.hpp"
#include "graph/backend/dnnl/kernels/decomp_kernel/sdp_decomp.hpp"
#include "graph/backend/dnnl/kernels/gen_index.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/batch_norm.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/binary.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/concat.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/conv.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/conv_transpose.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/dummy.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/eltwise.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/group_norm.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/large_partition.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/layer_norm.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/log_softmax.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/matmul.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/pool.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/prelu.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/quantize.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/reduction.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/reorder.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/resampling.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/select.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/shuffle.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/softmax.hpp"
#include "graph/backend/dnnl/kernels/primitive_base/sum.hpp"
#include "graph/backend/dnnl/kernels/ukernel/sdp_primitive.hpp"

#endif
