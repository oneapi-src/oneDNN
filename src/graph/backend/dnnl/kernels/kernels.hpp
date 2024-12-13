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

#include "graph/backend/dnnl/kernels/decomp/mqa_decomp.hpp"
#include "graph/backend/dnnl/kernels/decomp/sdp_decomp.hpp"
#include "graph/backend/dnnl/kernels/gen_index.hpp"
#include "graph/backend/dnnl/kernels/prim/batch_norm.hpp"
#include "graph/backend/dnnl/kernels/prim/binary.hpp"
#include "graph/backend/dnnl/kernels/prim/concat.hpp"
#include "graph/backend/dnnl/kernels/prim/conv.hpp"
#include "graph/backend/dnnl/kernels/prim/conv_transpose.hpp"
#include "graph/backend/dnnl/kernels/prim/dummy.hpp"
#include "graph/backend/dnnl/kernels/prim/eltwise.hpp"
#include "graph/backend/dnnl/kernels/prim/group_norm.hpp"
#include "graph/backend/dnnl/kernels/prim/large_partition.hpp"
#include "graph/backend/dnnl/kernels/prim/layer_norm.hpp"
#include "graph/backend/dnnl/kernels/prim/log_softmax.hpp"
#include "graph/backend/dnnl/kernels/prim/matmul.hpp"
#include "graph/backend/dnnl/kernels/prim/pool.hpp"
#include "graph/backend/dnnl/kernels/prim/prelu.hpp"
#include "graph/backend/dnnl/kernels/prim/quantize.hpp"
#include "graph/backend/dnnl/kernels/prim/reduction.hpp"
#include "graph/backend/dnnl/kernels/prim/reorder.hpp"
#include "graph/backend/dnnl/kernels/prim/resampling.hpp"
#include "graph/backend/dnnl/kernels/prim/select.hpp"
#include "graph/backend/dnnl/kernels/prim/shuffle.hpp"
#include "graph/backend/dnnl/kernels/prim/softmax.hpp"
#include "graph/backend/dnnl/kernels/prim/sum.hpp"
#include "graph/backend/dnnl/kernels/uker/sdp_primitive.hpp"

#endif
