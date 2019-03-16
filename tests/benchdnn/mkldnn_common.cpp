/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <assert.h>
#include "mkldnn.h"

#include "mkldnn_common.hpp"

// Engine kind used to run MKL-DNN primitives for testing
mkldnn_engine_kind_t engine_tgt_kind = mkldnn_cpu;

// Engine used for reference benchdnn computations
mkldnn_engine_t engine_ref;

// Engine used to run MKL-DNN primitives for testing
mkldnn_engine_t engine_tgt;

// Stream for reference engine
mkldnn_stream_t stream_ref;

// Stream for target engine
mkldnn_stream_t stream_tgt;
