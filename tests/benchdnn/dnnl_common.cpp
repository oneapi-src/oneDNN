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
#include "dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

// Engine kind used to run DNNL primitives for testing
dnnl_engine_kind_t engine_tgt_kind = dnnl_cpu;

// Engine used for reference benchdnn computations
dnnl_engine_t engine_ref;

// Engine used to run DNNL primitives for testing
dnnl_engine_t engine_tgt;

// Stream for reference engine
dnnl_stream_t stream_ref;

// Stream for target engine
dnnl_stream_t stream_tgt;

args_t &args_t::set(int arg, const dnn_mem_t &mem) {
    args_.push_back(std::make_pair(arg, &mem));
    return *this;
}

dnnl_status_t execute_and_wait(
        dnnl_primitive_t prim, dnnl_stream_t stream, const args_t &args) {

    std::vector<dnnl_exec_arg_t> dnnl_args(args.size());
    for (int i = 0; i < args.size(); ++i) {
        // Unmap before passing the memory to execute
        if (args.dnn_mem(i).is_mapped()) args.dnn_mem(i).unmap();

        dnnl_args[i].arg = args.arg(i);
        dnnl_args[i].memory = args.dnn_mem(i).m_;
    }

    dnnl_status_t status = dnnl_primitive_execute(
            prim, stream, dnnl_args.size(), dnnl_args.data());
    if (status != dnnl_success) return status;
    status = dnnl_stream_wait(stream);
    if (status != dnnl_success) return status;

    // Map the memory back
    for (int i = 0; i < args.size(); ++i)
        if (!args.dnn_mem(i).is_mapped()) args.dnn_mem(i).map();
    return dnnl_success;
}
