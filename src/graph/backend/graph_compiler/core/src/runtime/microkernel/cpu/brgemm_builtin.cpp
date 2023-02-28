/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#include <runtime/data_type.hpp>
#include <runtime/microkernel/cpu/brgemm_common.hpp>
#include <runtime/microkernel/cpu/microkernel.hpp>
#include <util/assert.hpp>

using namespace dnnl::impl::graph::gc;
typedef sc_data_etype sc_dtype;

extern "C" {

SC_API void *sc_brgemm_get_amx_scratch(
        const char *palette, bool *need_config_amx, runtime::stream_t *stream) {
    bool amx_exclusive = false;
    *need_config_amx = false;
    return do_get_amx_tile_buf(
            palette, stream, amx_exclusive, *need_config_amx);
}
}
