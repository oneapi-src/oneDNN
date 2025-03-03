/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#include "oneapi/dnnl/dnnl.h"

#include "common/dnnl_thread.hpp"
#include "common/engine.hpp"
#include "common/primitive_desc.hpp"
#include "common/primitive_serialization.hpp"
#include "common/serialization.hpp"

namespace dnnl {
namespace impl {

const std::vector<uint8_t> &cache_blob_id_t::get(
        const engine_t *engine, const primitive_desc_t *pd) {
    if (is_initialized_) return sstream_.get_data();

    auto engine_kind = engine->kind();
    auto runtime_kind = engine->runtime_kind();

    if (engine_kind != engine_kind::gpu
            || (engine_kind == engine_kind::gpu
                    && runtime_kind != runtime_kind::ocl)) {
        return sstream_.get_data();
    }

    if (pd->kind() == primitive_kind::zero_pad) { return sstream_.get_data(); }

    assert(engine->kind() == engine_kind::gpu
            && engine->runtime_kind() == runtime_kind::ocl);

    const auto init_id = [&]() {
        serialize_desc(sstream_, pd->op_desc());
        serialize(sstream_, *pd->attr());

        const int nthr = engine->kind() == engine_kind::gpu
                ? 0
                : dnnl_get_max_threads();
        sstream_.append(nthr);

        for (const auto &md : pd->hint_mds(false /* is_hint */)) {
            serialize(sstream_, md);
        }

        sstream_.append(engine_kind);
        // TODO: blob object can probably be re-used for different runtimes
        // if the engine kind is the same. Check this assumption when extending
        // this API to DPCPP runtime.
        sstream_.append(runtime_kind);

        engine->serialize_device(sstream_);

        auto pd_iterator_offset = pd->pd_iterator_offset();
        sstream_.append(pd_iterator_offset);
        auto pd_skip_idx = pd->skip_idx();
        sstream_.append(pd_skip_idx);

        auto version = dnnl_version();
        sstream_.append(version->major);
        sstream_.append(version->minor);
        sstream_.append(version->patch);

        sstream_.append_array(std::strlen(version->hash), version->hash);

        is_initialized_ = true;
    };

    std::call_once(flag_, init_id);
    return sstream_.get_data();
}

} // namespace impl
} // namespace dnnl
