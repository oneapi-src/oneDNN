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

#ifndef GPU_JIT_CONV_TILER_HPP
#define GPU_JIT_CONV_TILER_HPP

#include <memory>

#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class conv_config_t;
class conv_tuner_t;
class conv_tiler_impl_t;

class conv_tiler_t {
public:
    conv_tiler_t(const conv_config_t &cfg);
    void set_tuner(conv_tuner_t *tuner);
    int configs() const;
    bool is_tuning_mode() const;
    bool can_move_next() const;
    void set_params(conv_config_t &cfg);
    void notify_out_of_registers(const conv_config_t &cfg);
    bool is_grf_limit_ok(const conv_config_t &cfg) const;
    static void after_create_hook(const conv_config_t &cfg);
    static void before_exec_hook(const conv_config_t &cfg, stream_t *stream);

private:
    std::shared_ptr<conv_tiler_impl_t> impl_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
