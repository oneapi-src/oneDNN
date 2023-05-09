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

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class conv_config_t;
class conv_params_t;

namespace model {

float get_score(const conv_config_t &cfg, const conv_params_t &params);

} // namespace model
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
