/*******************************************************************************
 * Copyright 2021 Intel Corporation
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
#ifndef BACKEND_DNNL_F32_KERNEL_RESOURCE_HPP
#define BACKEND_DNNL_F32_KERNEL_RESOURCE_HPP

#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>

#include "utils/utils.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/thread_local_cache.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

class f32_kernel_resource_t {
public:
    class desc_t {
    public:
        // for bn folding
        memory::desc upd_wei_;
        memory::desc upd_bias_;

        // for preprocessed in/outputs
        memory::desc cvt_src_;
        memory::desc cvt_wei_;
        memory::desc cvt_bias_;
        memory::desc cvt_dst_;
        memory::desc cvt_post_src_;

        memory::desc cvt_src1_;

        // for optimal layout in/outputs
        memory::desc opt_src_;
        memory::desc opt_wei_;
        memory::desc opt_bias_;
        memory::desc opt_dst_;

        memory::desc opt_src1_;

        // for scratchpad
        memory::desc scratchpad_;

        memory::desc workspace_;
    };

    f32_kernel_resource_t(const desc_t &desc, const dnnl::engine &eng) {
        upd_wei_ = make_dnnl_memory(desc.upd_wei_, eng, nullptr);
        upd_bias_ = make_dnnl_memory(desc.upd_bias_, eng, nullptr);

        cvt_src_ = make_dnnl_memory(desc.cvt_src_, eng, nullptr);
        cvt_wei_ = make_dnnl_memory(desc.cvt_wei_, eng, nullptr);
        cvt_bias_ = make_dnnl_memory(desc.cvt_bias_, eng, nullptr);
        cvt_dst_ = make_dnnl_memory(desc.cvt_dst_, eng, nullptr);
        cvt_post_src_ = make_dnnl_memory(desc.cvt_post_src_, eng, nullptr);
        cvt_src1_ = make_dnnl_memory(desc.cvt_src1_, eng, nullptr);

        opt_src_ = make_dnnl_memory(desc.opt_src_, eng, nullptr);
        opt_wei_ = make_dnnl_memory(desc.opt_wei_, eng, nullptr);
        opt_bias_ = make_dnnl_memory(desc.opt_bias_, eng, nullptr);
        opt_dst_ = make_dnnl_memory(desc.opt_dst_, eng, nullptr);
        opt_src1_ = make_dnnl_memory(desc.opt_src1_, eng, nullptr);

        scratchpad_ = make_dnnl_memory(desc.scratchpad_, eng, nullptr);
        workspace_ = make_dnnl_memory(desc.workspace_, eng, nullptr);

        exec_args_ = {
                {DNNL_ARG_SRC, opt_src_},
                {DNNL_ARG_SRC_1, opt_src1_},
                {DNNL_ARG_WEIGHTS, opt_wei_},
                {DNNL_ARG_BIAS, opt_bias_},
                {DNNL_ARG_DST, opt_dst_},
                {DNNL_ARG_SCRATCHPAD, scratchpad_},
                {(DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1),
                        cvt_post_src_},
                {DNNL_ARG_WORKSPACE, workspace_},
        };

        exec_args_no_bias_ = {
                {DNNL_ARG_SRC, opt_src_},
                {DNNL_ARG_SRC_1, opt_src1_},
                {DNNL_ARG_WEIGHTS, opt_wei_},
                {DNNL_ARG_DST, opt_dst_},
                {DNNL_ARG_SCRATCHPAD, scratchpad_},
                {(DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1),
                        cvt_post_src_},
                {DNNL_ARG_WORKSPACE, workspace_},
        };
    }

    // for bn folding
    memory upd_wei_;
    memory upd_bias_;

    // for preprocessed in/outputs
    memory cvt_src_;
    memory cvt_wei_;
    memory cvt_bias_;
    memory cvt_dst_;
    memory cvt_post_src_;

    memory cvt_src1_;

    // for optimal layout in/outputs
    memory opt_src_;
    memory opt_wei_;
    memory opt_bias_;
    memory opt_dst_;

    memory opt_src1_;

    // for scratchpad
    memory scratchpad_;
    memory workspace_;

    // wrapped alias
    exec_args exec_args_;
    exec_args exec_args_no_bias_;
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
