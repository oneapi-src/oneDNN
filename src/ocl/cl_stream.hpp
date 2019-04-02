/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef CL_STREAM_HPP
#define CL_STREAM_HPP

#include <memory>

#include "common/stream.hpp"
#include "ocl/cl_executor.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

// Abstract stream class providing access to cl_executor.
// Intended to use as a base class for OpenCL-like stream classes.
struct cl_stream_t : public stream_t {
    cl_stream_t(engine_t *engine, unsigned flags) : stream_t(engine, flags) {}
    virtual ~cl_stream_t() override = default;
    virtual cl_executor_t *cl_executor() const { return cl_executor_.get(); }

protected:
    void set_cl_executor(cl_executor_t *cl_executor) {
        cl_executor_.reset(cl_executor);
    }

private:
    std::unique_ptr<cl_executor_t> cl_executor_;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif
