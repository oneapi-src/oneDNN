/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "dnnl.hpp"
#include <CL/sycl.hpp>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc.hpp"
#include "common/primitive_exec_types.hpp"
#include "sycl/sycl_stream.hpp"

#ifdef DNNL_SYCL_DPCPP

namespace dnnl {

cl::sycl::event primitive::execute_sycl(const stream &astream,
        const std::unordered_map<int, memory> &aargs,
        const std::vector<cl::sycl::event> &deps) const {
    auto *sycl_stream
            = impl::utils::downcast<impl::sycl::sycl_stream_t *>(astream.get());
    sycl_stream->set_deps(deps);

    // run primitive
    auto *primitive_iface = impl::utils::downcast<primitive_iface_t *>(get());

    std::vector<dnnl_exec_arg_t> c_args;
    c_args.reserve(aargs.size());
    for (const auto &a : aargs)
        c_args.push_back({a.first, a.second.get()});

    impl::exec_args_t args;
    error::wrap_c_api(
            impl::cvt_primtive_args(primitive_iface->pd()->impl().get(),
                    (int)aargs.size(), c_args.data(), args),
            "could not execute a primitive");

    impl::exec_ctx_t ctx(sycl_stream, std::move(args));
    error::wrap_c_api(dnnl::impl::primitive_execute(primitive_iface, ctx),
            "could not execute a primitive");

    // return output event
    return sycl_stream->get_output_event();
}

} // namespace dnnl
#endif
