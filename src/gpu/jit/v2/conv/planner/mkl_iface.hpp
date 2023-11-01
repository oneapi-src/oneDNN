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

#ifndef GPU_JIT_V2_CONV_PLANNER_MKL_IFACE_HPP
#define GPU_JIT_V2_CONV_PLANNER_MKL_IFACE_HPP

#include <dlfcn.h>
#include <stdexcept>

#include "common/cpp_compat.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {
namespace conv {
namespace planner {

template <typename FuncTypeT>
struct func_t {
    void load(void *handle, const char *name) {
        if (!handle) return;
        name_ = name;
        ptr_ = reinterpret_cast<FuncTypeT>(dlsym(handle, name));
    }

    template <typename... ArgsT>
    typename cpp_compat::invoke_result<FuncTypeT, ArgsT...>::type operator()(
            ArgsT... args) const {
        if (!ptr_) throw std::runtime_error("Cannot call function " + name_);
        return ptr_(args...);
    }

    std::string name_;
    FuncTypeT ptr_ = nullptr;
};

struct mkl_iface_t {
    using LAPACKE_sgelsd_func_type = int (*)(int, int, int, int, float *, int,
            float *, int, float *, float, int *);
    using mkl_set_threading_layer_func_type = int (*)(int);
    func_t<mkl_set_threading_layer_func_type> mkl_set_threading_layer;
    func_t<LAPACKE_sgelsd_func_type> LAPACKE_sgelsd;

    static mkl_iface_t &instance() {
        static mkl_iface_t _instance;
        return _instance;
    }

    mkl_iface_t() {
        const char *library_name = "libmkl_rt.so";
        lib_ = dlopen(library_name, RTLD_LAZY);
        if (!lib_) {
            printf("Error: cannot open library: %s\n", library_name);
            exit(1);
        }
        mkl_set_threading_layer.load(lib_, "MKL_Set_Threading_Layer");
        LAPACKE_sgelsd.load(lib_, "LAPACKE_sgelsd");
        mkl_set_threading_layer(3);
    }

    ~mkl_iface_t() {
        if (lib_) dlclose(lib_);
    }

    void *lib_ = nullptr;
};

} // namespace planner
} // namespace conv
} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
