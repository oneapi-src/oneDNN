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

#ifndef UTILS_PARALLEL_HPP
#define UTILS_PARALLEL_HPP

#include <cstdint>
#include <functional>

// `benchdnn_parallel_nd` wrapper is essential for threadpool configuration
// since testing infrastructure contains a single threadpool object but obliges
// to differentiate is as two different - one for the library which is activated
// through the stream object and the other one is for internal library funtions
// used in tests such as `parallel_nd`. Threadpool for internal needs is
// referred as `scoped_threadpool` and its activation/deactivation is under test
// control. This is needed since each primitive execute call will activate the
// threadpool object passed to the library (which is supposed to be different
// from the "scoped") and if "scoped" threadpool will be left activated, it will
// cause an error while trying to activate already activated threadpool.

void benchdnn_parallel_nd(int64_t D0, const std::function<void(int64_t)> &f);
void benchdnn_parallel_nd(
        int64_t D0, int64_t D1, const std::function<void(int64_t, int64_t)> &f);
void benchdnn_parallel_nd(int64_t D0, int64_t D1, int64_t D2,
        const std::function<void(int64_t, int64_t, int64_t)> &f);
void benchdnn_parallel_nd(int64_t D0, int64_t D1, int64_t D2, int64_t D3,
        const std::function<void(int64_t, int64_t, int64_t, int64_t)> &f);
void benchdnn_parallel_nd(int64_t D0, int64_t D1, int64_t D2, int64_t D3,
        int64_t D4,
        const std::function<void(int64_t, int64_t, int64_t, int64_t, int64_t)>
                &f);
void benchdnn_parallel_nd(int64_t D0, int64_t D1, int64_t D2, int64_t D3,
        int64_t D4, int64_t D5,
        const std::function<void(
                int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)> &f);

int benchdnn_get_max_threads();

#endif
