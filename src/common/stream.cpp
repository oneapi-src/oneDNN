/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include <cassert>

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "nstl.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;

// TODO: thread-safety

// Note on async engines:
//  - engines are responsible for tracking the dependencies: they cannot
//    schedule a primitive for execution unless all its unputs are ready

struct mkldnn_stream: public c_compatible {
private:
    int _is_lazy;
    nstl::vector<primitive*> _queue;

    status_t submit_queue(size_t start_idx, primitive **error_primitive) {
        assert(start_idx < _queue.size());
        assert(error_primitive);
        engine *engine = _queue[start_idx]->engine();
        size_t base_idx = start_idx;
        for (size_t i = start_idx; i < _queue.size(); i++)
            if (engine != _queue[i]->engine() || i == _queue.size() - 1) {
                status_t s = engine->submit(i - base_idx + 1,
                        &_queue[base_idx], error_primitive);
                if (s != success)
                    return s;
                engine = _queue[i]->engine();
                base_idx = i;
            }
        return success;
    }

    status_t wait_queue(bool block, primitive **error_primitive)
    {
        // This assumes that the engines start execution as soon as primitives
        // are submitted and do not need any additional notification about
        // wait()

        assert(error_primitive);
        bool all_ready;
        do {
            all_ready = true;
            for (auto i = 0UL; i < _queue.size(); i++) {
                auto p = _queue[i];
                auto s = p->get_exec_state();
                switch (s) {
                    case primitive::exec_state::not_ready:
                    all_ready = false;
                    break;
                case primitive::exec_state::ready:
                    break;
                default:
                    *error_primitive = p;
                }
                if (!all_ready) {
                    yield_thread();
                    break;
                }
            }
            if (all_ready) break;
        } while (block);
        if (all_ready)
            _queue.clear();
        return all_ready ? success : try_again;
    }

public:
    mkldnn_stream(): _is_lazy(-1) {}

    status_t submit(size_t n, primitive *primitives[],
            primitive **error_primitive)
    {
        primitive *p = 0;
        if (!error_primitive)
            error_primitive = &p;

        // Check all primitives have the same laziness
        int old_is_lazy = _is_lazy;
        if (_is_lazy == -1) _is_lazy = primitives[0]->engine()->is_lazy();
        for (size_t i = 0; i < n; i++)
            if (primitives[i]->engine()->is_lazy() != _is_lazy) {
                _is_lazy = old_is_lazy;
                return invalid_arguments;
            }

        // XXX: start_idx should be returned by _queue.insert()
        int start_idx = _queue.size();
        if (_queue.insert(_queue.end(), primitives, primitives + n)
                != mkldnn::impl::nstl::success)
            return out_of_memory;
        if (!_is_lazy)
            return submit_queue(start_idx, error_primitive);
        return success;
    }

    status_t wait(bool block, primitive **error_primitive) {
        primitive *p = 0;
        if (!error_primitive)
            error_primitive = &p;

        if (_is_lazy) {
            status_t rc = submit_queue(0, error_primitive);
            if (rc != success)
                return rc;
        }
        return wait_queue(block, error_primitive);
    }
};

status_t mkldnn_stream_create(stream **astream) {
    if (!astream)
        return invalid_arguments;
    *astream = new stream;
    return *astream ? success : out_of_memory;
}

status_t mkldnn_stream_submit(stream *astream, size_t n,
        primitive *primitives[],
        primitive **error_primitive) {
    return astream->submit(n, primitives, error_primitive);
}

status_t mkldnn_stream_wait(stream *astream, int block,
        primitive **error_primitive) {
    return astream->wait(!!block, error_primitive);
}

status_t mkldnn_stream_destroy(stream *astream) {
    delete astream;
    return success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
