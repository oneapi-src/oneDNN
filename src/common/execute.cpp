#include <cassert>

#include "mkl_dnn.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "nstl.hpp"
#include "utils.hpp"

using namespace mkl_dnn::impl;

// TODO: thread-safety

// Note on async engines:
//  - engines are responsible for tracking the dependencies: they cannot
//    schedule a primitive for execution unless all its unputs are ready

struct mkl_dnn_stream: public c_compatible {
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
                if (s != mkl_dnn_success)
                    return s;
                engine = _queue[i]->engine();
                base_idx = i;
            }
        return mkl_dnn_success;
    }

    status_t wait_queue(bool block, primitive **error_primitive)
    {
        //assert(error_primitive);
        // This assumes that the engines start execution as soon as primitives
        // are submitted and do not need any additional notification about
        // wait()
        bool all_done;
        do {
            all_done = true;
            for (auto i = 0UL; i < _queue.size(); i++) {
                auto p = _queue[i];
                auto s = p->get_exec_state();
                switch (s) {
                    case primitive::exec_state::busy:
                    all_done = false;
                    break;
                case primitive::exec_state::done:
                    break;
                default:
                    *error_primitive = p;
                }
                if (!all_done) {
                    mkl_dnn_yield_thread();
                    break;
                }
            }
            if (all_done) break;
        } while (block);
        if (all_done)
            _queue.clear();
        return all_done ? mkl_dnn_success : mkl_dnn_try_again;
    }

public:
    mkl_dnn_stream(): _is_lazy(-1) {}

    status_t submit(size_t n, primitive *primitives[],
            primitive **error_primitive)
    {
        primitive *p;
        if (!error_primitive)
            error_primitive = &p;
        *error_primitive = 0;

        // Check all primitives have the same laziness
        int old_is_lazy = _is_lazy;
        if (_is_lazy == -1) _is_lazy = primitives[0]->engine()->is_lazy();
        for (size_t i = 0; i < n; i++)
            if (primitives[i]->engine()->is_lazy() != _is_lazy) {
                _is_lazy = old_is_lazy;
                return mkl_dnn_invalid;
            }

        // XXX: start_idx should be returned by _queue.insert()
        int start_idx = _queue.size();
        if (_queue.insert(_queue.end(), primitives, primitives + n)
                != mkl_dnn::impl::nstl::success)
            return mkl_dnn_out_of_memory;
        if (!_is_lazy)
            return submit_queue(start_idx, error_primitive);
        return mkl_dnn_success;
    }

    status_t wait(bool block, primitive **error_primitive) {
        if (_is_lazy) {
            status_t rc = submit_queue(0, error_primitive);
            if (rc != mkl_dnn_success)
                return rc;
        }
        return wait_queue(block, error_primitive);
    }
};

mkl_dnn_status_t mkl_dnn_stream_create(mkl_dnn_stream_t *stream) {
    *stream = new mkl_dnn_stream;
    return stream ? mkl_dnn_success : mkl_dnn_out_of_memory;
}

status_t mkl_dnn_stream_submit(mkl_dnn_stream_t stream, size_t n,
        mkl_dnn_primitive_t primitives[],
        mkl_dnn_primitive_t *error_primitive) {
    return stream->submit(n, primitives, error_primitive);
}

status_t mkl_dnn_stream_wait(mkl_dnn_stream_t stream, int block,
        mkl_dnn_primitive_t *error_primitive) {
    return stream->wait(!!block, error_primitive);
}

status_t mkl_dnn_stream_destroy(mkl_dnn_stream_t stream) {
    delete stream;
    return mkl_dnn_success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0
