#include <cassert>

#include "mkl_dnn.h"

#include "engine.hpp"
#include "nstl.hpp"
#include "utils.hpp"

// TODO: thread-safety

// Note on async engines:
//  - engines are responsible for tracking the dependencies: they cannot
//    schedule a primitive for execution unless all its unputs are ready

struct dnn_stream: public mkl_dnn::impl::c_compatible {
private:
    int _is_lazy;
    mkl_dnn::impl::nstl::vector<dnn_primitive*> _queue;

    status_t submit_queue(size_t start_idx, dnn_primitive **error_primitive) {
        assert(start_idx < _queue.size());
        assert(error_primitive);
        dnn_engine *engine = _queue[start_idx]->engine();
        size_t base_idx = start_idx;
        for (size_t i = start_idx; i < _queue.size(); i++)
            if (engine != _queue[i]->engine() || i == _queue.size() - 1) {
                status_t s = engine->submit(i - base_idx + 1, &_queue[base_idx],
                        error_primitive);
                if (s != success)
                    return s;
                engine = _queue[i]->engine();
                base_idx = i;
            }
        return success;
    }

    status_t wait_queue(bool block, dnn_primitive **error_primitive)
    {
        assert(error_primitive);
        // This assumes that the engines start execution as soon as primitives
        // are submitted and do not need any additional notification about
        // wait()
        bool all_done;
        do {
            all_done = true;
            for (auto i = 0; i < _queue.size(); i++) {
                auto p = _queue[i];
                auto s = p->get_exec_state();
                switch (s) {
                    case dnn_primitive::exec_state::busy:
                    all_done = false;
                    break;
                case dnn_primitive::exec_state::done:
                    break;
                default:
                    *error_primitive = p;
                }
                if (!all_done) {
                    mkl_dnn::impl::mkl_dnn_yield_thread();
                    break;
                }
            }
        } while (block);
        if (all_done) _queue.clear();
        return all_done ? success : try_again;
    }

public:
    dnn_stream(): _is_lazy(-1) {}

    status_t submit(size_t n, dnn_primitive *primitives[],
            dnn_primitive **error_primitive)
    {
        dnn_primitive *p;
        if (!error_primitive) error_primitive = &p;
        *error_primitive = 0;

        // Check all primitives have the same laziness
        int old_is_lazy = _is_lazy;
        if (_is_lazy == -1) _is_lazy = primitives[0]->engine()->is_lazy();
        for (size_t i = 0; i < n; i++)
            if (primitives[i]->engine()->is_lazy() != _is_lazy) {
                _is_lazy = old_is_lazy;
                return invalid;
            }

        // XXX: start_idx should be returned by _queue.insert()
        int start_idx = _queue.size();
        if (_queue.insert(_queue.end(), primitives, primitives + n)
                != mkl_dnn::impl::nstl::success)
            return out_of_memory;
        if (!_is_lazy)
            return submit_queue(start_idx, error_primitive);
        return success;
    }

    status_t wait(bool block, dnn_primitive **error_primitive) {
        if (_is_lazy) {
            status_t rc = submit_queue(0, error_primitive);
            if (rc != success)
                return rc;
        }
        return wait_queue(block, error_primitive);
    }
};

status_t stream_create(dnn_stream_t *stream) {
    *stream = new dnn_stream;
    return stream ? success : out_of_memory;
}

status_t stream_submit(dnn_stream_t stream, size_t n,
        dnn_primitive_t primitives[], dnn_primitive_t *error_primitive) {
    return stream->submit(n, primitives, error_primitive);
}

status_t stream_wait(dnn_stream_t stream, int block,
        dnn_primitive_t *error_primitive) {
    return stream->wait(!!block, error_primitive);
}

status_t stream_destroy(dnn_stream_t stream) {
    delete stream;
}

// vim: et ts=4 sw=4
