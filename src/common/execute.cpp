#include <cassert>

#include "mkl_dnn.h"

#include "engine.hpp"
#include "nstl.hpp"
#include "utils.hpp"

namespace mkl_dnn { namespace impl {

// TODO: thread-safety

// Note on async engines:
//  - engines are responsible for tracking the dependencies: they cannot
//    schedule a primitive for execution unless all its unputs are ready

struct stream: public c_compatible {
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
                status_t s = engine->submit(i - base_idx + 1, &_queue[base_idx],
                        error_primitive);
                if (s != success)
                    return s;
                engine = _queue[i]->engine();
                base_idx = i;
            }
        return success;
    }

    status_t wait_queue(bool block, primitive **error_primitive) {
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
        } while (block);
        if (all_done) _queue.clear();
        return all_done ? success : try_again;
    }

public:
    stream(): _is_lazy(-1) {}

    status_t submit(size_t n, primitive *primitives[],
            primitive **error_primitive)
    {
        primitive *p;
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
                != nstl::success)
            return out_of_memory;
        if (!_is_lazy)
            return submit_queue(start_idx, error_primitive);
        return success;
    }

    status_t wait(bool block, primitive **error_primitive) {
        if (_is_lazy) {
            status_t rc = submit_queue(0, error_primitive);
            if (rc != success)
                return rc;
        }
        return wait_queue(block, error_primitive);
    }

};

}}

status_t stream_create(stream_t *stream)
{
    *stream = new mkl_dnn::impl::stream;
    return stream ? success : out_of_memory;
}

status_t stream_submit(stream_t stream,
        size_t n, primitive_t primitives[], primitive_t *error_primitive)
{
    auto s = reinterpret_cast<mkl_dnn::impl::stream*>(stream);
    return s->submit(n, reinterpret_cast<mkl_dnn::impl::primitive**>(primitives),
            reinterpret_cast<mkl_dnn::impl::primitive**>(error_primitive));
}

status_t stream_wait(stream_t stream, int block, primitive_t *error_primitive)
{
    auto s = reinterpret_cast<mkl_dnn::impl::stream*>(stream);
    return s->wait(!!block,
            reinterpret_cast<mkl_dnn::impl::primitive**>(error_primitive));
}

status_t stream_destroy(stream_t stream)
{
    auto s = reinterpret_cast<mkl_dnn::impl::stream*>(stream);
    delete s;
}

// vim: et ts=4 sw=4
