#pragma once

#include "dnnl.hpp"

// A separate header for exposition purposes. All things here may be declared
// in dnnl.hpp.

namespace dnnl {

// Memory factory.
template <typename... Args>
memory make(const engine &e, const stream &s, Args &&... args);

// Returns the underlying memory storage object. Throws if the underlying
// memory storage object has a different type.
template <typename RuntimeT, typename... Args>
RuntimeT get_native(const memory &m, Args &&... args);

// Sets the underlying memory storage object. Throws if the memory object does
// not support this type of object.
template <typename... Args>
void set_native(memory &m, const stream &s, Args &&... args);

// Engine factory.
template <typename... Args>
engine make(engine::kind kind, Args &&... args);

// Returns the device or the context underlying an engine. Throws on type
// mismatch.
template <typename RuntimeT, typename... Args>
RuntimeT get_native(const engine &e, Args &&... args);

// Stream factory.
template <typename... Args>
stream make(const engine &e, Args &&... args);

// Returns the queue underlying a stream. Throws if there is none.
template <typename RuntimeT, typename... Args>
RuntimeT get_native(const stream &s, Args &&... args);

// Executes a primitive.
template <typename EventT>
EventT execute(const primitive &p, const stream &s,
        const std::unordered_map<int, memory> &args,
        const std::vector<EventT> &dependencies = {});

// Executes a primitive. Standalone function for consistency. Should belong to
// dnnl.hpp
void execute(const primitive &p, const stream &s,
        const std::unordered_map<int, memory> &args);

// Executes a 'reorder'.
template <typename EventT>
EventT execute(const primitive &p, const stream &s, const memory &src,
        memory &dst, const std::vector<EventT> &dependencies = {});

// Executes a 'reorder'. Standalone function for consistency. Should belong to
// dnnl.hpp
void execute(const primitive &p, const stream &s, const memory &src,
        memory &dst);
}
