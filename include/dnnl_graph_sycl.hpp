#ifndef ONEAPI_DNNL_DNNL_GRAPH_SYCL_HPP
#define ONEAPI_DNNL_DNNL_GRAPH_SYCL_HPP

/// @cond DO_NOT_DOCUMENT_THIS
#include <vector>

#include <CL/sycl.hpp>

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_base.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.h"
/// @endcond

/// @addtogroup dnnl_graph_api
/// @{

namespace dnnl {
namespace graph {

/// @addtogroup dnnl_graph_api_interop Runtime interoperability API
/// API extensions to interact with the underlying run-time.
/// @{

/// @addtogroup dnnl_graph_api_sycl_interop SYCL interoperability API
/// API extensions to interact with the underlying SYCL run-time.
///
/// @{

/// SYCL interoperability namespace
namespace sycl_interop {

/// Constructs an engine from SYCL device and context objects.
///
/// @param adevice SYCL device.
/// @param acontext SYCL context.
///
/// @returns Created engine.
inline engine make_engine(
        const cl::sycl::device &adevice, const cl::sycl::context &acontext);

/// Creates an execution stream for a given engine associated with a SYCL
/// queue.
///
/// @param aengine Engine object to use for the stream.
/// @param aqueue SYCL queue to use for the stream.
///
/// @returns An execution stream.
inline stream make_stream(engine aengine, const cl::sycl::queue &aqueue);

/// Executes a compiled partition in a specified stream and returns a SYCL
/// event.
///
/// @param c_partition Compiled partition to execute.
/// @param astream Stream object to run over
/// @param inputs Arguments map.
/// @param outputs Arguments map.
/// @param deps Optional vector with `cl::sycl::event` dependencies.
/// @returns Output event.
inline cl::sycl::event execute(compiled_partition &c_partition, stream &astream,
        const std::vector<tensor> &inputs, std::vector<tensor> &outputs,
        const std::vector<cl::sycl::event> &deps = {});

} // namespace sycl_interop

/// @} dnnl_graph_api_sycl_interop

/// @} dnnl_graph_api_interop

} // namespace graph
} // namespace dnnl

/// @} dnnl_graph_api

#endif
