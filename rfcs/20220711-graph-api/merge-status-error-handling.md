# API Status Code and Error Handling

In order to merge oneDNN Graph API into oneDNN and make the APIs to have a
consistent look and feel, we feel that it will better to unify the status code
on C API and the exception on C++ API. It will also help users to have a unified
way to handle the error and exceptions returned by oneDNN API.

Please note graph APIs mentioned in this document are based on [oneDNN graph
v0.5 release](https://github.com/oneapi-src/oneDNN/releases/tag/graph-v0.5)
which is slightly different from what they look on the latest dev-graph.

## Existing approach in oneDNN Graph technical preview

In oneDNN Graph API technical preview3, we provided the below enumerations as
the status code returned on C API. They are defined in `dnnl_graph_types.h`.

```cpp
/// Status values returned by the library functions.
typedef enum {
    /// The operation was successful
    dnnl_graph_result_success = 0,
    /// The operation was not ready
    dnnl_graph_result_not_ready = 1,
    /// The operation failed because device was not found
    dnnl_graph_result_error_device_not_found = 2,
    /// The operation failed because requested functionality is not implemented.
    dnnl_graph_result_error_unsupported = 3,
    /// The operation failed because of incorrect function arguments
    dnnl_graph_result_error_invalid_argument = 4,
    /// The operation failed because of the failed compilation
    dnnl_graph_result_error_compile_fail = 5,
    /// The operation failed because of incorrect index
    dnnl_graph_result_error_invalid_index = 6,
    /// The operation failed because of incorrect graph
    dnnl_graph_result_error_invalid_graph = 7,
    /// The operation failed because of incorrect shape
    dnnl_graph_result_error_invalid_shape = 8,
    /// The operation failed because of incorrect type
    dnnl_graph_result_error_invalid_type = 9,
    /// The operation failed because of incorrect op
    dnnl_graph_result_error_invalid_op = 10,
    /// The operation failed because of missing inputs or outputs
    dnnl_graph_result_error_miss_ins_outs = 11,
    /// Unknown error
    dnnl_graph_result_error_unknown = 0x7fffffff,
} dnnl_graph_result_t;
```

And, on C++ API layer, those errors returned by C API will be wrapped into
exceptions in type `dnnl::graph::error`.

```cpp
/// in namespace dnnl::graph::.
struct error : public std::exception {
    dnnl_graph_result_t result;
    std::string detailed_message;

    /// Constructs an instance of an exception class.
    ///
    /// @param result The error status returned by a C API function.
    /// @param message The error message.
    error(dnnl_graph_result_t result, const std::string &message)
        : result(result)
        , detailed_message(message + ": " + result2str(result)) {}

    /// Convert dnnl_graph_result_t to string.
    ///
    /// @param result The error status returned by a C API function.
    /// @return A string that describes the error status
    std::string result2str(dnnl_graph_result_t result) {
        switch (result) {
            case dnnl_graph_result_success: return "success";
            case dnnl_graph_result_not_ready: return "not ready";
            case dnnl_graph_result_error_device_not_found:
                return "device not found";
            case dnnl_graph_result_error_unsupported: return "unsupported";
            case dnnl_graph_result_error_invalid_argument:
                return "invalid argument";
            case dnnl_graph_result_error_compile_fail: return "compile fail";
            case dnnl_graph_result_error_invalid_index: return "invalid index";
            case dnnl_graph_result_error_invalid_graph: return "invalid graph";
            case dnnl_graph_result_error_invalid_shape: return "invalid shape";
            case dnnl_graph_result_error_invalid_type: return "invalid type";
            case dnnl_graph_result_error_invalid_op: return "invalid op";
            case dnnl_graph_result_error_miss_ins_outs:
                return "miss inputs or outputs";
            default: return "unknown error";
        }
    }

    /// Returns the explanatory string.
    ///
    /// @return A const char * that describes the error status
    const char *what() const noexcept override {
        return detailed_message.c_str();
    }

    /// Checks the return status and throws an error in case of failure.
    ///
    /// @param result The error status returned by a C API function.
    /// @param message The error message.
    static void check_succeed(
            dnnl_graph_result_t result, const std::string &message) {
        if (result != dnnl_graph_result_success) throw error(result, message);
    }
};
```

`dnnl::graph::error` also provided a convenient static function
`check_succeed()` which can be call in C++ API layer to check the return status
of C API and throw it as an exception when an error occurs.

All of these are very similar to the types and APIs supported in oneDNN.

- `dnnl_graph_result_t` is corresponding to `dnnl_status_t` with more status
  values describing errors in graph component.
- `dnnl::graph::error` is corresponding to `dnnl::error` with an additional
  improvement on the error message. It provides a leading string to describe the
  error code.
- The `check_succeed()` function is corresponding to `wrap_c_api()` in
  `dnnl::error`.

In the following sections, we wil describe the proposals for merging status code
and error handling in graph API and primitive API.

## Merge status code

We propose to merge `dnnl_graph_result_t` and `dnnl_status_t` and use
`dnnl_status_t` to describe the return status of both graph API and primitive
API.

The new enumeration type after merge will look like as follows to preserve the
backward compatibility.

```cpp
/// Status values returned by the library functions.
typedef enum {
    /// The operation was successful
    dnnl_success = 0,
    /// The operation failed due to an out-of-memory condition
    dnnl_out_of_memory = 1,
    /// The operation failed because of incorrect function arguments
    dnnl_invalid_arguments = 2,
    /// The operation failed because requested functionality is not implemented
    dnnl_unimplemented = 3,
    /// Primitive iterator passed over last primitive descriptor
    dnnl_iterator_ends = 4,
    /// Primitive or engine failed on execution
    dnnl_runtime_error = 5,
    /// Queried element is not required for given primitive
    dnnl_not_required = 6,

    /// New status values added for graph API
    /// The graph is not legitimate
    dnnl_invalid_graph = 7,
    /// The operation is not legitimate according to op schema
    dnnl_invalid_op = 8,
    /// The shape cannot be inferred or compiled
    dnnl_invalid_shape = 9,
    /// The data type cannot be inferred or compiled
    dnnl_invalid_data_type = 10,
} dnnl_status_t;
```

To map the values in `dnnl_graph_result_t` to the new `dnnl_status_t`, the
considerations are listed as follows:

- `dnnl_graph_result_success` can be mapped to the existing `dnnl_success`.
- `dnnl_graph_result_error_invalid_argument` and
  `dnnl_graph_result_error_miss_ins_outs` can be mapped to the existing
  `dnnl_invalid_arguments`.
- `dnnl_graph_result_error_unsupported` can be mapped to the existing
  `dnnl_unimplemented`.
- `dnnl_graph_result_not_ready`, `dnnl_graph_result_error_device_not_found`,
  `dnnl_graph_result_error_unknown`, and `dnnl_graph_result_error_invalid_index`
  are not used in technical previews. Hence removed.
- `dnnl_graph_result_error_compile_fail` should be replaced by more descriptive
  error values. Hence removed.
- `dnnl_invalid_graph`, `dnnl_invalid_op`, `dnnl_invalid_shape`, and
  `dnnl_invalid_data_type` are added to describe the errors happened in graph API.

The new `dnnl_status_t` will be defined in `dnnl_types.h` and used in
`dnnl_graph.h` as the returned value of graph C APIs.

## Exception on C++ API

`dnnl::error` will be kept with adding a new member function to convert status
code into a string. The string can be appended to the error message to improve
the debugging experience.

```cpp
struct error : public std::exception {
    // ...

    error(dnnl_status_t status, const std::string &msg)
        : result(result)
        , message(msg + ": " + str(status)) {}

    std::string str(dnnl_status_t status) const {
        switch (status) {
            case dnnl_success: return "success";
            case dnnl_out_of_memory: return "out of memory";
            case dnnl_invalid_arguments: return "invalid arguments";
            case dnnl_unimplemented: return "unimplemented";
            case dnnl_iterator_ends: return "iterator ends";
            case dnnl_runtime_error: return "runtime error";
            case dnnl_not_required: return "not required";
            case dnnl_invalid_graph: return "invalid graph";
            case dnnl_invalid_op: return "invalid op";
            case dnnl_invalid_shape: return "invalid shape";
            case dnnl_invalid_data_type: return "invalid data type";
            default: return "unknown error";
        }
    }

    // ...
};
```

`dnnl::error` will be used in `dnnl_graph.hpp` as the exception type raised in
graph C++ API layer. `wrap_c_api()` will be used to check the return values of C
API. To achieve this, we may need to consider moving `dnnl::error` into a common
header file to avoid including the huge `dnnl.hpp` into `dnnl_graph.hpp`.
