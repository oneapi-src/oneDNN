class engine {
public:
/// engine kind
enum class kind {
    /// An unspecified engine
    any = dnnl_graph_any_engine,
    /// CPU engine
    cpu = dnnl_graph_cpu,
    /// GPU engine
    gpu = dnnl_graph_gpu,
};

/// Constructs an engine with specified kind and device_id
///
/// @param akind The kind of engine to construct
/// @param device_id Specify which device to be used
engine(kind akind, int device_id);

/// Returns device handle of the current engine
///
/// @returns Device handle
void *get_device_handle() const;
 
/// Returns device id of the current engine
///
/// @returns Device id
int get_device_id() const;
 
/// Returns concrete kind of the current engine
///
///@returns Kind of engine
kind get_kind() const;
};

/// Constructs an engine from SYCL device and context objects.
///
/// @param adevice SYCL device.
/// @param acontext SYCL context.
///
/// @returns Created engine.
inline engine dnnl::graph::sycl_interop::make_engine(
        const cl::sycl::device &adevice, const cl::sycl::context &acontext);
