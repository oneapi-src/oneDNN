class stream {
public:
    /// Constructs a stream for the specified engine.
    /// @param engine Engine to create stream on
    /// @param attr A stream attribute, defaults to nullptr    
    stream(engine &aengine, const stream_attr *attr = nullptr); 

    /// Waits for all compiled partitions executing in the stream to finish.
    /// @returns The stream itself.
    stream &wait();
};

/// Creates an execution stream for a given engine associated with a SYCL
/// queue.
///
/// @param aengine Engine object to use for the stream.
/// @param aqueue SYCL queue to use for the stream.
///
/// @returns An execution stream.
inline stream dnnl::graph::sycl_interop::make_stream(engine aengine,
        const cl::sycl::queue &aqueue);
