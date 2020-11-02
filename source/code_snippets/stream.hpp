class stream {
public:
    /// Constructs a stream for the specified engine.
    /// @param engine Engine to create stream on
    /// @param attr A stream attribute, defaults to nullptr    
    stream(engine &aengine, const stream_attr *attr = nullptr); 
 
    /// Constructs a stream for the specified engine and SYCL queue.
    /// @param engine Engine to create stream on
    /// @param queue SYCL queue to create stream on
    /// @param attr A stream attribute, defaults to nullptr
    stream(engine &aengine, const sycl::queue &queue, 
        const stream_attr *attr = nullptr); 
};
