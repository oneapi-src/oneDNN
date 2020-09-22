namespace dnnl {

struct engine : public handle<dnnl_engine_t> {
    enum class kind {
        any = dnnl_any_engine,
        cpu = dnnl_cpu,
        gpu = dnnl_gpu,
    };

    static size_t get_count(kind akind);

    engine(kind akind, size_t index = 0);

    kind get_kind() const;
    api get_api() const;
};

struct stream : public handle<dnnl_stream_t> {
    // Based on Eugene's proposal for dropping `stream::flags::default_order`:
    // ../changing-stream-flags-default-order-behavior.md
    enum class flags : unsigned {
        in_order = dnnl_stream_default_order,
        out_of_order = dnnl_stream_out_of_order,
        default_flags = in_order,
    };

    // stream_attr are dropped
    stream(const engine &aengine, flags aflags = flags::default_flags);
    stream &wait();
};

struct memory : public handle<dnnl_memory_t> {
    // May fail if handle != NONE / ALLOCATE and engine's memory model
    // doesn't support pointers (e.g. in case of OpenCL).
    memory(const desc &md, const engine &aengine,
            void *handle = DNNL_MEMORY_ALLOCATE);
    // This ctor requires only engine to be alive along with memory, the stream
    // is only used to perform zero padding, if necessary, and is discarded
    // afterwards.
    memory(const desc &md, const stream &astream,
            void *handle = DNNL_MEMORY_ALLOCATE);

    desc get_desc() const;
    engine get_engine() const;

    // Works only when memory buffer is represented by pure pointer.
    void *get_data_handle() const;
    void set_data_handle(void *handle) const;
    void set_data_handle(void *handle, const stream &astream) const;

    template <typename T = void> T *map_data() const;
    void unmap_data(void *mapped_ptr) const;
};

struct primitive : public handle<dnnl_primitive_t> {
    void execute(const stream &astream,
            const std::unordered_map<int, memory> &args) const;
};

} // namespace dnnl
