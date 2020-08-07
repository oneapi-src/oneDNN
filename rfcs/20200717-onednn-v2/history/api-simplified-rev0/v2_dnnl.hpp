namespace dnnl {

// Maybe put under engine?
// Maybe call api_class to avoid overloading the word API?..
enum class api { c, sycl, ocl, threadpool };

struct engine : public handle<dnnl_engine_t> {
    enum class kind {
        any = dnnl_any_engine,
        cpu = dnnl_cpu,
        gpu = dnnl_gpu,
    };

    static size_t get_count(kind akind, api aapi = c);

    // For non-c api:
    // 1. Creation may fail if api is not supported
    // 2. The extra parameters in the equivalent dnnl::$api::make_engine(..)
    //    are implementation specific (but documented). The rule of thumb to
    //    choose parameters so that user's code continue working with as small
    //    changes, as possible.
    //    For instance, for DPCPP the memory model will be USM (but device?).
    //
    // For api::c one engine with kind != kind::cpu cannot be created.
    engine(kind akind, size_t index, api aapi = c);
    kind get_kind() const;
    api get_api() const;

    // It could also be useful to add the following query, that would return
    // true if the underlying memory model could be represented with `void *`:
    // - api::c || api::threadpool                          true
    // - api::sycl && memory_kind = usm_{device,shared}     true
    // - api::sycl && memory_kind = buffer                  false
    // - api::ocl                                           false
    bool is_memory_kind_pointer_based() const;
};

struct stream : public handle<dnnl_stream_t> {
    // Based on Eugene's proposal for `stream::flags`
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
    // doesn't support pointers (i.e. api=c or api=sycl && USM).
    // For DPCPP may fail if pointer doesn't correspond to the memory_kind.
    memory(const desc &md, const engine &aengine, void *handle);
    memory(const desc &md, const engine &aengine);
    // This ctor requires only engine to be alive along with memory, the stream
    // is only used to perform zero padding, if necessary, and is discarded
    // afterwards.
    memory(const desc &md, const stream &astream, void *handle);

    desc get_desc() const;
    engine get_engine() const;

    // Works only when memory buffer is represented by pure pointer, i.e.
    // engine::is_memory_kind_pointer_based() == true.
    void *get_data_handle() const;
    void set_data_handle(void *handle, const stream &astream) const;
    void set_data_handle(void *handle) const;

    template <typename T = void> T *map_data() const;
    void unmap_data(void *mapped_ptr) const;
};

struct primitive : public handle<dnnl_primitive_t> {
    void execute(const stream &astream,
            const std::unordered_map<int, memory> &args) const;
};

} // namespace dnnl
