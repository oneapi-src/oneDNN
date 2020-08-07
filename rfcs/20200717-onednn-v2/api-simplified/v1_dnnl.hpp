namespace dnnl {

struct engine : public handle<dnnl_engine_t> {
    enum class kind {
        any = dnnl_any_engine,
        cpu = dnnl_cpu,
        gpu = dnnl_gpu,
    };

    static size_t get_count(kind akind);

    engine(kind akind, size_t index);
    kind get_kind() const;

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    engine(kind akind, cl_device_id device, cl_context context);
    cl_context get_ocl_context() const;
    cl_device_id get_ocl_device() const;
#endif

#if DNNL_WITH_SYCL
    engine(kind akind, const cl::sycl::device &dev, const cl::sycl::context &ctx);
    cl::sycl::context DNNL_API get_sycl_context() const;
    cl::sycl::device DNNL_API get_sycl_device() const;
#endif
};

struct stream : public handle<dnnl_stream_t> {
    enum class flags : unsigned {
        default_order = dnnl_stream_default_order,
        in_order = dnnl_stream_default_order,
        out_of_order = dnnl_stream_out_of_order,
        default_flags = dnnl_stream_default_flags,
    };

    stream(const engine &aengine, flags aflags = flags::default_flags,
            const stream_attr &attr = stream_attr());

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    stream(const engine &aengine, cl_command_queue queue);
    cl_command_queue get_ocl_command_queue() const;
#endif

#if DNNL_WITH_SYCL
    stream(const engine &aengine, cl::sycl::queue &queue);
    cl::sycl::queue DNNL_API get_sycl_queue() const;
#endif

    stream &wait();
};

struct memory : public handle<dnnl_memory_t> {
    memory(const desc &md, const engine &aengine, void *handle);
    memory(const desc &md, const engine &aengine);

    desc get_desc() const;
    engine get_engine() const;

    void *get_data_handle() const;

    void set_data_handle(void *handle, const stream &astream) const;
    void set_data_handle(void *handle) const;

    template <typename T = void> T *map_data() const;
    void unmap_data(void *mapped_ptr) const;

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_mem get_ocl_mem_object() const;
    void set_ocl_mem_object(cl_mem mem_object);
#endif

#if DNNL_WITH_SYCL && defined(DNNL_USE_SYCL_BUFFERS)
    template <typename T, int ndims = 1>
    memory(const desc &md, const engine &aengine, cl::sycl::buffer<T, ndims> &buf);

    template <typename T, int ndims = 1>
    cl::sycl::buffer<T, ndims> get_sycl_buffer(size_t *offset = nullptr) const;

    template <typename T, int ndims>
    void set_sycl_buffer(cl::sycl::buffer<T, ndims> &buf);
#endif
};

struct primitive : public handle<dnnl_primitive_t> {
    void execute(const stream &astream,
            const std::unordered_map<int, memory> &args) const;

#ifdef DNNL_SYCL_DPCPP
    cl::sycl::event DNNL_API execute_sycl(const stream &astream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl::sycl::event> &deps = {}) const;
#endif
};

} // namespace dnnl
