# AMD backend support (RFC)

## Introduction
The idea for this RFC is to expose the AMD backend for oneDNN.
- This branch introduces HIP backend support for the primitives that are currently supported for CUDA backend.
- Testing and Performance analysis are yet to be done.
- Build process and compilation is successfully validated.

## Proposal
The primitives are built using MIOpen library, which are the open-source DNN libraries of AMD.
- The backend can be exposed to the user via DNNL_GPU_VENDOR=AMD flag used in CMake.
- This contribution would extend oneDNN's support for primitives and their supported post-ops and Quantization from CUDA backend to HIP backend.

Since MIOpen backend implementation is heavily inspired by cuDNN backend , the limitations or bugs from there are likely to be inherited here.

## Supported Primitives and Implementation Limitations:


## Binary:

Binary primitive in MIOpen library is implemented through miopenOpTensor, 
MIOpen supports only 4 modes of binary operations via enumerators: miopenTensorOpAdd, miopenTensorOpMul, miopenTensorOpMin, miopenTensorOpMax.

* This implementation keeps the same conditions as CUDA backend for blocking, broadcast and supported datatypes.
* Testing: This developed backend will be tested after addressing the comments from the oneDNN team and the results will be updated.

Limitations :
1. We currently do not have much information about blocking, broadcast in MIOpen library documentation.
2. MIOpen only supports tensor layout NCHW.
3. sum, binary & eltwise post-ops supported.


## ETLWISE :

The miopenActivationForward and miopenActivationBackward is the equivalent of eltwise forward and eltwise backward in oneDNN respectively.
The eltwise primitive works with arbitrary data tensors. There is no special meaning associated with any logical dimensions.
A primitive to perform elementwise operations such as the rectifier linear unit (ReLU).

Propagation: 
Forward,Backward

Limitations: 
 
1. Supported Data Types: supports f32, f16 data types. Doesn’t support s8 & bf16.
2. MIOpen  only supports the following operations - RELU, ELU, TANH, LOGISTIC,  CLIPPEDRELU, ABS, POWER.
3. Eltwise primitive doesn’t support binary post-ops in MIOpen.
4. Quantization is not applicable.



## LRN(LOCAL RESPONSE NORMALIZATION) :

The local response normalization primitive in the AMD backend is implemented with the
miopenLRNForward and miopenLRNBackward functions for forward and backward propagation respectively.

Propagation:
Forward and Backward.

Limitations :
1. Data Types: supports f32, f16 data types. Doesn’t support bf16.
2.The LRN primitive does not support any post-ops or attributes.
3.Quantization is not applicable.


## SOFTMAX :

Softmax implementation algorithms :
* Softmax has three implementations of its algorithms, these are enumerated upon "miopenSoftmaxAlgorithm_t" varible,
* Currently MIOpen supports three algorithmic implementations namely:
   MIOPEN_SOFTMAX_FAST
   MIOPEN_SOFTMAX_ACCURATE
   MIOPEN_SOFTMAX_LOG
* NOTE: The MIOPEN_SOFTMAX_LOG implementation in MIOpen is the direct implementation of logsoftmax in oneDNN.

Modes of Operation :
MIOPEN_SOFTMAX_MODE_INSTANCE(selected by default)
MIOPEN_SOFTMAX_MODE_CHANNEL(can be used if there is forwardv2 or backwardv2 implementation)

Propagation:
Forward and Backward.

Limitations:

1. Datatypes are limited to fp16 and fp32.
2. Supported Datatypes:
   miopenHalf  - 16-bit floating point(fp16)
   miopenFloat - 32-bit floating point(fp32)
3. Post-ops are not applicable.
4. Scaling the output is not supported .



## BATCH NORMALIZATION :

The equivalent to oneDNN batch normalization are miopenBatchNormalizationForward includes miopenBatchNormalizationForwardTraining,
miopenBatchNormalizationForwardInference and miopenBatchNormalizationBackward operations.

Propagation:
Forward and Backward.

LIMITATIONS :

1. Supported Data Types: 
supports f32, f16 data types. Doesn’t support s8 & bf16.
2. Eltwise post-op supported.
3. Quantization is not applicable.



## POOLING :

The pooling primitive in the AMD backend is implemented with the miopenPoolingForward 
and miopenPoolingBackward functions for forward and backward propagation respectively.

Propagation:
Forward and Backward.


LIMITATIONS :
1. Supported Data Types: 
supports f32, f16 ,u8, miopenint8data types. Doesn’t support s8 & bf16,S32.
2. The pooling primitive  is not supported for Binary postop in MIOpen.
3. Quantization is not applicable.



## CONVOLUTION  : 

1. This convolution primitive in AMD Backend is implemented as miopenConvolutionForward, miopenConvolutionBackward  
is used to compute forward, backward by data or backward by weights for a convolution operation.


PROPOGRATION :
Forward, BackwardData and BackwardBias.

LIMITATIONS :

1. Supported Data Types:  fp16, fp32, bf16, miopenInt8. Doesn’t support  U8 datatype.
2. sum, binary & eltwise post-ops supported.
4. Output scales are supported.
3. Zero-point attributes are not supported in MIOpen.  



## DECONVOLUTION :

Deconvolution primitive is implemented through the convolution with  miopenConvolutionBackwardBias.

Limitations :
1. Supported Data Types:  fp16, fp32, bf16, miopenInt8
doesn’t support U8 datatype.


##  System requirements 

 Versions of DPC++ compiler, HIP, MIOpen
 LLVM 14.0 from Intel
 DPC++ 2022.0.2
 ROCm 4.3 
 MIOpen 2.12



## Changes in CMake Module:

We have made additional changes to the existing cmake files,
for including AMD related libraries.

For Ex.

options.cmake
Line 232: if(NOT ""${DNNL_GPU_VENDOR}"" MATCHES ""^(INTEL|NVIDIA|AMD)$"")
Line 254:  if(DNNL_GPU_VENDOR STREQUAL ""AMD"")

SYCL.cmake
Line 73:end
find_package(HIP REQUIRED)
    # find_package(miopen REQUIRED)
    find_package(rocblas REQUIRED)

## NOTE :
1. There are some datatypes like bf16,s8,u8,s32 the support for these may subject to change after testing because it wasn't mentioned expliclity in the MIOpen documention.
2. MIOpen only supports tensor layout NCHW. 

## Build command
export CC=/path/to/hip/install/bin/clang    --> hip supported SYCL C compiler

export CXX=/path/to/hip/install/bin/clang++ -->  hip supported SYCL CPP compiler

mkdir build

cd build

cmake -DDNNL_CPU_RUNTIME=DPCPP -DDNNL_GPU_RUNTIME=SYCL -DDNNL_GPU_VENDOR=AMD ..

## Below is the structure of the code

## sycl_hip_compat.hpp

    namespace dnnl {
    namespace impl {
    namespace gpu {
    namespace amd {
    namespace compat {

    using interop_handle = ::sycl::interop_handler;
    T get_native_mem(const interop_handle &ih, U acc) {
    //Receives a SYCL accessor that has been defined as a requirement for the command group, and returns the memory object that is used by the SYCL runtime.
    }

    void host_task(::sycl::handler &cgh, const T &task) {
            //Enqueues a command to the SYCL runtime to invoke Func once.
    }


    template <typename native_object_t, typename sycl_object_t>
    native_object_t get_native(const sycl_object_t &sycl_object) {
     //Returns a SYCL application interoperability native backend object associated with syclObject.
    }

*********************************************************************************
## sycl_hip_stream.hpp

    class sycl_hip_stream_t : public dnnl::impl::sycl::sycl_stream_t {
    public:

    using base_t = dnnl::impl::sycl::sycl_stream_t;
    miopenHandle_t &get_miopen_handle();

    static status_t create_stream(...................) {
        //Creating a hip_stream
    }
    status_t interop_task(std::function<void(::sycl::handler &)>);
    hipStream_t get_underlying_stream();
    hipCtx_t get_underlying_context();
    };


***********************************************************
## sycl_hip_stream.cpp

    miopenHandle_t &sycl_hip_stream_t::get_miopen_handle() {
    //Function returning miopen_handle;;
    }

    HIPstream sycl_hip_stream_t::get_underlying_stream() {
    return compat::get_native<HIPstream>(*queue_);
    }

    status_t sycl_hip_stream_t::init() {
    if ((flags() & stream_flags::in_order) == 0
            && (flags() & stream_flags::out_of_order) == 0)
        return status::invalid_arguments;

    // If queue_ is not set then construct it
    auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine());
    auto status = status::success;
    //....................   
    }

    status_t sycl_hip_stream_t::interop_task(
        std::function<void(::sycl::handler &)> sycl_hip_interop_) {
    try {
        this->set_deps({queue().submit(
                [&](::sycl::handler &cgh) { sycl_hip_interop_(cgh); })});
        return status::success;
    } catch (std::runtime_error &e) {
        error::wrap_c_api(status::runtime_error, e.what());
        return status::runtime_error;
    }
}
***********************************************************************************************
## sycl_hip_utils.cpp

    //comparing lhs and rhs hip handle.
    bool compare_hip_devices(const ::sycl::device &lhs, const ::sycl::device &rhs) { 
        auto lhs_hip_handle = compat::get_native<HIPdevice>(lhs);
        auto rhs_hip_handle = compat::get_native<HIPdevice>(rhs);
        return lhs_hip_handle == rhs_hip_handle;
    }

******************************************************
## sycl_hip_scoped_contex.cpp

    hip_sycl_scoped_context_handler_t::hip_sycl_scoped_context_handler_t(
        const sycl_hip_engine_t &engine)
    : need_to_recover_(false) {
    try {
        auto desired = engine.get_underlying_context();
        .
        .
        .
        .
        .
        }
    } catch (const std::runtime_error &e) {
        error::wrap_c_api(status::runtime_error, e.what());
    }
    }

    hip_sycl_scoped_context_handler_t::
        ~hip_sycl_scoped_context_handler_t() noexcept(false) {
            ...
    }
**************************************************************************

## sycl_hip_scoped_contex.hpp


    // Scoped context is required to set the current context of a thread to the context of the using queue. The scoped handle class is
    // required to put the stream context on top of the hip stack
    class hip_sycl_scoped_context_handler_t {
        hipCtx_t original_;
        bool need_to_recover_;

    public:
     hip_sycl_scoped_context_handler_t(const sycl_hip_engine_t &);
     // Destruct the scope p_context placed_context_.
      ~hip_sycl_scoped_context_handler_t() noexcept(false);

      template <typename T, typename U>
      inline T memory(const compat::interop_handle &ih, U acc) {
          return compat::get_native_mem<T>(ih, acc);
    }
    };
******************************************************************************************

## sycl_hip_engine.hpp


    class hip_gpu_engine_impl_list_t {
    public:
    // The lists of reorder,concat,sum implementations.
    static const impl_list_item_t *get_reorder_implementation_list(
    const memory_desc_t *src_md, const memory_desc_t *dst_md);
    ................................
    };


    class sycl_hip_engine_t : public dnnl::impl::sycl::sycl_engine_base_t {
        public:
    using base_t = dnnl::impl::sycl::sycl_engine_base_t;
    
    //Creates an engine object using a specified SYCL device and SYCL context objects.
    sycl_hip_engine_t(engine_kind_t kind, const ::sycl::device &dev,
            const ::sycl::context &ctx, size_t index);
    sycl_hip_engine_t(const ::sycl::device &dev, const ::sycl::context &ctx,
            size_t index);
    
    //Constructs a stream for the specified engine and with behavior controlled by the specified flags.
    status_t create_stream(stream_t **stream, unsigned flags) override;
    status_t create_stream(stream_t **stream, ::sycl::queue &queue);
    
    //Defination for implementation list of reorder,sum, concat.
    const dnnl::impl::impl_list_item_t *get_reorder_implementation_list(
            const memory_desc_t *src_md,
            const memory_desc_t *dst_md) const override {
        return NULL;
    } 
    // activating miopen stream
    void activate_stream_miopen(stream_t *stream); 


    #ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
    protected:
    ~sycl_hip_engine_t() override = default;
    #endif

    private:
    // This functions sets the context type. Since hip requires different approach in retaining/releasing primary/non-primary context.
    status_t underlying_context_type();
    status_t set_miopen_handle();


    // To avoid performance penalty miopen required to have one handle per thread per context therefor the handles will be the properties of the
    // engine. an engine can be assigned to multiple streams: lets say engine eng(kind, 0); stream str1(eng,...); stream str2(eng,...); stream str3(eng,...); In multi-threading environment both engin and stream should be created in a different thread in order to allow safe
    //  multi-threading programming If all the streams belongs to one thread, the same handle will be used for all. Creation of handle is expensive and must be avoided when it is not necessary.
 
    utils::thread_local_storage_t<
            std::unique_ptr<miopenHandle_t, void (*)(miopenHandle_t *)>>
            miopen_handle_;
            bool primary_context_;
    };

*******************************
## sycl_hip_engine.cpp


    bool is_amd_gpu(const ::sycl::device &dev) {
    //Checking vendor id as amd
    ............................
    }

    status_t hip_engine_create(engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index) {
    // creating hip_engine 
    ...............................
    }

    sycl_hip_engine_t::sycl_hip_engine_t(engine_kind_t kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index)
    : base_t(kind, dev, ctx, index) {
    //Defination to Creates an engine object using a specified SYCL device and SYCL context objects.
    ................................................
    }

    sycl_hip_engine_t::sycl_hip_engine_t(
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index)
    : sycl_hip_engine_t(engine_kind::gpu, dev, ctx, index) {
    assert(is_amd_gpu(dev));
    //Creates an engine object using a specified SYCL device and include assert check for "is_amd_gpu(dev)" 
    }

    status_t sycl_hip_engine_t::set_miopen_handle() {
    // scoped context will make sure the top of the stack context is
    // the engine context while creating the miopen handle.
    hip_sycl_scoped_context_handler_t sc(*this);
    miopenHandle_t handle;
    CHECK(MIOPEN_EXECUTE_FUNC_S(miopenCreate, &handle));
     // ..
    }


    status_t sycl_hip_engine_t::create_stream(stream_t **stream, unsigned flags) {
    //Function defination Constructs a stream for the specified engine
    ......................................................
    }


    status_t sycl_hip_engine_t::underlying_context_type() {
    // this is a costly function which take avarage up to 75ms
    // on titanrx. So we must run it once and store the variable
    // in  is_primary_context_;
    }

    miopenHandle_t *sycl_hip_engine_t::get_miopen_handle() {
     // Check condition for miopen handle,if not set the miopen handle
    }


    device_id_t sycl_hip_engine_t::device_id() const {
     //Return device UUID
    }

    void sycl_hip_engine_t::activate_stream_miopen(stream_t *stream) {
    hip_sycl_scoped_context_handler_t sc(*this);
    //activate miopen stream here 
    }
    }

    namespace {
    using namespace dnnl::impl::data_type;

    // clang-format off
    //like Elementwise we will be calling instances for all the primitives eg: Softmax, convolution etc.
    constexpr dnnl::impl::impl_list_item_t sycl_hip_impl_list[] = {
       // primitive instances INSTANCE(miopen_<primitive>_fwd_t)
       //                     INSTANCE(miopen_<primitive>_bwd_t)
       // Elementwise example 
        INSTANCE(miopen_eltwise_fwd_t)
        INSTANCE(miopen_eltwise_bwd_t)
    };
    }
    const dnnl::impl::impl_list_item_t *sycl_hip_engine_t::get_implementation_list(
        const op_desc_t *) const {
    return sycl_hip_impl_list;
    }

*******************
## sycl_hip_utils.hpp

    // we define #define CTX_OUT_ACCESSOR(arg) for mode write
    //#define CTX_IN_ACCESSOR(arg)  for mode : read 
    //and #define CTX_SCRATCH_ACCESSOR(arg) for read_write.
    #define CTX_OUT_ACCESSOR(arg) \
    utils::downcast<sycl::sycl_buffer_memory_storage_t *>( \
            &CTX_OUT_STORAGE(arg)) \
            ->buffer() \
            .get_access<::sycl::access::mode::write>(cgh) 
            //Access to the buffer is controlled via an accessor
    #define CTX_IN_ACCESSOR(arg) \..................
    #define CTX_SCRATCH_ACCESSOR(arg) \......................


    //function declaration for compare_hip_devices
    bool compare_hip_devices(const ::sycl::device &lhs, const ::sycl::device &rhs);

    // Check if the device type matches the passed engine kind
    inline status_t check_device(dnnl::impl::engine_kind_t eng_kind) {
      // ..
    }

    // this func is coverting dnnl_dims_array
    static void convert_dnnl_dims_array(
        const dnnl_dim_t *dims, int *new_dims, int n_dims) {
    // converts dnnl_dims
     /..
    }

    // this func is coverting dims
    static void convert_dims(const dnnl_dim_t *dims, int *new_dims, int n_dims,
        int adjustment_size = 4, int adjustment_value = 1) {
    convert_dnnl_dims_array(dims, new_dims, n_dims);
     //..
    }

    static bool memory_desc_matches_nchw_vect_c(const memory_desc_t *mem_desc) {
    // Only one block is supported for second (C) dimension and the block size
    // must be 4 and the dimension has to be a multiple of block size.
      // ..
    }

    // expected to return true/false
    static bool has_different_block_size(
        const memory_desc_t *src_md, const memory_desc_t *dst_md) {
    return ((src_md->format_desc.blocking.inner_nblks > 0
                    && dst_md->format_desc.blocking.inner_nblks == 0)
            || (src_md->format_desc.blocking.inner_nblks == 0
                    && dst_md->format_desc.blocking.inner_nblks > 0));
    }

    //Check if they can adjust stride for dnn , returns true if they do.
    static bool adjust_dim_for_dnn(
        int *dims, int n_dims, const memory_desc_t *mem_desc) {
    if (memory_desc_matches_nchw_vect_c(mem_desc)) {
        dims[n_dims] = mem_desc->format_desc.blocking.inner_blks[0];
        dims[mem_desc->format_desc.blocking.inner_idxs[0]]
                /= mem_desc->format_desc.blocking.inner_blks[0];
        return true;
    }
    return false;
    }

    //Check if they can adjust stride for dnn , returns true if they do.
    static bool adjust_stride_for_dnn(
        int *stride, int n_dims, const memory_desc_t *mem_desc) {
    if (memory_desc_matches_nchw_vect_c(mem_desc)) {
        stride[n_dims] = mem_desc->format_desc.blocking.inner_nblks;
        return true;
    }
    return false;
    }

    // Check if the dimensions contain any zeros, returns true if they do.
    static bool has_zero_dims(const dnnl_dim_t *dims, int n_dims) {
    for (size_t i = 0; i < n_dims; i++) {
        if (dims[i] == 0) { return true; }
    }
    return false;
    }

    //setting miopenTensorFormat_t as per the format_tags value
    static status_t get_format(const memory_desc_t *md,
        miopenTensorFormat_t &format, bool consider_ab_as_nhwc = false) {
    //setting miopenTensorFormat_t as per the format_tags value
    ...........................................................
    }

    // expected to return true/false
    static bool memory_format_ok(const memory_desc_t *mem_desc) {
    return (memory_desc_matches_nchw_vect_c(mem_desc)
            || mem_desc->format_desc.blocking.inner_nblks == 0);
    }
    typedef enum {
    //list  out the tensor formats
    }miopenTensorFormat_t;


    static status_t convert_data_type(const memory_desc_t *mem_desc,
        miopenDataType_t *miopen_data_type, bool vectorized = true) {
    switch (mem_desc->data_type) {
        // Function to convert data types from dnnl to miopen 
    }
    
    //class having *hip_error_map, returning hip status 
    class hip_error : virtual public std::runtime_error {
    protected:
    inline const char *hip_error_map(hipError_t result) {
        switch (result) {
            //returning hipstatus
            case hipSuccess: return "hipSuccess";
            ..........   }
    int error_number_;

    public:
    explicit hip_error(const std::string &message, hipError_t result)
        : std::runtime_error((message + std::string(hip_error_map(result)))) {
        error_number_ = static_cast<int>(result);
    }
    virtual ~hip_error() throw() {}

    virtual int get_error_number() const throw() { return error_number_; }
    };

    template <typename T>
    ::sycl::event copy(::sycl::queue &q, T *src, ::sycl::buffer<T, 1> &dst) {

    auto event = q.submit([&, src](::sycl::handler &cgh) {
        // Retrieve a  write accessor to a global buffer
        auto acc = dst.template get_access<::sycl::access::mode::write,
                sycl::compat::target_device>(cgh);
        // Copy from the input pointer into the buffer associated with the
        // accessor
        cgh.copy(src, acc);
    });
    return event;
    }

    template <typename T>
    ::sycl::event copy(::sycl::queue &q, ::sycl::buffer<T, 1> &src, T *dst) {

    auto event = q.submit([&, dst](::sycl::handler &cgh) {
        // Retrieve a read accessor to a global buffer
        auto acc = src.template get_access<::sycl::access::mode::read,
                sycl::compat::target_device>(cgh);
        // Copy from the buffer associated with the accessor into the output
        // pointer
        cgh.copy(acc, dst);
    });

    return event;
    }

    template <typename T>
    ::sycl::event copy(::sycl::queue &q, ::sycl::buffer<T, 1> &src,
        ::sycl::buffer<T, 1> &dst) {
    auto event = q.submit([&](::sycl::handler &cgh) {
        auto src_acc
                = src.template get_access<::sycl::access::mode::read_write>(
                        cgh);
        auto dst_acc
                = dst.template get_access<::sycl::access::mode::read_write>(
                        cgh);
        cgh.copy(src_acc, dst_acc);
    });
    return event;
    }

    // this func is to find the status on coversion from miopen to dnnl
    static status_t miopen_to_dnnl_status(miopenStatus_t miopen_status) {
    switch (miopen_status) {
        case miopenStatusSuccess: return status::success;
        ..............................................
        ..............................................
    }
    }

    #define HIP_ERROR_LOCATION __FILE__ " : " STRINGIFY(__LINE__)


    #define MIOPEN_EXECUTE_FUNC(name, ...) \
    { \
      //primitive execution
    }
    #define MIOPEN_EXECUTE_FUNC_V(name, ...) \
    { \
    //Destroy tensor descriptor
    }

    #define MIOPEN_EXECUTE_FUNC_S(name, ...) \
    [&]() { \
        //To set creat and set descriptor
        \
    }()

    //this func is used to create and set tensor descriptor
    static status_t create_and_set_tensor_descriptor(
        miopenTensorDescriptor_t *tensor_desc, miopenDataType_t data_type,
        int ndims, int *dims, int *strides) {
        ........................................
        ........................................
        ........................................
    }

    //this func is used to create miopenCreateTensorDescriptor
    static status_t create_and_set_tensor_descriptor_ex(
        miopenTensorDescriptor_t *tensor_desc, miopenTensorFormat_t format,
        miopenDataType_t data_type, int ndims, int *dims) {
            ...........................
            ...........................
    return status::success;
    }

    //this func is used to create and set filter descriptor
    static status_t create_and_set_filter_descriptor(
    //this func is used to create and set descriptor for convolution layer

    return status::success;
    }

    class miopen_error : virtual public std::runtime_error {

    protected:
    inline const char *miopen_get_error_string(miopenStatus_t status) {
        switch (status) {
          // checking the status  for miopen and return the status message.
        }
    }
    int error_number_;

    public:
    explicit miopen_error(const std::string &message, miopenStatus_t result)
        : std::runtime_error(
                (message + std::string(miopen_get_error_string(result)))) {
        error_number_ = static_cast<int>(result);
    }

    virtual ~miopen_error() throw() {}

    virtual int get_error_number() const throw() { return error_number_; }
    };


## Open Questions
1.The implementation is subject to change as we go through the review and the testing phases

2.Currently the HIP support for DPCPP(SYCL) compiler is in experimental stage, and the backend is not completely supported on AMD devices

3.Hence this effort will also explore any alternatives for running HIP backend on AMD platforms
