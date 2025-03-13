/*******************************************************************************
* Copyright 2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GRAPH_EXAMPLE_COMPAT_COMPAT_HELPERS_HPP
#define GRAPH_EXAMPLE_COMPAT_COMPAT_HELPERS_HPP

#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "../graph_example_utils.hpp"

namespace compat {

// mimic cudnnHandle_t
class Handle {
    std::shared_ptr<dnnl::engine> eng_;
    std::shared_ptr<dnnl::stream> str_;

public:
    Handle(dnnl::engine::kind ekind, int index) {
        eng_ = std::make_shared<dnnl::engine>(ekind, index);
        str_ = std::make_shared<dnnl::stream>(*eng_);
    }

    dnnl::engine::kind get_engine_kind() const { return eng_->get_kind(); }

    dnnl::engine *get_engine() { return eng_.get(); }
    dnnl::stream *get_stream() { return str_.get(); }

    void synchronize() { str_->wait(); }
};

inline void fill_random(std::vector<float> &out) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;

    if (random_data_f.empty()) {
        std::mt19937 generator;
        std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator);
    }

    for (size_t i = 0; i < out.size(); i += nrand) {
        size_t chunk = std::min(nrand, out.size() - i);
        std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    }
}

// Surface is not part fo cuDNN API, but a utility for the example. Here we use
// it to allocate and initialize user data. We rely on the memory allocation in
// `dnnl::graph::tensor` inside the library, to avoid the explicit code for
// cpu/ocl/sycl memory allocation.
struct Surface {
    using lt = dnnl::graph::logical_tensor;

    int64_t n_elems_ = 0;
    Handle *handle_ = NULL;
    lt::data_type dt_ = lt::data_type::undef;
    dnnl::graph::tensor ts_;

protected:
    explicit Surface() {}

public:
    Surface(lt::data_type dt, int64_t n_elems, Handle *handle)
        : n_elems_(n_elems), handle_(handle), dt_(dt) {
        std::vector<float> raw_data(n_elems);
        fill_random(raw_data);
        const lt::dims d = {n_elems_};
        auto desc = dnnl::graph::logical_tensor(
                0, dt_, d, lt::layout_type::strided);
        ts_ = dnnl::graph::tensor(desc, *(handle_->get_engine()));
        // TODO(xxx): seems to have memory issue on gpu.
        // write_to_dnnl_tensor(raw_data.data(), ts_);
    }

    Surface(lt::data_type dt, int64_t n_elems, Handle *handle, float val)
        : n_elems_(n_elems), handle_(handle), dt_(dt) {
        std::vector<float> raw_data(n_elems, val);
        const lt::dims d = {n_elems_};
        auto desc = dnnl::graph::logical_tensor(
                0, dt_, d, lt::layout_type::strided);
        ts_ = dnnl::graph::tensor(desc, *(handle_->get_engine()));
        // TODO(xxx): seems to have memory issue on gpu.
        // write_to_dnnl_tensor(raw_data.data(), ts_);
    }

    void *get_ptr() const { return ts_.get_data_handle(); }

    ~Surface() = default;
};

} // namespace compat

namespace compat_0_x {

int64_t op_id = 0;
using lt = dnnl::graph::logical_tensor;
using op = dnnl::graph::op;
using partition = dnnl::graph::partition;
using compiled_partition = dnnl::graph::compiled_partition;
using graph = dnnl::graph::graph;
using DataType_t = lt::data_type;

constexpr int64_t MAX_OPGRAPH_OPS = 50;

enum class MHA_Layout {
    NOT_INTERLEAVED = 0,
    QKV_INTERLEAVED = 1,
    KV_INTERLEAVED = 2,
    SBH_INTERLEAVED = 3
};

enum class MHA_Matrix {
    Q_Matrix = 0, // queries
    K_Matrix = 1, // keys
    K_Matrix_Transpose = 2, // keys tranposed
    V_Matrix = 3, // values
    V_Matrix_Transpose = 4, // values transposed
    S_Matrix = 5, // output of GEMM1
    O_Matrix = 6, // final output
};

// mimic cudnnHandle_t
class Handle {
    std::shared_ptr<dnnl::engine> eng_;
    std::shared_ptr<dnnl::stream> str_;

public:
    Handle(dnnl::engine::kind ekind, int index) {
        eng_ = std::make_shared<dnnl::engine>(ekind, index);
        str_ = std::make_shared<dnnl::stream>(*eng_);
    }

    dnnl::engine::kind get_engine_kind() const { return eng_->get_kind(); }

    dnnl::engine *get_engine() { return eng_.get(); }
    dnnl::stream *get_stream() { return str_.get(); }

    void synchronize() { str_->wait(); }
};

inline void fill_random(std::vector<float> &out) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;

    if (random_data_f.empty()) {
        std::mt19937 generator;
        std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator);
    }

    for (size_t i = 0; i < out.size(); i += nrand) {
        size_t chunk = std::min(nrand, out.size() - i);
        std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    }
}

// Surface is not part fo cuDNN API, but a utility for the example. Here we use
// it to allocate and initialize user data. We rely on the memory allocation in
// `dnnl::graph::tensor` inside the library, to avoid the explicit code for
// cpu/ocl/sycl memory allocation.
struct Surface {
    using lt = dnnl::graph::logical_tensor;

    int64_t n_elems_ = 0;
    Handle *handle_ = NULL;
    lt::data_type dt_ = lt::data_type::undef;
    dnnl::graph::tensor ts_;

protected:
    explicit Surface() {}

public:
    Surface(lt::data_type dt, int64_t n_elems, Handle *handle)
        : n_elems_(n_elems), handle_(handle), dt_(dt) {
        std::vector<float> raw_data(n_elems);
        fill_random(raw_data);
        const lt::dims d = {n_elems_};
        auto desc = dnnl::graph::logical_tensor(
                0, dt_, d, lt::layout_type::strided);
        ts_ = dnnl::graph::tensor(desc, *(handle_->get_engine()));
        // TODO(xxx): seems to have memory issue on gpu.
        // write_to_dnnl_tensor(raw_data.data(), ts_);
    }

    Surface(lt::data_type dt, int64_t n_elems, Handle *handle, float val)
        : n_elems_(n_elems), handle_(handle), dt_(dt) {
        std::vector<float> raw_data(n_elems, val);
        const lt::dims d = {n_elems_};
        auto desc = dnnl::graph::logical_tensor(
                0, dt_, d, lt::layout_type::strided);
        ts_ = dnnl::graph::tensor(desc, *(handle_->get_engine()));
        // TODO(xxx): seems to have memory issue on gpu.
        // write_to_dnnl_tensor(raw_data.data(), ts_);
    }

    void *get_ptr() const { return ts_.get_data_handle(); }

    ~Surface() = default;
};

typedef void *onednnBackendDescriptor_t;

enum class BackendDescriptorType_t {
    ENGINE_CONFIG_TYPE,
    OPERATION_TYPE,
    UNKNOWN
};

dnnl_status_t create_descriptor(
        BackendDescriptorType_t type, onednnBackendDescriptor_t desc) {
    if (desc == nullptr) return dnnl_invalid_arguments;
    desc = nullptr;
    return dnnl_success;
}

dnnl_status_t destroy_descriptor(onednnBackendDescriptor_t desc) {
    return dnnl_success;
}

class OpaqueBackendPointer {
    onednnBackendDescriptor_t m_desc = nullptr; //!< Raw void pointer
    dnnl_status_t status
            = dnnl_success; //!< status of creation of the Descriptor

public:
    OpaqueBackendPointer(const OpaqueBackendPointer &)
            = delete; //!< Delete the copy constructor to prevent bad copies
    OpaqueBackendPointer &operator=(const OpaqueBackendPointer &) = delete;
    OpaqueBackendPointer(OpaqueBackendPointer &&) = default;

    /**
     * OpaqueBackendPointer constructor.
     * Calls the cudnnBackendCreateDescriptor. Allocates memory according to the type.
     */
    OpaqueBackendPointer(BackendDescriptorType_t type) {
        status = create_descriptor(type, &m_desc);
    }
    /**
     * OpaqueBackendPointer destructor.
     * Calls the cudnnBackendDestroyDescriptor. Frees memory allocated in the constructor.
     */
    ~OpaqueBackendPointer() { destroy_descriptor(m_desc); };
    /**
     * Accessor.
     * Returns the const reference to raw underlying descriptor.
     * Treat it like the data() function of a smart pointer. Can be freed behind the back.
     */
    onednnBackendDescriptor_t const &get_backend_descriptor() const {
        return m_desc;
    }
    /**
     * Accessor.
     * Queries the status of the descriptor after calling the cudnnCreate.
     */
    dnnl_status_t get_status() const { return status; }
    /**
     * Accessor.
     * Queries the status of the descriptor returns true if all good.
     */
    bool is_good() const { return status == dnnl_success; }
};

/*! \var A shared_ptr wrapper on top of the OpaqueBackendPointer */
using ManagedOpaqueDescriptor = std::shared_ptr<OpaqueBackendPointer>;

// /*! \fn A wrapper on top of the std::make_shared for the OpaqueBackendPointer */
// static ManagedOpaqueDescriptor make_shared_backend_pointer(
//         cudnnBackendDescriptorType_t type) {
//     return std::make_shared<OpaqueBackendPointer>(type);
// }

///
/// BackendDescriptor class
/// Holds a Managed pointer to OpaqueBackendPointer class
/// Contains the status and error message if set after any operation.
/// If exception is disabled the user must query the status after
/// build operation in order to check if the cudnn construct was built
/// correctly.
class BackendDescriptor {
public:
    //! Return a string describing the backend Descriptor
    //virtual std::string describe() const = 0;

    //! Get a copy of the raw descriptor pointer. Ownership is reatined and
    //! gets deleted when out of scope
    onednnBackendDescriptor_t get_raw_desc() const {
        return pointer->get_backend_descriptor();
    }

    //! Current status of the descriptor
    dnnl_status_t get_status() const { return status; }

    //! Set status of the descriptor
    void set_status(dnnl_status_t const status_) const { status = status_; }

    //! Set Diagonistic error message.
    void set_error(const char *message) const { err_msg = message; }

    //! Diagonistic error message if any
    const char *get_error() const { return err_msg.c_str(); }

    //! Returns a copy of underlying managed descriptor
    ManagedOpaqueDescriptor get_desc() const { return pointer; }

protected:
    /**
     * BackendDescriptor constructor.
     * Initializes the member variables as passed.
     */
    BackendDescriptor(ManagedOpaqueDescriptor pointer_, dnnl_status_t status_,
            std::string err_msg_)
        : pointer(pointer_), status(status_), err_msg(err_msg_) {}
    BackendDescriptor() = default;

    virtual ~BackendDescriptor() {};

    ManagedOpaqueDescriptor
            pointer; //! Shared pointer of the OpaqueBackendPointer

    mutable dnnl_status_t status
            = dnnl_success; //!< Error code if any being set
    mutable std::string err_msg; //!< Error message if any being set
};

// // v1 ------------------------------------------
// class Tensor_v8 : public BackendDescriptor {
// public:
//     friend class TensorBuilder_v8;
//     //     std::string describe() const override {
//     //         std::stringstream ss;
//     // #ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
//     //         ss << "CUDNN_BACKEND_TENSOR_DESCRIPTOR :"
//     //            << " Datatype: " << json {data_type} << " Id: " << std::to_string(id)
//     // #else
//     //         ss << "CUDNN_BACKEND_TENSOR_DESCRIPTOR :"
//     //            << " Datatype: " << int(data_type) << " Id: " << std::to_string(id)
//     // #endif
//     //            << " nDims " << nDims << " VectorCount: " << vectorCount
//     //            << " vectorDimension " << vectorDimension;
//     //         ss << " Dim [ ";
//     //         for (auto i = 0; i < nDims; i++) {
//     //             if (i != 0) { ss << ','; }
//     //             ss << btensor_dimA[i];
//     //         }
//     //         ss << " ] Str [ ";
//     //         for (auto i = 0; i < nDims; i++) {
//     //             if (i != 0) { ss << ','; }
//     //             ss << btensor_strA[i];
//     //         }
//     //         ss << " ]";
//     //         ss << " isVirtual: " << isVirtual << " isByValue: " << isByValue
//     //            << " Alignment: " << alignment;
//     // #ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
//     //         ss << " reorder_type: " << json {reorder_type};
//     // #else
//     //         ss << " reorder_type: " << int(reorder_type);
//     // #endif
//     //         return ss.str();
//     //     }

//     int64_t getDimCount() const { return nDims; }

//     lt::dims const getDim() const { return btensor_dimA; }

//     lt::dims const getStride() const { return btensor_strA; }

//     lt::data_type getDataType() const {
//         return static_cast<DataType_t>(data_type);
//     }

//     int64_t getId() const { return id; }

//     int64_t getAlignment() const { return alignment; }

//     bool isVirtualTensor() const { return isVirtual; }

//     lt const *getInternal_lt() const { return internal_lt; }

//     Tensor_v8(Tensor_v8 &&from) = default;
//     Tensor_v8 &operator=(Tensor_v8 &&) = default;

//     ~Tensor_v8() = default;

// private:
//     Tensor_v8() = default;
//     Tensor_v8(Tensor_v8 const &) = delete;
//     Tensor_v8 &operator=(Tensor_v8 const &) = delete;

//     DataType_t data_type = DataType_t::undef; //! Datatype of the elements
//     lt::dims btensor_dimA; //! n, g, c, d, h, w
//     lt::dims btensor_strA; //! n, g, c, d, h, w
//     int64_t id = -1; //! Unique id of the tensor
//     int64_t alignment = -1; //! Alignment of the tensor.
//     //! Certain engine config expect minimum alignment of 16B
//     int64_t nDims = -1; //! Number of Dimensions of the tensor
//     bool isVirtual
//             = false; //! Whether it is an intermediate tensor of an op graph
//     bool isByValue
//             = false; //! Whether the tensor is in host memory that needs to be passed to the kernel by value
//     lt *internal_lt;
// };

// /// TensorBuilder_v8 Class
// /// Helper class used to build Tensor_v8 class
// class TensorBuilder_v8 {
// public:
//     using lt = dnnl::graph::logical_tensor;
//     /** @defgroup TensorBuilder_v8
//      *  Set individual property of Tensor_v8 class
//      *  @{
//      */
//     //! Set Datatype for the Tensor_v8
//     auto setDataType(lt::data_type data_type) -> TensorBuilder_v8 & {
//         m_tensor.data_type = data_type;
//         return *this;
//     }

//     //! Set Dimensions of the tensor
//     // auto setDim(int64_t ndim, lt::dim const *dim) -> TensorBuilder_v8 & {
//     //     std::copy((dim), dim + ndim, m_tensor.btensor_dimA);
//     //     m_tensor.nDims = ndim;
//     //     return *this;
//     // }

//     auto setDim(int64_t ndim, const lt::dims &dims) -> TensorBuilder_v8 & {
//         m_tensor.nDims = dims.size();
//         std::copy(dims.begin(), dims.end(), m_tensor.btensor_dimA);
//         return *this;
//     }

//     //! Set Strides of the tensor
//     // auto setStride(int64_t ndim, lt::dim const *strides) -> TensorBuilder_v8 & {
//     //     std::copy(strides, strides + ndim, m_tensor.btensor_strA);
//     //     return *this;
//     // }

//     auto setStride(int64_t ndim, const lt::dims &strides)
//             -> TensorBuilder_v8 & {
//         std::copy(strides.begin(), strides.end(), m_tensor.btensor_strA);
//         return *this;
//     }

//     //! Set Unique Id  of the tensor
//     auto setId(int64_t id_) -> TensorBuilder_v8 & {
//         m_tensor.id = id_;
//         return *this;
//     }
//     //! Set Alignment of the tensor
//     auto setAlignment(int64_t alignment_) -> TensorBuilder_v8 & {
//         m_tensor.alignment = alignment_;
//         return *this;
//     }
//     //! Set isVirtual of the tensor
//     auto setVirtual(bool virtual_ = true) -> TensorBuilder_v8 & {
//         m_tensor.isVirtual = virtual_;
//         return *this;
//     }
//     //! Set isByValue of the tensor
//     auto setByValue(bool isByValue_ = true) -> TensorBuilder_v8 & {
//         m_tensor.isByValue = isByValue_;
//         return *this;
//     }

//     //! constructs the Tensor_v8 by calling the cudnn API
//     //! Throws the appropriate error message
//     Tensor_v8 &&build() {
//         auto lt_tmp = dnnl::graph::logical_tensor(m_tensor.getId(),
//                 m_tensor.getDataType(), m_tensor.getDim(),
//                 m_tensor.getStride());
//         m_tensor.internal_lt = &lt_tmp;
//         return std::move(m_tensor);
//     }

//     explicit TensorBuilder_v8() = default;
//     ~TensorBuilder_v8() = default;
//     TensorBuilder_v8(TensorBuilder_v8 &&) = delete;
//     TensorBuilder_v8(TensorBuilder_v8 const &) = delete;
//     TensorBuilder_v8 &operator=(TensorBuilder_v8 const &) = delete;

// private:
//     Tensor_v8 m_tensor; //! Tensor built by the TensorBuilder class.
// };

// using Tensor = Tensor_v8;
// using TensorBuilder = TensorBuilder_v8;

// class Operation_v8 : public BackendDescriptor {
// public:
//     friend class OperationBuilder_v8;
//     // std::string describe() const override {
//     //     std::stringstream ss;
//     //     ss << "CUDNN_BACKEND_OPERATION :"
//     //        << " OpMode: " << op_mode;
//     //     ss << std::hex << " X " << xdesc;
//     //     ss << std::hex << " Y " << ydesc;
//     //     ss << std::hex << " W " << wdesc;
//     //     ss << std::hex << " B " << bdesc;
//     //     ss << std::hex << " T " << tdesc;
//     //     ss << std::hex << " DW " << dwdesc;
//     //     ss << std::hex << " DY " << dydesc;
//     //     ss << std::hex << " DX " << dxdesc;
//     //     ss << std::hex << " C " << cdesc;
//     //     ss << std::hex << " A Mtrix " << amatdesc;
//     //     ss << std::hex << " B Mtrix " << bmatdesc;
//     //     ss << std::hex << " C Mtrix " << cmatdesc;
//     //     ss << std::hex << " P " << pwdesc;
//     //     ss << std::hex << " MatMul " << matmuldesc;
//     //     ss << std::hex << " Reduction " << reductiondesc;
//     //     ss << std::dec << " alphabetaType " << alphabetaType;
//     //     ss << " Alpha: " << alpha_s << " " << alpha_d;
//     //     ss << " Alpha2: " << alpha2_s << " " << alpha2_d;
//     //     ss << " Beta: " << beta_s << " " << beta_d;
//     //     return ss.str();
//     // }

//     Operation_v8(Operation_v8 &&from) = default;
//     Operation_v8 &operator=(Operation_v8 &&from) = default;

//     ~Operation_v8() = default;

// private:
//     Operation_v8() = default;
//     Operation_v8(Operation_v8 const &) = delete;
//     Operation_v8 &operator=(Operation_v8 const &) = delete;

//     op::kind op_kind = op::kind::MatMul;

//     lt *amatdesc = nullptr;
//     lt *bmatdesc = nullptr;
//     lt *cmatdesc = nullptr;

//     op *internal_op = nullptr;
// };

// class OperationBuilder_v8 {
// private:
//     Operation_v8 m_operation;

//     OperationBuilder_v8(op::kind op_kind) { m_operation.op_kind = op_kind; }

//     Operation_v8 &&build_matmul_op() {

//         // score = query x key.T
//         //auto score = logical_tensor(id++, dt, score_sz, layout_type::strided);
//         auto bmm1 = op(op_id++, op::kind::MatMul, "bmm1");
//         bmm1.set_attr<bool>(op::attr::transpose_b, true);
//         bmm1.add_inputs({m_operation.amatdesc, m_operation.bmatdesc});
//         bmm1.add_outputs({m_operation.cmatdesc});
//         m_operation.internal_op = &bmm1;
//         return std::move(m_operation);
//     }

// public:
//     auto setaMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
//         m_operation.amatdesc = tensor.getInternal_lt();
//         return *this;
//     }
//     auto setbMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
//         m_operation.bmatdesc = tensor.getInternal_lt();
//         return *this;
//     }
//     auto setcMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
//         m_operation.cmatdesc = tensor.getInternal_lt();
//         return *this;
//     }

//     //! constructs the backend Operation_v8 by calling the cudnn API
//     //! Throws the appropriate error message
//     Operation_v8 &&build() {
//         if (m_operation.op_kind == op::kind::MatMul) {
//             return build_matmul_op();
//         }
//         return std::move(m_operation);
//     }
// };

// using Operation = Operation_v8;
// using OperationBuilder = OperationBuilder_v8;

// // v1 ------------------------------------------------------

/*******************************************************
 *  Tensor_v8
 ******************************************************/
class Tensor_v8 : public BackendDescriptor {
public:
    friend class TensorBuilder_v8;

    int64_t getDimCount() const { return nDims; }
    lt::dims const getDim() const { return btensor_dimA; }
    lt::dims const getStride() const { return btensor_strA; }
    lt::data_type getDataType() const {
        return static_cast<DataType_t>(data_type);
    }
    int64_t getId() const { return id; }
    int64_t getAlignment() const { return alignment; }
    bool isVirtualTensor() const { return isVirtual; }

    // [CHANGED] internal_lt 改成 “引用”
    lt &getInternal_lt() const { return internal_lt; }

    // 构造函数
    // [CHANGED] 必须在构造时就绑定 internal_lt 引用
    Tensor_v8(lt &someLT)
        : internal_lt(someLT) // 在初始化列表中绑定
    {
        // 其他成员保持默认
    }

    // [CHANGED] 去除默认构造 Tensor_v8() = default;
    Tensor_v8(Tensor_v8 &&from) = default;
    Tensor_v8 &operator=(Tensor_v8 &&) = default;

    ~Tensor_v8() = default;

private:
    Tensor_v8(Tensor_v8 const &) = delete;
    Tensor_v8 &operator=(Tensor_v8 const &) = delete;

    DataType_t data_type = DataType_t::undef;
    lt::dims btensor_dimA;
    lt::dims btensor_strA;
    int64_t id = -1;
    int64_t alignment = -1;
    int64_t nDims = -1;
    bool isVirtual = false;
    bool isByValue = false;

    // [CHANGED]
    // 原先: lt *internal_lt;
    // 改为: lt &internal_lt;
    lt &internal_lt;
};

/*******************************************************
 *  TensorBuilder_v8
 ******************************************************/
class TensorBuilder_v8 {
public:
    using lt = dnnl::graph::logical_tensor;

    auto setDataType(lt::data_type data_type) -> TensorBuilder_v8 & {
        m_dataType = data_type;
        return *this;
    }
    auto setDim(int64_t ndim, const lt::dims &dims) -> TensorBuilder_v8 & {
        nDims = ndim;
        btensor_dimA = dims;
        return *this;
    }
    auto setStride(int64_t ndim, const lt::dims &strides)
            -> TensorBuilder_v8 & {
        btensor_strA = strides;
        return *this;
    }
    auto setId(int64_t id_) -> TensorBuilder_v8 & {
        m_id = id_;
        return *this;
    }
    auto setAlignment(int64_t alignment_) -> TensorBuilder_v8 & {
        m_alignment = alignment_;
        return *this;
    }
    auto setVirtual(bool virtual_ = true) -> TensorBuilder_v8 & {
        m_isVirtual = virtual_;
        return *this;
    }
    auto setByValue(bool isByValue_ = true) -> TensorBuilder_v8 & {
        m_isByValue = isByValue_;
        return *this;
    }

    // [CHANGED]
    // 这里不再生成局部 lt_tmp, 而是由 builder
    // 来管理一个更长生命周期的 dnnl::graph::logical_tensor
    // 以免我们把 Tensor_v8::internal_lt 绑定到局部变量
    // 这个 lumpsumLT 需要在 builder 的生命周期里保持有效
    // (理想情况更应该放在某个更持久的地方)
    Tensor_v8 build() {
        lumpsumLT = dnnl::graph::logical_tensor(
                m_id, m_dataType, btensor_dimA, btensor_strA);
        // 构造 Tensor_v8, 强行绑定 lumpsumLT
        Tensor_v8 t {lumpsumLT};
        t.data_type = static_cast<DataType_t>(m_dataType);
        t.btensor_dimA = btensor_dimA;
        t.btensor_strA = btensor_strA;
        t.id = m_id;
        t.alignment = m_alignment;
        t.nDims = nDims;
        t.isVirtual = m_isVirtual;
        t.isByValue = m_isByValue;
        // 现在 t.internal_lt 引用 lumpsumLT
        return t;
    }

    TensorBuilder_v8() = default;
    ~TensorBuilder_v8() = default;
    TensorBuilder_v8(TensorBuilder_v8 &&) = delete;
    TensorBuilder_v8(TensorBuilder_v8 const &) = delete;
    TensorBuilder_v8 &operator=(TensorBuilder_v8 const &) = delete;

private:
    // [CHANGED]
    // 先把原先 Tensor_v8 里的字段独立出来
    // builder 构造好后, 在 build() 时填到 lumpsumLT + Tensor_v8
    lt::data_type m_dataType = lt::data_type::undef;
    lt::dims btensor_dimA;
    lt::dims btensor_strA;
    int64_t m_id = -1;
    int64_t m_alignment = -1;
    int64_t nDims = -1;
    bool m_isVirtual = false;
    bool m_isByValue = false;

    // [CHANGED]
    // lumpsumLT: 用来承载 "long-lived" logical_tensor
    // 这样 build() 构造的 Tensor_v8.internal_lt 可引用 lumpsumLT
    // 但要注意 lumpsumLT 的生命周期 => builder 与 Tensor_v8
    dnnl::graph::logical_tensor lumpsumLT;
};

using Tensor = Tensor_v8;
using TensorBuilder = TensorBuilder_v8;

/*******************************************************
 *  Operation_v8 + OperationBuilder_v8
 ******************************************************/

class Operation_v8 : public BackendDescriptor {
public:
    friend class OperationBuilder_v8;

    // [CHANGED] 构造函数必须绑定 internal_op, 以及 amatdesc,bmatdesc,cmatdesc
    Operation_v8(lt &a, lt &b, lt &c, op &theop, op::kind k)
        : amatdesc(a)
        , bmatdesc(b)
        , cmatdesc(c)
        , internal_op(theop)
        , op_kind(k) {}

    Operation_v8(Operation_v8 &&from) = default;
    Operation_v8 &operator=(Operation_v8 &&from) = default;
    ~Operation_v8() = default;

private:
    Operation_v8() = delete; // 无法默认构造, 因为引用需要绑定
    Operation_v8(Operation_v8 const &) = delete;
    Operation_v8 &operator=(Operation_v8 const &) = delete;

    // [CHANGED] 全部改成引用
    lt &amatdesc;
    lt &bmatdesc;
    lt &cmatdesc;

    // [CHANGED] internal_op 也改为引用
    op &internal_op;

    op::kind op_kind;
};

class OperationBuilder_v8 {
private:
    op::kind op_kind;
    lt *aPtr = nullptr;
    lt *bPtr = nullptr;
    lt *cPtr = nullptr;

    // [CHANGED]
    // 需要一个 "long-lived" op 对象
    // 不能在 build() 里造局部 op bmm1 并返回对它的引用
    // 这里先放成一个成员 "bigOp"
    // 真实情况, 你也得确保这个 bigOp 生命周期比 Operation_v8 更长
    op bigOp;

public:
    explicit OperationBuilder_v8(op::kind k)
        : op_kind(k), bigOp(op_id++, k, "bmm1") {}

    auto setaMatDesc(const Tensor_v8 &tensor) -> OperationBuilder_v8 & {
        aPtr = &tensor.getInternal_lt();
        return *this;
    }
    auto setbMatDesc(const Tensor_v8 &tensor) -> OperationBuilder_v8 & {
        bPtr = &tensor.getInternal_lt();
        return *this;
    }
    auto setcMatDesc(const Tensor_v8 &tensor) -> OperationBuilder_v8 & {
        cPtr = &tensor.getInternal_lt();
        return *this;
    }

    Operation_v8 build() {
        // 检查
        if (!aPtr || !bPtr || !cPtr) {
            throw std::runtime_error(
                    "OperationBuilder_v8: some pointer not set!");
        }

        // [CHANGED]
        // bigOp 对象在 builder 里, 这里可以设置相关属性
        if (op_kind == op::kind::MatMul) {
            bigOp.set_attr<bool>(op::attr::transpose_b, true);
        }

        // bigOp.add_inputs(...) => 需要 pointer
        bigOp.add_inputs({*aPtr, *bPtr});
        bigOp.add_outputs({*cPtr});

        // 构造 Operation_v8(lt&, lt&, lt&, op&, op::kind)
        Operation_v8 opObj(*aPtr, *bPtr, *cPtr, bigOp, op_kind);

        return opObj;
    }
};

using Operation = Operation_v8;
using OperationBuilder = OperationBuilder_v8;

///
/// OperationGraph_v8 Class
/// This class tells the properties of the Tensor_v8 on which the operation will be
/// performed
/// Properties:
///    - handle
///    - operation
///
/// Use OperationGraphBuilder_v8 to build this class.
/// Describe returns a string describing the tensor class
///
class OperationGraph_v8 : public BackendDescriptor {
public:
    friend class OperationGraphBuilder_v8;
    // std::string describe() const override {
    //     std::stringstream ss;
    //     ss << "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR has " << numOps
    //        << " operations." << std::endl;
    //     ss << "Tag: " << opGraphTag << std::endl;
    //     return ss.str();
    // }

    OperationGraph_v8(OperationGraph_v8 &&from) = default;
    OperationGraph_v8 &operator=(OperationGraph_v8 &&from) = default;

    ~OperationGraph_v8() = default;

    uint64_t getOpCount() const { return numOps; }

    // const std::array<ManagedOpaqueDescriptor, MAX_OPGRAPH_OPS> &getOps() const {
    //     return ops;
    // }
    const std::array<Operation_v8, MAX_OPGRAPH_OPS> &getOps() const {
        return ops;
    }

private:
    OperationGraph_v8() = default;
    OperationGraph_v8(OperationGraph_v8 const &) = delete;
    OperationGraph_v8 &operator=(OperationGraph_v8 const &) = delete;

    Handle handle;
    //std::array<ManagedOpaqueDescriptor, MAX_OPGRAPH_OPS> ops {};
    std::array<Operation_v8, MAX_OPGRAPH_OPS> ops {};
    int64_t numOps = -1;
    graph *internal_graph = nullptr;
    partition *internal_partition = nullptr;
};

///
/// OperationGraphBuilder_v8 Class
/// Helper class used to build OperationGraph_v8 class
class OperationGraphBuilder_v8 {
public:
    /** @defgroup OperationGraphBuilder_v8
     *  Set individual property of OperationGraph_v8 class
     *  @{
     */

    auto setHandle(Handle handle_) -> OperationGraphBuilder_v8 & {
        m_operationGraph.handle = handle_;
        return *this;
    }

    // //! Set numoperations and the operations
    // auto setOperationGraph(int64_t numOps_, Operation_v8 const **ops_)
    //         -> OperationGraphBuilder_v8 & {
    //     m_operationGraph.numOps = numOps_;
    //     for (auto i = 0u; i < numOps_; i++) {
    //         m_operationGraph.ops[i] = ops_[i]->get_desc();
    //     }
    //     return *this;
    // }

    // //! Set numoperations and the operations
    // auto setOperationGraph(std::vector<Operation> const &ops_)
    //         -> OperationGraphBuilder_v8 & {
    //     m_operationGraph.numOps = ops_.size();
    //     for (auto i = 0u; i < ops_.size(); i++) {
    //         m_operationGraph.ops[i] = ops_[i].get_desc();
    //     }
    //     return *this;
    // }

    auto addOperation(ManagedOpaqueDescriptor desc)
            -> OperationGraphBuilder_v8 & {
        m_operationGraph.ops[m_operationGraph.numOps] = desc;
        ++m_operationGraph.numOps;
        return *this;
    }
    /** @} */

    //! constructs the OperationGraph_v8 by calling the cudnn API
    //! Throws the appropriate error message
    OperationGraph_v8 &&build() {

        dnnl::graph::graph BMM1(m_operationGraph.handle.get_engine_kind());

        for (auto i = 0u; i < m_operationGraph.numOps; i++) {
            auto op = m_operationGraph.ops[i];
            BMM1.add_op(op.internal_op);
        }

        BMM1.finalize();
        m_operationGraph.internal_graph = &BMM1;

        auto parts = BMM1.get_partitions();
        if (parts.size() != 1) throw std::runtime_error("partition failed ...");
        m_operationGraph.internal_partition = parts[0].get();

        return std::move(m_operationGraph);
    }

    explicit OperationGraphBuilder_v8() = default;
    ~OperationGraphBuilder_v8() = default;
    OperationGraphBuilder_v8(OperationGraphBuilder_v8 &&) = delete;
    OperationGraphBuilder_v8(OperationGraphBuilder_v8 const &) = delete;
    OperationGraphBuilder_v8 &operator=(OperationGraphBuilder_v8 const &)
            = delete;

private:
    OperationGraph_v8 m_operationGraph;
};

using OperationGraph = OperationGraph_v8;
using OperationGraphBuilder = OperationGraphBuilder_v8;

class EngineConfig_v8 : public BackendDescriptor {
public:
    friend class EngineConfigBuilder_v8;

    // std::string describe() const override {
    //     std::stringstream ss;
    //     ss << "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR :";
    //     ss << " Number of knobs: " << numKnobs;
    //     return ss.str();
    // }

    EngineConfig_v8 &operator=(EngineConfig_v8 &&from) = default;

    EngineConfig_v8(EngineConfig_v8 &&from) = default;

    ~EngineConfig_v8() = default;

    std::string const &getTag() const { return opGraphTag; }

private:
    EngineConfig_v8() = default;
    EngineConfig_v8(EngineConfig_v8 const &) = delete;
    EngineConfig_v8 &operator=(EngineConfig_v8 const &) = delete;
    partition *internal_partition = nullptr;
};

static inline std::vector<dnnl_status_t> get_heuristics_list(
        std::vector<std::string> const &modes, OperationGraph_v8 &opGraph,
        std::function<bool(onednnBackendDescriptor_t)> filter_fn,
        EngineConfigList &filtered_configs, bool evaluate_all = false) {
    std::vector<dnnl_status_t> statuses;

    auto parts = opGraph.internal_graph->get_partitions();
    if (parts.size() != 1) throw std::runtime_error("partition failed ...");
    filtered_configs.internal_configs = parts[0];
    return statuses;
}

using EngineConfig = EngineConfig_v8;

///
/// ExecutionPlan_v8 Class
/// This class tells the Configuration of the Engine in terms of the knob
/// choices
/// Properties:
///    - num knobs
///    - Choice
///    - Engine
///
/// Use ExecutionPlanBuilder_v8 to build this class.
/// Describe returns a string describing the tensor class
///
class ExecutionPlan_v8 : public BackendDescriptor {
public:
    friend class ExecutionPlanBuilder_v8;

    ExecutionPlan_v8(ExecutionPlan_v8 &&from) = default;
    ExecutionPlan_v8 &operator=(ExecutionPlan_v8 &&) = default;

    ~ExecutionPlan_v8() = default;
    /** @defgroup ExecutionPlanQuery
     *  Query individual property of ExecutionPlan_v8 class
     *  @{
     */
    //! Query the workspace requirement for the given plan

    // std::string describe() const override {
    //     std::stringstream ss;
    //     ss << "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR : ";
    //     ss << getTag() << ", ";
    //     ss << "numeric_notes:"
    //        << "[";
    //     for (auto note : numeric_notes_vec) {
    //         ss << cudnn_frontend::to_string(note) << ",";
    //     }
    //     ss << "] behavior_notes:"
    //        << "[";
    //     for (auto note : behavior_notes_vec) {
    //         ss << cudnn_frontend::to_string(note) << ",";
    //     }
    //     ss << "] workSpaceSize: " << workSpaceSize;
    //     return ss.str();
    // }

    ExecutionPlan_v8(ExecutionPlan_v8 const &) = default;
    ExecutionPlan_v8 &operator=(ExecutionPlan_v8 const &) = default;

private:
    ExecutionPlan_v8() = default;
    //EngineConfig *engine_config;
    ManagedOpaqueDescriptor engine_config = nullptr;
    Handle handle = nullptr;
    compiled_partition *internal_compiled_partition = nullptr;
};

///
/// ExecutionPlanBuilder_v8 Class
/// Helper class used to build ExecutionPlan_v8 class
class ExecutionPlanBuilder_v8 {
public:
    /** @defgroup ExecutionPlanBuilder_v8
     *  Set individual property of ExecutionPlan_v8 class
     *  @{
     */
    //! Set engine for the ExecutionPlan_v8
    auto setHandle(Handle handle_) -> ExecutionPlanBuilder_v8 & {
        m_execution_plan.handle = handle_;
        return *this;
    }

    //! Set engine Config for the Plan
    auto setEngineConfig(EngineConfig_v8 const &engine_config_)
            -> ExecutionPlanBuilder_v8 & {
        m_execution_plan.engine_config = engine_config_.get_desc();
        return *this;
    }

    //! constructs the Engine Config by calling the cudnn API
    //! Throws the appropriate error message
    ExecutionPlan_v8 &&build() {

        auto partition = m_execution_plan.engine_config.internal_partition;

        std::vector<logical_tensor> inputs
                = partition
                          .get_input_ports(); // Get the input ports of the partition
        std::vector<logical_tensor> outputs
                = partition
                          .get_output_ports(); // Get the output ports of the partition
        // Compile the partition with inputs, outputs, and an engine.
        m_execution_plan.internal_compiled_partition = partition.compile(
                inputs, {output}, *(m_execution_plan.handle.get_engine()));

        return std::move(m_execution_plan);
    }

    explicit ExecutionPlanBuilder_v8() = default;
    ~ExecutionPlanBuilder_v8() = default;
    ExecutionPlanBuilder_v8(ExecutionPlanBuilder_v8 &&) = delete;
    ExecutionPlanBuilder_v8(ExecutionPlanBuilder_v8 const &) = delete;
    ExecutionPlanBuilder_v8 &operator=(ExecutionPlanBuilder_v8 const &)
            = delete;

private:
    ExecutionPlan_v8 m_execution_plan;
};

using ExecutionPlan = ExecutionPlan_v8;
using ExecutionPlanBuilder = ExecutionPlanBuilder_v8;

void onednnGraphExecute(Handle handle, ExecutionPlan executionPlan,
        std::set<std::pair<uint64_t, void *>> data_ptrs) {

    auto eng = handle.get_engine();
    std::vector<logical_tensor> inputs
            = executionPlan.engine_config.internal_partition->get_input_ports();
    std::vector<logical_tensor> outputs
            = executionPlan.engine_config.internal_partition
                      ->get_output_ports();
    auto inputs_num = inputs.size();
    query = inputs[Q_ID];
    key = inputs[K_ID];
    out = outputs[O_ID - inputs_num];
    auto ts_q = tensor(query, *eng, data_ptrs[Q_ID]);
    auto ts_k = tensor(key, *eng, data_ptrs[K_ID]);
    auto ts_o = tensor(out, *eng, data_ptrs[O_ID]);

    executionPlan.internal_compiled_partition->execute(
            data_ptrs, *(handle.get_stream()));
}

} // namespace compat_0_x

#endif