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

using lt = dnnl::graph::logical_tensor;
using Operation = dnnl::graph::op;
using DataType_t = logical_tensor::data_type;
//using Tensor = dnnl::graph::logical_tensor;

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

///

class Tensor_v8 {
public:
    friend class TensorBuilder_v8;
    //     std::string describe() const override {
    //         std::stringstream ss;
    // #ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    //         ss << "CUDNN_BACKEND_TENSOR_DESCRIPTOR :"
    //            << " Datatype: " << json {data_type} << " Id: " << std::to_string(id)
    // #else
    //         ss << "CUDNN_BACKEND_TENSOR_DESCRIPTOR :"
    //            << " Datatype: " << int(data_type) << " Id: " << std::to_string(id)
    // #endif
    //            << " nDims " << nDims << " VectorCount: " << vectorCount
    //            << " vectorDimension " << vectorDimension;
    //         ss << " Dim [ ";
    //         for (auto i = 0; i < nDims; i++) {
    //             if (i != 0) { ss << ','; }
    //             ss << btensor_dimA[i];
    //         }
    //         ss << " ] Str [ ";
    //         for (auto i = 0; i < nDims; i++) {
    //             if (i != 0) { ss << ','; }
    //             ss << btensor_strA[i];
    //         }
    //         ss << " ]";
    //         ss << " isVirtual: " << isVirtual << " isByValue: " << isByValue
    //            << " Alignment: " << alignment;
    // #ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    //         ss << " reorder_type: " << json {reorder_type};
    // #else
    //         ss << " reorder_type: " << int(reorder_type);
    // #endif
    //         return ss.str();
    //     }

    int64_t getDimCount() const { return nDims; }

    lt::dim const *getDim() const { return btensor_dimA; }

    lt::dim const *getStride() const { return btensor_strA; }

    lt::data_type getDataType() const {
        return static_cast<int64_t>(data_type);
    }

    int64_t getId() const { return id; }

    int64_t getAlignment() const { return alignment; }

    bool isVirtualTensor() const { return isVirtual; }

    lt const *getInternal_lt() const { return internal_lt; }

    Tensor_v8(Tensor_v8 &&from) = default;
    Tensor_v8 &operator=(Tensor_v8 &&) = default;

    ~Tensor_v8() = default;

private:
    Tensor_v8() = default;
    Tensor_v8(Tensor_v8 const &) = delete;
    Tensor_v8 &operator=(Tensor_v8 const &) = delete;

    DataType_t data_type = DataType_t::undef; //! Datatype of the elements
    int64_t btensor_dimA[CUDNN_DIM_MAX + 1] = {-1}; //! n, g, c, d, h, w
    int64_t btensor_strA[CUDNN_DIM_MAX + 1] = {-1}; //! n, g, c, d, h, w
    int64_t id = -1; //! Unique id of the tensor
    int64_t alignment = -1; //! Alignment of the tensor.
    //! Certain engine config expect minimum alignment of 16B
    int64_t nDims = -1; //! Number of Dimensions of the tensor
    bool isVirtual
            = false; //! Whether it is an intermediate tensor of an op graph
    bool isByValue
            = false; //! Whether the tensor is in host memory that needs to be passed to the kernel by value
    lt *internal_lt;
};

/// TensorBuilder_v8 Class
/// Helper class used to build Tensor_v8 class
class TensorBuilder_v8 {
public:
    using lt = dnnl::graph::logical_tensor;
    /** @defgroup TensorBuilder_v8
     *  Set individual property of Tensor_v8 class
     *  @{
     */
    //! Set Datatype for the Tensor_v8
    auto setDataType(lt::data_type data_type) -> TensorBuilder_v8 & {
        m_tensor.data_type = data_type;
        return *this;
    }

    //! Set Dimensions of the tensor
    auto setDim(int64_t ndim, lt::dim const *dim) -> TensorBuilder_v8 & {
        std::copy((dim), dim + ndim, m_tensor.btensor_dimA);
        m_tensor.nDims = ndim;
        return *this;
    }
    //! Set Strides of the tensor
    auto setStride(int64_t ndim, lt::dim const *strides) -> TensorBuilder_v8 & {
        std::copy(strides, strides + ndim, m_tensor.btensor_strA);
        return *this;
    }
    //! Set Unique Id  of the tensor
    auto setId(int64_t id_) -> TensorBuilder_v8 & {
        m_tensor.id = id_;
        return *this;
    }
    //! Set Alignment of the tensor
    auto setAlignment(int64_t alignment_) -> TensorBuilder_v8 & {
        m_tensor.alignment = alignment_;
        return *this;
    }
    //! Set isVirtual of the tensor
    auto setVirtual(bool virtual_ = true) -> TensorBuilder_v8 & {
        m_tensor.isVirtual = virtual_;
        return *this;
    }
    //! Set isByValue of the tensor
    auto setByValue(bool isByValue_ = true) -> TensorBuilder_v8 & {
        m_tensor.isByValue = isByValue_;
        return *this;
    }

    //! constructs the Tensor_v8 by calling the cudnn API
    //! Throws the appropriate error message
    Tensor_v8 &&build() {
        m_tensor.internal_lt = dnnl::graph::logical_tensor(m_tensor.getId(),
                m_tensor.getDataType(), m_tensor.getDim(),
                m_tensor.getStride());
        return std::move(m_tensor);
    }

    explicit TensorBuilder_v8() = default;
    ~TensorBuilder_v8() = default;
    TensorBuilder_v8(TensorBuilder_v8 &&) = delete;
    TensorBuilder_v8(TensorBuilder_v8 const &) = delete;
    TensorBuilder_v8 &operator=(TensorBuilder_v8 const &) = delete;

private:
    Tensor_v8 m_tensor; //! Tensor built by the TensorBuilder class.
};

using Tensor = Tensor_v8;
using TensorBuilder = TensorBuilder_v8;

class Operation_v8 : public BackendDescriptor {
public:
    friend class OperationBuilder_v8;
    // std::string describe() const override {
    //     std::stringstream ss;
    //     ss << "CUDNN_BACKEND_OPERATION :"
    //        << " OpMode: " << op_mode;
    //     ss << std::hex << " X " << xdesc;
    //     ss << std::hex << " Y " << ydesc;
    //     ss << std::hex << " W " << wdesc;
    //     ss << std::hex << " B " << bdesc;
    //     ss << std::hex << " T " << tdesc;
    //     ss << std::hex << " DW " << dwdesc;
    //     ss << std::hex << " DY " << dydesc;
    //     ss << std::hex << " DX " << dxdesc;
    //     ss << std::hex << " C " << cdesc;
    //     ss << std::hex << " A Mtrix " << amatdesc;
    //     ss << std::hex << " B Mtrix " << bmatdesc;
    //     ss << std::hex << " C Mtrix " << cmatdesc;
    //     ss << std::hex << " P " << pwdesc;
    //     ss << std::hex << " MatMul " << matmuldesc;
    //     ss << std::hex << " Reduction " << reductiondesc;
    //     ss << std::dec << " alphabetaType " << alphabetaType;
    //     ss << " Alpha: " << alpha_s << " " << alpha_d;
    //     ss << " Alpha2: " << alpha2_s << " " << alpha2_d;
    //     ss << " Beta: " << beta_s << " " << beta_d;
    //     return ss.str();
    // }

    Operation_v8(Operation_v8 &&from) = default;
    Operation_v8 &operator=(Operation_v8 &&from) = default;

    ~Operation_v8() = default;

private:
    Operation_v8() = default;
    Operation_v8(Operation_v8 const &) = delete;
    Operation_v8 &operator=(Operation_v8 const &) = delete;

    DescriptorType_t op_mode = DescriptorType_t::NOT_SET;

    lt *amatdesc = nullptr;
    lt *bmatdesc = nullptr;
    lt *cmatdesc = nullptr;
};

class OperationBuilder_v8 {
private:
    Operation_v8 m_operation;
    Operation_v8 &&build_matmul_op() {}

public:
    auto setaMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.amatdesc = tensor.getInternal_lt();
        return *this;
    }
    auto setbMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.bmatdesc = tensor.getInternal_lt();
        return *this;
    }
    auto setcMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.cmatdesc = tensor.getInternal_lt();
        return *this;
    }

    //! constructs the backend Operation_v8 by calling the cudnn API
    //! Throws the appropriate error message
    Operation_v8 &&build() {
        if (m_operation.status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, m_operation.status,
                    "CUDNN_BACKEND_OPERATION: Operation not initialized "
                    "properly");
            return std::move(m_operation);
        }

        Message_t msg = nullptr;
        cudnnStatus_t status_ = CUDNN_STATUS_SUCCESS;
        if (is_convolution_op) {
            status_ = validate_convolution_op(msg);
        } else if (is_pointwise_op) {
            status_ = validate_pointwise_op(msg);
        } else if (is_matmul_op) {
            status_ = validate_matmul_op(msg);
        } else if (is_reduction_op) {
            status_ = validate_reduction_op(msg);
        } else if (is_genstats_op) {
            status_ = CUDNN_STATUS_SUCCESS;
        } else if (is_bn_finalize_op) {
            status_ = CUDNN_STATUS_SUCCESS;
        } else if (is_bn_bwd_weight) {
            status_ = validate_bn_bwd_weight_op(msg);
        } else if (is_resample_fwd_op) {
            status_ = validate_resample_op(msg);
        } else if (is_resample_bwd_op) {
            status_ = validate_resample_op(msg);
        } else if (is_rng_op) {
            status_ = validate_rng_op(msg);
        } else if (is_norm_forward_op || is_norm_backward_op) {
            status_ = validate_norm_op(msg);
        } else if (is_reshape_op) {
            status_ = validate_reshape_op(msg);
        } else if (is_paged_cache_load_op) {
            status_ = CUDNN_STATUS_SUCCESS;
        } else {
            status_ = CUDNN_STATUS_BAD_PARAM;
            msg = "CUDNN_BACKEND_OPERATION_DESCRIPTOR: Unsupported cudnn "
                  "backend descriptor type. Check and set "
                  "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR";
        }
        if (status_ != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status_, msg);
            return std::move(m_operation);
        }

        // Create the descriptor.
        cudnnBackendDescriptorType_t cudnn_backend_descriptor_type;
        auto status = detail::convert_to_cudnn_type(
                m_operation.op_mode, cudnn_backend_descriptor_type);
        if (status != CUDNN_STATUS_SUCCESS) {
            std::stringstream ss;
            ss << "CUDNN_BACKEND_OPERATION: unable to identify backend "
                  "operation for "
               << m_operation.op_mode;
            set_error_and_throw_exception(
                    &m_operation, status, (ss.str()).c_str());
            return std::move(m_operation);
        }
        status = m_operation.initialize_managed_backend_pointer(
                cudnn_backend_descriptor_type);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status,
                    "CUDNN_BACKEND_OPERATION: cudnnCreate Failed");
            return std::move(m_operation);
        }

        if (m_operation.op_mode
                == DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
            return build_conv_forward();
        } else if (m_operation.op_mode
                == DescriptorType_t::
                        OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
            return build_conv_backward_filter();
        } else if (m_operation.op_mode
                == DescriptorType_t::
                        OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
            return build_conv_backward_data();
        } else if (m_operation.op_mode
                == DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR) {
            return build_pointwise_op();
        } else if (m_operation.op_mode
                == DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR) {
            return build_matmul_op();
        } else if (m_operation.op_mode
                == DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR) {
            return build_reduction_op();
        } else if (m_operation.op_mode
                == DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR) {
            return build_genstats_op();
        } else if (m_operation.op_mode
                == DescriptorType_t::
                        OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR) {
            return build_bn_finalize_op();
        } else if (m_operation.op_mode
                == DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR) {
            return build_bn_bwd_weight_op();
        } else if (m_operation.op_mode
                == DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR) {
            return build_resample_fwd_operation();
        } else if (m_operation.op_mode
                == DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR) {
            return build_norm_forward();
        } else if (m_operation.op_mode
                == DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR) {
            return build_norm_backward();
        } else if (m_operation.op_mode
                == DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR) {
            return build_resample_bwd_operation();
        } else if (m_operation.op_mode
                == DescriptorType_t::OPERATION_RNG_DESCRIPTOR) {
            return build_rng_operation();
        } else if (m_operation.op_mode
                == DescriptorType_t::OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR) {
            return build_paged_cache_load_op();
        } else if (m_operation.op_mode
                == DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR) {
            return build_reshape_operation();
        } else {
            set_error_and_throw_exception(&m_operation, status,
                    "CUDNN_BACKEND_OPERATION: unimplemented operation in "
                    "frontend");
        }
        CUDNN_FE_LOG_LABEL_ENDL(m_operation);
        return std::move(m_operation);
    }
}

} // namespace compat_0_x

#endif