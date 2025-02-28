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
#include "test_utils.hpp"

using dnnl::memory;
using mdt = memory::data_type;

memory::dim product(const std::vector<int64_t> &dims) {
    return dims.empty() ? 0
                        : std::accumulate(dims.begin(), dims.end(),
                                (memory::dim)1, std::multiplies<memory::dim>());
}

std::random_device &get_random_device() {
    static std::random_device rd;
    return rd;
}

std::mt19937 &get_generator() {
    static std::mt19937 generator(get_random_device()());
    return generator;
}

// this is changed from the fill_random() function in matmul_perf.cpp.
void fill_random(std::vector<float> &out, const memory::desc &desc,
        float minval, float maxval) {
    static std::vector<float> random_data_f;
    constexpr memory::dim nrand = 1037;

    if (random_data_f.empty()) {
        std::uniform_real_distribution<float> dist_f(minval, maxval);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(get_generator());
    }

    auto elems = product(desc.get_dims());
    for (memory::dim i = 0; i < elems; i += nrand) {
        size_t chunk = std::min(nrand, elems - i);
        std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    }
}

void fill_random_scales(std::vector<float> &out, const memory::desc &desc) {
    static std::vector<float> random_data_f;
    constexpr memory::dim nrand = 1037;

    if (random_data_f.empty()) {
        std::uniform_int_distribution<int> dist_f(-16, 16);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f) {
            auto value = dist_f(get_generator()) * 0.125f;
            if (value == 0.f) value = dist_f(get_generator()) * 0.125f;
            d = value;
        }
    }

    auto elems = product(desc.get_dims());
    for (memory::dim i = 0; i < elems; i += nrand) {
        size_t chunk = std::min(nrand, elems - i);
        std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    }
}

void print_mem(const dnnl::memory &mem, const std::string &name) {
    auto eng = mem.get_engine();
    dnnl::stream s(eng);
    s.wait();
    auto desc = mem.get_desc();
    auto dims = desc.get_dims();
    auto strides = desc.get_strides();

    size_t ndims = dims.size();
    size_t lastdim = ndims - 1;

    printf("%sbegin : ", name.c_str());
    printf("dims : [");
    for (auto d : dims) {
        printf("%6ld ", (long)d);
    }
    printf("]  strides : [");
    for (auto s : strides) {
        printf("%6ld ", (long)s);
    }

    if (mem.get_desc().get_data_type() == dnnl_bf16) { printf("bf16\n"); }
    void *mapped_ptr_ = (void *)mem.map_data();
    printf("]\ni:");
    for (int i = 0; i < dims[lastdim]; i++) {
        switch ((int)desc.get_data_type()) {
            case dnnl_u4:
            case dnnl_s4: printf("%4d", i); break;
            case dnnl_u8:
            case dnnl_s8: printf("%4d", i); break;
            case dnnl_f32:
            case dnnl_bf16:
            case dnnl_f16: printf("%9d", i); break;
        }
    }
    printf("\n-----\n");

    switch ((int)desc.get_data_type()) {
        case dnnl_u4:
        case dnnl_s4: {
            char *mapped_ptr = (char *)mapped_ptr_;

            std::vector<size_t> idxs(ndims, 0);
            while (true) {
                size_t offset = 0;
                for (size_t i = 0; i < ndims; ++i) {
                    offset += idxs[i] * strides[i];
                }
                offset /= 2;

                const bool odd_lastdim = idxs[lastdim] % 2;
                bool is_odd = odd_lastdim;
                if (ndims > 1 && strides[lastdim] != 1) {
                    // assumes last 2 dims transposed, TODO: arbitrary continuous dim?
                    const bool odd_2lastdim = idxs[lastdim - 1] % 2;
                    is_odd = odd_2lastdim;
                }
                int bits;
                if (is_odd) {
                    bits = (mapped_ptr[offset] & 0xf0) >> 4;
                } else {
                    bits = (mapped_ptr[offset] & 0x0f);
                }
                if (desc.get_data_type() == dnnl_s4) {
                    int sign = (bits & 0x08) ? -1 : 1;
                    if (sign == -1) {
                        bits = (bits & 0x07) - 8;
                    } else {
                        bits = (bits & 0x07);
                    }
                }
                printf("%4d", bits);

                int d = lastdim;
                while (d >= 0) {
                    if (++idxs[d] < dims[d]) {
                        break;
                    } else {
                        idxs[d--] = 0;
                        printf("\n");
                    }
                }
                if (d < 0) { break; }
            }
        } break;

        case dnnl_u8:
        case dnnl_s8: {
            char *mapped_ptr = (char *)mapped_ptr_;

            std::vector<size_t> idxs(ndims, 0);
            while (true) {
                size_t offset = 0;
                for (size_t i = 0; i < ndims; ++i) {
                    offset += idxs[i] * strides[i];
                }
                printf("%4d", mapped_ptr[offset]);

                int d = lastdim;
                while (d >= 0) {
                    if (++idxs[d] < dims[d]) {
                        break;
                    } else {
                        idxs[d--] = 0;
                        printf("\n");
                    }
                }
                if (d < 0) { break; }
            }
        } break;
        case dnnl_bf16: {
            using dnnl::impl::bfloat16_t;
            bfloat16_t *mapped_ptr = (bfloat16_t *)mapped_ptr_;

            std::vector<size_t> idxs(ndims, 0);
            while (true) {
                size_t offset = 0;
                for (size_t i = 0; i < ndims; ++i) {
                    offset += idxs[i] * strides[i];
                }
                printf("%+9.3f", (float)(mapped_ptr[offset]));

                int d = lastdim;
                while (d >= 0) {
                    if (++idxs[d] < dims[d]) {
                        break;
                    } else {
                        idxs[d--] = 0;
                        printf("\n");
                    }
                }
                if (d < 0) { break; }
            }
        } break;
        case dnnl_f16: {
            using dnnl::impl::float16_t;
            float16_t *mapped_ptr = (float16_t *)mapped_ptr_;

            std::vector<size_t> idxs(ndims, 0);
            while (true) {
                //               // uncomment to enable printing dim1 idxs
                //               if(idxs[lastdim] == 0) {
                //                   printf("(");
                //                   for(size_t i=0; i < ndims-1; ++i) {
                //                       printf("%3d%s ", idxs[i], (i < ndims-2) ? "," : "");
                //                   }
                //                   printf("): ");
                //               }

                size_t offset = 0;
                for (size_t i = 0; i < ndims; ++i) {
                    offset += idxs[i] * strides[i];
                }
                printf("%+9.3f", (mapped_ptr[offset].f()));

                int d = lastdim;
                while (d >= 0) {
                    if (++idxs[d] < dims[d]) {
                        break;
                    } else {
                        idxs[d--] = 0;
                        printf("\n");
                    }
                }
                if (d < 0) { break; }
            }
        } break;
        case dnnl_f32: {
            float *mapped_ptr = (float *)mapped_ptr_;

            std::vector<size_t> idxs(ndims, 0);
            while (true) {
                size_t offset = 0;
                for (size_t i = 0; i < ndims; ++i) {
                    offset += idxs[i] * strides[i];
                }
                printf("%+9.3f", (mapped_ptr[offset]));

                int d = lastdim;
                while (d >= 0) {
                    if (++idxs[d] < dims[d]) {
                        break;
                    } else {
                        idxs[d--] = 0;
                        printf("\n");
                    }
                }
                if (d < 0) { break; }
            }
        } break;
        default: throw std::runtime_error("Not supported");
    }
    mem.unmap_data(mapped_ptr_);
    printf("%send\n", name.c_str());
}

void transpose(const dnnl::engine &eng, memory &out, memory &in) {
    dnnl::stream s(eng);

    void *ptr2 = out.map_data();
    void *ptr1 = in.map_data();

    std::memcpy(ptr2, ptr1, in.get_desc().get_size());
    in.unmap_data(ptr1);
    out.unmap_data(ptr2);
}

void transpose_strides(const dnnl::engine &eng, memory &out, memory &in) {
    dnnl::stream s(eng);

    if (out.get_desc().get_data_type() == mdt::u4
            || out.get_desc().get_data_type() == mdt::s4) {
        auto desc = in.get_desc();
        auto dims = desc.get_dims();
        auto strides = desc.get_strides();
        auto strides_t = out.get_desc().get_strides();

        char *mapped_ptr = (char *)in.map_data();
        char *mapped_ptr_t = (char *)out.map_data();

        size_t ndims = dims.size();
        assert(ndims > 1); // TODO: will fail w/ndim == 1
        size_t lastdim = ndims - 1;
        size_t n2lastdim = lastdim - 1;

        std::vector<size_t> idxs(ndims, 0);
        while (true) {
            int is_odd = idxs[lastdim] % 2;
            int is_odd_t = idxs[n2lastdim] % 2;

            size_t offset = 0;
            size_t offset_t = 0;
            for (size_t i = 0; i < ndims; ++i) {
                offset += idxs[i] * strides[i];
                offset_t += idxs[i] * strides_t[i];
            }
            offset /= 2;
            offset_t /= 2;

            auto &val = mapped_ptr[offset];
            auto &val_t = mapped_ptr_t[offset_t];

            char bits;
            if (is_odd) {
                bits = val & 0xf0;
                bits >>= 4;
            } else {
                bits = val & 0x0f;
            }
            if (is_odd_t) {
                val_t |= (bits << 4);
            } else {
                val_t |= bits;
            }

            int d = lastdim;
            while (d >= 0) {
                if (++idxs[d] < dims[d]) {
                    break;
                } else {
                    idxs[d--] = 0;
                }
            }
            if (d < 0) { break; }
        }

        in.unmap_data(mapped_ptr);
        out.unmap_data(mapped_ptr_t);
    } else {
        dnnl::reorder(in, out).execute(s, in, out);
    }
}

std::ostream &operator<<(std::ostream &ss, const quantize_type &qt) {
    switch (qt) {
        case quantize_type::no_quantization: ss << "no_quantization"; break;
        case quantize_type::per_tensor: ss << "per_tensor"; break;
        case quantize_type::per_tensor1: ss << "per_tensor1"; break;
        case quantize_type::per_tensor3: ss << "per_tensor3"; break;
        case quantize_type::per_token: ss << "per_token"; break;
        case quantize_type::per_token_with_groups:
            ss << "per_token_with_groups";
            break;
    }
    return ss;
}

std::ostream &operator<<(std::ostream &ss, const memory::data_type &dt) {
    switch (dt) {
        case mdt::f32: ss << "f32"; break;
        case mdt::s32: ss << "s32"; break;
        case mdt::f16: ss << "f16"; break;
        case mdt::s8: ss << "s8"; break;
        case mdt::u8: ss << "u8"; break;
        case mdt::s4: ss << "s4"; break;
        case mdt::u4: ss << "u4"; break;
        default: ss << "na"; break;
    }
    return ss;
}
