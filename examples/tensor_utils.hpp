/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef TENSOR_UTILS_HPP
#define TENSOR_UTILS_HPP

using namespace dnnl;
using namespace std;

using tag = memory::format_tag;
using dt = memory::data_type;

dnnl::engine global_engine;
dnnl::stream global_engine_stream;

template <typename T>
memory::data_type type2dt() {
    if (std::is_same<T, float>::value) {
        return memory::data_type::f32;
    } else if (std::is_same<T, double>::value) {
        return memory::data_type::f64;
    } else if (std::is_same<T, int32_t>::value) {
        return memory::data_type::s32;
    } else if (std::is_same<T, int8_t>::value) {
        return memory::data_type::s8;
    } else if (std::is_same<T, uint8_t>::value) {
        return memory::data_type::u8;
    } else {
        throw std::invalid_argument("Unsupported data type");
    }
}

tag dims2tag(const memory::dims &dims) {
    switch (dims.size()) {
        case 1: return tag::a;
        case 2: return tag::ab;
        case 3: return tag::abc;
        case 4: return tag::abcd;
        default: throw std::invalid_argument("Unsupported number of dims");
    }
}

struct tensor {
    memory::desc md_;
    memory mem_;
    tensor() : md_({}, dt::undef, tag::undef), mem_() {}

    tensor(std::vector<float> vec, const memory::dims dims) {
        tag out_tag = dims2tag(dims);
        memory::data_type dtype = type2dt<float>();

        md_ = memory::desc(dims, dtype, out_tag);
        mem_ = memory(md_, global_engine);

        write_to_dnnl_memory((void *)vec.data(), mem_);
    }

    tensor(std::vector<float> vec, const memory::dims dims, tag out_tag) {
        memory::data_type dtype = type2dt<float>();

        md_ = memory::desc(dims, dtype, out_tag);
        mem_ = memory(md_, global_engine);

        write_to_dnnl_memory((void *)vec.data(), mem_);
    }
};

tensor cast(tensor in, memory::data_type dt, tag out_tag = tag::undef) {
    tensor out;
    if (out_tag == tag::undef)
        out_tag = dims2tag(in.md_.get_dims());
    out.md_ = memory::desc(in.md_.get_dims(), dt, out_tag);
    out.mem_ = memory(out.md_, global_engine);

    primitive_attr reorder_attr;
    auto reorder_pd = reorder::primitive_desc(
            global_engine, in.md_, global_engine, out.md_, reorder_attr);
    auto reorder_prim = reorder(reorder_pd);

    reorder_prim.execute(global_engine_stream, in.mem_, out.mem_);
    global_engine_stream.wait();

    return out;
}


template <typename ty>
std::vector<ty> tensor2vec(tensor in) {
    if (in.md_.get_dims().empty()) { return {}; }

    memory::data_type dst_dt = type2dt<ty>();
    tensor converted_tensor = cast(in, dst_dt);

    std::vector<ty> out;
    out.resize(product(in.md_.get_dims()), 0);
    read_from_dnnl_memory(out.data(), converted_tensor.mem_);

    return out;
}

tensor reshape(tensor in, const memory::dims dims) {
    auto mem = tensor2vec<float>(in);
    tensor out = tensor(mem, dims);
    return out;
}

tensor eye(const memory::dims &dims) {
    std::vector<float> vec;
    int dim0 = dims[0], dim1 = dims[1], rows = dims[2], cols = dims[3];
    vec.resize(dim0 * dim1 * rows * cols, 0.0f);
    int square_size = std::min(rows, cols);
    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            for (int k = 0; k < square_size; ++k) {
                int index = i * dim1 * rows * cols + j * rows * cols + k * cols
                        + k;
                vec[index] = 1.0f;
            }
        }
    }
    return cast(tensor(vec, dims), dt::f16);
}

tensor rand(const memory::dims &dims) {
    std::vector<float> vec(dims[0] * dims[1] * dims[2] * dims[3]);
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 9);

    for (int i = 0; i < dims[0]; ++i)
        for (int j = 0; j < dims[1]; ++j)
            for (int k = 0; k < dims[2]; ++k)
                for (int l = 0; l < dims[3]; ++l)
                    vec[i * dims[1] * dims[2] * dims[3] + j * dims[2] * dims[3]
                            + k * dims[3] + l]
                            = dis(gen);

    return cast(tensor(vec, dims), dt::f16);
}

tensor zeros(const memory::dims &dims, tag out_tag = tag::undef) {
    std::vector<float> vec;
    int dim0 = dims[0], dim1 = dims[1], rows = dims[2], cols = dims[3];
    vec.resize(dim0 * dim1 * rows * cols, 0.0f);
    if (out_tag == tag::undef) 
        return cast(tensor(vec, dims), dt::f16);
    else 
        return cast(tensor(vec, dims, out_tag), dt::f16, out_tag);
}

tensor iota(const int numel) {
    std::vector<float> vec(numel);
    std::iota(vec.begin(), vec.end(), 0);
    return cast(tensor(vec, {1, 1, 1, numel}), dt::f32);
}

tensor rand(const int numel) {
    std::seed_seq seed {12345};
    std::mt19937 generator(seed);
    std::vector<float> vec(numel);
    std::iota(vec.begin(), vec.end(), 0);
    std::shuffle(vec.begin(), vec.end(), generator);
    return cast(tensor(vec, {1, 1, 1, numel}), dt::f32);
}

tensor concat(tensor a, tensor b) { // down 4th dimension
    std::vector<float> a_vec = tensor2vec<float>(a);
    std::vector<float> b_vec = tensor2vec<float>(b);
    std::vector<float> c_vec(a_vec.size() + b_vec.size());
    for (size_t i = 0; i < a_vec.size(); i++) {
        c_vec[i] = a_vec[i];
    }
    for (size_t i = 0; i < b_vec.size(); i++) {
        c_vec[i + a_vec.size()] = b_vec[i];
    }
    return tensor(c_vec, {1, 1, 1, (int)c_vec.size()});
}

const std::string dt2str(memory::data_type dtype) {
    switch (dtype) {
        case dt::f32: return "f32";
        case dt::f16: return "f16";
        case dt::s32: return "s32";
        default: throw std::invalid_argument("Unsupported data type");
    }
}

void render(const tensor &t, std::ostream &out, const bool terse) {
    std::vector<float> vec = tensor2vec<float>(t);
    auto dims = t.md_.get_dims();

    if (!terse) {
        for (auto d : dims)
            out << d << " ";
        out << "[" << dt2str(t.md_.get_data_type()) << "]\n";
    }

    while (dims.size() < 4)
        dims.insert(dims.begin(), 1);
    int dim0 = dims[0], dim1 = dims[1], cols = dims[2], rows = dims[3];

    std::map<float, int> value_map;
    std::vector<float> unique_values(vec.begin(), vec.end());
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()),
            unique_values.end());
    for (int i = 0; i < (int)unique_values.size(); ++i)
        value_map[unique_values[i]] = i;

    if (!terse) {
        out << "Legend:\n";
        for (const auto &it : value_map)
            out << std::setw(3) << it.second << ": " << it.first << "\n";
        out << "\n";
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            out << "(" << i << "," << j << ")\n";
            for (int k = 0; k < rows; ++k) {
                for (int l = 0; l < cols; ++l) {
                    int index = i * dim1 * rows * cols + j * rows * cols
                            + l * rows + k;
                    out << std::setw(3) << value_map[vec[index]];
                }
                out << "\n";
            }
            out << "\n";
        }
    }
}

void show(const tensor &t, const bool terse = false) {
    render(t, std::cout, terse);
}

void write(const tensor &t, const std::string &file_path) {
    std::ofstream file(file_path);
    if (!file) throw std::runtime_error("Could not open file: " + file_path);
    render(t, file, false);
}

std::vector<float> transpose_vector(
        const std::vector<float> &data, const std::vector<long int> &dims) {
    const int A = dims[0], B = dims[1], rows = dims[2], cols = dims[3];
    std::vector<float> transposed(data.size());
    for (int a = 0; a < A; ++a) {
        for (int b = 0; b < B; ++b) {
            for (int col = 0; col < cols; ++col) {
                for (int row = 0; row < rows; ++row) {
                    int src_index = a * (B * rows * cols) + b * (rows * cols)
                            + col * rows + row;
                    int dst_index = a * (B * rows * cols) + b * (rows * cols)
                            + row * cols + col;
                    transposed[dst_index] = data[src_index];
                }
            }
        }
    }
    return transposed;
}

tensor transpose_backing(tensor &t) {
    auto vec = tensor2vec<float>(t);
    std::vector<long int> dims = t.md_.get_dims();
    memory::data_type dt = t.md_.get_data_type();
    auto transposed_vec = transpose_vector(vec, dims);
    return cast(tensor(transposed_vec, dims), dt);
}

tensor transpose(const tensor &t) {
    memory::data_type dt = t.md_.get_data_type();
    std::vector<float> mem = tensor2vec<float>(t);
    std::vector<long int> dims = t.md_.get_dims();

    std::vector<float> transposed_mem = transpose_vector(mem, dims);
    std::vector<long int> transposed_dims({dims[0], dims[1], dims[3], dims[2]});

    memory::dims memory_dims(transposed_dims.begin(), transposed_dims.end());
    return cast(tensor(transposed_mem, memory_dims), dt);
}

tensor read(const std::string &file_path) {
    std::ifstream file(file_path);
    if (!file) throw std::runtime_error("Could not open file: " + file_path);

    std::string line;
    std::getline(file, line);
    std::istringstream line_stream(line);

    std::vector<long int> dims;
    int dim;
    while (line_stream >> dim) {
        dims.push_back(dim);
    }

    line_stream.clear();
    std::string dtype_str;
    line_stream >> dtype_str;
    memory::data_type dtype;
    if (dtype_str == "[f32]")
        dtype = dt::f32;
    else if (dtype_str == "[f16]")
        dtype = dt::f16;
    else if (dtype_str == "[s32]")
        dtype = dt::s32;
    else
        throw std::runtime_error("Unsupported data type read from file.");

    while (std::getline(file, line) && line != "Legend:") {}

    std::map<int, float> key_to_value;
    while (std::getline(file, line) && !line.empty()) {
        std::istringstream legend_stream(line);
        int key;
        char colon;
        float value;
        legend_stream >> key >> colon >> value;
        key_to_value[key] = value;
    }

    std::vector<float> data;
    while (std::getline(file, line)) {
        if (line.empty() || std::all_of(line.begin(), line.end(), ::isspace))
            continue;
        std::istringstream data_stream(line);
        while (!data_stream.eof()) {
            int key;
            data_stream >> key;
            if (data_stream.fail()) break;
            data.push_back(key_to_value[key]);
        }
    }
    if (data.size() < (size_t)(dims[0] * dims[1] * dims[2] * dims[3])) {
        fprintf(stderr, "error: insufficient data within %s\n",
                file_path.c_str());
        exit(1);
    }

    std::vector<float> transposed_data = transpose_vector(data, dims);

    memory::dims memory_dims(dims.begin(), dims.end());
    tensor read_tensor(transposed_data, memory_dims);
    return cast(read_tensor, dtype);
}

tensor repl(const tensor &input, int times) {
    memory::dims new_dims = input.md_.get_dims();
    if (new_dims.empty() || times <= 0)
        throw std::invalid_argument("Invalid input or times.");
    new_dims[0] *= times;
    std::vector<float> input_data = tensor2vec<float>(input), replicated_data;
    replicated_data.reserve(input_data.size() * times);
    for (int i = 0; i < times; ++i)
        replicated_data.insert(
                replicated_data.end(), input_data.begin(), input_data.end());
    return cast(tensor(replicated_data, new_dims), input.md_.get_data_type());
}

std::chrono::high_resolution_clock::time_point global_tic_time;

void tic() {
    global_tic_time = std::chrono::high_resolution_clock::now();
}

double toc() {
    auto toc_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = toc_time - global_tic_time;
    // std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    return elapsed.count();
}

bool same(const tensor &a, const tensor &b) {
    if (a.md_.get_dims() != b.md_.get_dims()) return false;
    std::vector<float> data_a = tensor2vec<float>(a);
    std::vector<float> data_b = tensor2vec<float>(b);
    return data_a == data_b;
}

#endif
