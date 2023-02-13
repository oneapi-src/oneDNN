/*******************************************************************************
* Copyright 2023 IBM Corporation
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
#ifndef CPU_S390X_VEC_H
#define CPU_S390X_VEC_H
#include <vecintrin.h>
namespace dnnl {
namespace impl {
namespace cpu {
namespace s390x {

constexpr int VLEN_BYTES = 16;
constexpr bool ISLASTINDEX_FAST = false;
#define aPtr(i, j) A[(j)*ldA + (i)] // map aPtr( i,j ) to array A
#define bPtr(i, j) B[(j)*ldB + (i)] // map bPtr( i,j ) to array B
#define gPtr(i, j) C[(j)*ldC + (i)] // map gPtr( i,j ) to array C

#define ALWAYS_INLINE __attribute__((always_inline))

template <typename T>
struct vec_inner_type_t {
    using Type __attribute__((vector_size(VLEN_BYTES))) = T;
};

template <typename T>
struct vec_type_t {
public:
    using Type = typename vec_inner_type_t<T>::Type;
    using ElementType = T;
    operator Type &() { return _val; }
    operator Type() const { return _val; }
    static constexpr int size() { return VLEN_BYTES / sizeof(ElementType); }
    ALWAYS_INLINE vec_type_t() { _val = Type {}; }

    ALWAYS_INLINE explicit vec_type_t(T scalar)
        : _val {vec_splats((T)scalar)} {}

    ALWAYS_INLINE vec_type_t(Type v) : _val {v} {}

    static vec_type_t<T> ALWAYS_INLINE loadu(const void *ptr) {
        return {vec_xl(0, reinterpret_cast<const ElementType *>(ptr))};
    }

    static ALWAYS_INLINE vec_type_t<T> loadLen(
            const void *ptr, uint32_t BYTE_INDEX) {
        return {vec_load_len(
                reinterpret_cast<const ElementType *>(ptr), BYTE_INDEX)};
    }

    static vec_type_t<T> ALWAYS_INLINE load_hinted(const void *ptr) {
        Type const *addr = (Type const *)ptr;
        Type y;
        y = *addr;
        return y;
    }

    void ALWAYS_INLINE store(void *ptr) const {
        vec_xst(_val, 0, reinterpret_cast<ElementType *>(ptr));
    }

    void ALWAYS_INLINE storeLen(void *ptr, uint32_t BYTE_INDEX) const {
        vec_store_len(_val, reinterpret_cast<ElementType *>(ptr), BYTE_INDEX);
    }

    ALWAYS_INLINE const Type &vec() const { return _val; }

    vec_type_t<T> &ALWAYS_INLINE operator+=(const vec_type_t<T> &other) {
        _val = _val + other._val;
        return *this;
    }

private:
    Type _val;
};

template <typename V, typename T>
vec_type_t<V> cast(const vec_type_t<T> &x) {
    using cast_type = typename vec_type_t<V>::Type;
    return vec_type_t<V> {(cast_type)(x.vec())};
}

inline vec_type_t<int32_t> multiplyAdd(vec_type_t<int16_t> va,
        vec_type_t<int16_t> vb, vec_type_t<int32_t> vc) {
    // 2 ops  2 moad
    auto a = va.vec();
    auto b = vb.vec();
    auto c = vc.vec();
    c = vec_moadd(a, b, c);
    c = vec_meadd(a, b, c);
    return vec_type_t<int32_t> {c};
}

template <typename T, bool LastIndexFast = ISLASTINDEX_FAST,
        typename ElementCAST = T>
struct matrix_ptr_t {
    matrix_ptr_t(T *a, int64_t ld) : a {a}, ld {ld} {}

    T *ptr(int64_t i, int64_t j) {

        if (LastIndexFast)
            return a + i * ld + j;
        else
            return a + j * ld + i;
    }

    T &element(int64_t i, int64_t j) {

        if (LastIndexFast)
            return a[i * ld + j];
        else
            return a[j * ld + i];
    }

    T &operator()(int64_t i, int64_t j) { return element(i, j); }

    ElementCAST element(int64_t i, int64_t j) const {
        if (LastIndexFast)
            return (ElementCAST)a[i * ld + j];
        else
            return (ElementCAST)a[j * ld + i];
    }

    ElementCAST operator()(int64_t i, int64_t j) const { return element(i, j); }

    T *a;
    int64_t ld;
};

} // namespace s390x
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
