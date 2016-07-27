#ifndef UTILS_HPP
#define UTILS_HPP

#include <stddef.h>
#include <malloc.h>

namespace mkl_dnn { namespace impl {

#define UNUSED(x) ((void)x)

template <typename T, typename P>
inline bool one_of(T val, P item) { return val == item; }
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

template <typename... Args>
inline bool any_null(Args... ptrs) { return one_of(nullptr, ptrs...); }

template<typename T>
inline void array_copy(T *dst, const T *src, size_t size) {
    for (size_t i = 0; i < size; ++i) dst[i] = src[i];
}
template<typename T>
inline bool array_cmp(const T *a1, const T *a2, size_t size) {
    for (size_t i = 0; i < size; ++i) if (a1[i] != a2[i]) return false;
    return true;
}
template<typename T, typename U>
inline void array_set(T *arr, const U& val, size_t size) {
    for (size_t i = 0; i < size; ++i) arr[i] = val;
}

inline void* malloc(size_t size, int alignment) {
    UNUSED(alignment);
    return ::malloc(size);
}
inline void free(void* p) { ::free(p); }

struct c_compatible {
    enum { default_alignment = 64 };
    static void* operator new(size_t sz) {
        return malloc(sz, default_alignment);
    }
    static void* operator new(size_t sz, void* p) { UNUSED(sz); return p; }
    static void* operator new[](size_t sz) {
        return malloc(sz, default_alignment);
    }
    static void operator delete(void* p) { free(p); }
    static void operator delete[](void* p) { free(p); }
};

inline void yield_thread() { }

}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
