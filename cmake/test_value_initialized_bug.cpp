/** Expansion on ideas from Niels Dekker's large test suite for
 * boost::value_initialized.
 * \sa https://www.boost.org/doc/libs/1_60_0/libs/utility/value_init.htm
 */
#include <iostream>
#include <stdint.h>
#include <type_traits>

enum enum_type { negative_number = -1, magic_number = 42 };
struct pod_struct {
    enum_type e;
    bool b;
    char c;
    unsigned char uc;
    short s;
    int i;
    unsigned u;
    long l;
    float f;
    double d;
    long double ld;
    void *p;
};
struct pod_in_pod {
    int i;
    struct pod_struct sub_pod;
    int j;
};

typedef ::pod_struct PodStruct;
typedef ::pod_in_pod PodInPod;

bool is_value_initialized(const pod_struct &arg) {
    return arg.b == 0 && arg.e == 0 && arg.c == 0 && arg.uc == 0 && arg.s == 0
            && arg.i == 0 && arg.u == 0 && arg.l == 0 && arg.f == 0
            && arg.d == 0 && arg.p == 0;
}
bool is_value_initialized(const PodInPod &arg) {
    return arg.i == 0 && is_value_initialized(arg.sub_pod) && arg.j == 0;
}

// to mirror one dnnl idiom
namespace utils {
template <typename T>
inline typename std::remove_reference<T>::type zero() {
    auto zero = typename std::remove_reference<T>::type();
    // failure means we will use a memset here
    return zero;
}
} // namespace utils

// This particular one is more like dnnl actual usage than the examples
// in the boost::value_initialized compiler bug tests.
struct PodStaticInit {
    PodStruct data;
    PodInPod pip;

    static void pip_init(PodInPod *p) { // XXX nc++ issue?
        p->i = 0;
        auto zero = PodStruct();
        p->sub_pod = zero;
        p->j = 0;
    }
    static void data_init(PodStruct *dat) {
        auto zero = utils::zero<PodStruct &>();
        *dat = zero;
    }

    PodStaticInit() {
        pip_init(&pip);
        data_init(&data);
    }
    virtual ~PodStaticInit() {}
    size_t size() const { return sizeof(data); }
};
bool is_value_initialized(const PodStaticInit &arg) {
    bool ret = true
            && is_value_initialized(arg.data) // "simple" init is OK for nc++
            && is_value_initialized(arg.pip) // "sub_obj" init fails for nc++
            ;
    return ret;
}

// Equivalent to the dirty_stack() function from GCC Bug 33916,
// "Default constructor fails to initialize array members", reported in 2007 by
// Michael Elizabeth Chastain: http://gcc.gnu.org/bugzilla/show_bug.cgi?id=33916
// Slightly mods for nc++
static uint64_t dirty_sum = 0U;
void dirty_stack() {
    volatile unsigned char array_on_stack[sizeof(PodStaticInit) * 3U + 256U];
    for (unsigned i = 0; i < sizeof(array_on_stack); ++i) {
        array_on_stack[i] = 0x11;
    }
    for (unsigned i = 0; i < sizeof(array_on_stack); ++i) {
        dirty_sum += array_on_stack[i];
    }
}

// Returns zero when the specified object is value-initializated, and one otherwise.
// Prints a message to standard output if the value-initialization has failed.
template <class T>
unsigned failed_to_value_initialized(
        const T &object, const char *const object_name) {
    if (is_value_initialized(object)) {
        return 0u;
    } else {
        std::cout << "Note: Failed to value-initialize " << object_name << '.'
                  << std::endl;
        return 1u;
    }
}

// A macro that passed both the name and the value of the specified object to
// the function above here.

int main(int, char **) {
#define FAILED_TO_VALUE_INITIALIZE(value) \
    failed_to_value_initialized(value, #value)
    // SOME contexts fail (or maybe over-optimize) the zero-initialization.
    // Here's one case, doing things like DNNL.
    // boost::value_quantized tests many more contexts for this compiler bug.
    unsigned failures = 0U;
    dirty_stack();
    failures += FAILED_TO_VALUE_INITIALIZE(PodStaticInit());
    dirty_stack();
    PodStaticInit m_PodStaticInit[2];
    failures += FAILED_TO_VALUE_INITIALIZE(m_PodStaticInit[0]);
    failures += FAILED_TO_VALUE_INITIALIZE(m_PodStaticInit[1]);
    return failures;
}
