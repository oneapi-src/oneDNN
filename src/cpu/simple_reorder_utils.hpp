#pragma once

#include <immintrin.h>
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace policy {

template <int ...>
struct variant {
};

// Policy space
//
// Data policy
//
struct data_type_policy;
//
// Instruction Policy for choosing different instruction sets
//
enum instruction_set {
  scalar = 0, sse = 128, avx = 256, avx2 = 257, avx512 = 512
};

template <instruction_set>
class inst_policy;

template <typename T, int instruction_set, int width>
class ext_reg;

template <> class ext_reg<float, avx, 8> {
public:
  typedef __m256 type;
};

template <> class ext_reg<float, avx, 4> {
public:
  typedef __m128 type;
};

template <> class inst_policy<avx> {
public:
  template <int reg_width, typename T = float> using reg_type
    = typename ext_reg<T, avx, reg_width>::type;

  static constexpr auto max_width = 256 / 8;

  static inline reg_type<8> unpackhi(reg_type<8> a, reg_type<8> b) {
    return _mm256_unpackhi_ps(a, b);
  }

  static inline reg_type<8> unpacklo(reg_type<8> a, reg_type<8> b) {
    return _mm256_unpacklo_ps(a, b);
  }

  template <int imm8>
  static inline reg_type<8> shuffle(reg_type<8> a, reg_type<8> b) {
    return _mm256_shuffle_ps(a, b, imm8);
  }

  template <int imm8>
  static inline reg_type<8> permute2f128(reg_type<8> a, reg_type<8> b) {
    return _mm256_permute2f128_ps(a, b, imm8);
  }

  // load<8>
  template <int reg_w>
  static inline reg_type<reg_w> loadu(const float *adrs);
  static inline __m256i mask(int l) {
    int32_t mask_src[] = { -1, -1, -1, -1, -1, -1, -1, -1,
       0, 0, 0, 0, 0, 0, 0, 0};
    return _mm256_loadu_si256((const __m256i *)(mask_src + 8 -l));
  }

  // maskload<8>
  template <int reg_w>
  static inline reg_type<reg_w> maskload(const float *adrs, int l);
  // storeu<8>
  template <int reg_w>
  static inline void storeu(float *, reg_type<reg_w>);

  // maskload<8>
  template <int reg_w>
  static inline void maskstore(float *adrs, int l, reg_type<reg_w> a);
  template <int reg_width>
  static inline reg_type<reg_width> broadcast(float scalar);
};

template <>
inline inst_policy<avx>::reg_type<8>
inst_policy<avx>::loadu<8> (const float *adrs) {
  return _mm256_loadu_ps(adrs);
}
template <>
inline inst_policy<avx>::reg_type<8>
inst_policy<avx>::maskload<8> (const float *adrs, int l) {
  return _mm256_maskload_ps(adrs, mask(l));
}
template <>
inline void inst_policy<avx>::storeu<8>(float *__p, reg_type<8> __a) {
  _mm256_storeu_ps(__p, __a);
}
template <>
inline void inst_policy<avx>::maskstore<8> (float *adrs, int l, reg_type<8> a) {
  return _mm256_maskstore_ps(adrs, mask(l), a);
}
template <> inline inst_policy<avx>::reg_type<8>
inst_policy<avx>::broadcast<8>(float scalar) {
  return _mm256_set1_ps(scalar);
}

template <> class inst_policy<avx2> : public inst_policy<avx> {};
template <> class inst_policy<avx512> : public inst_policy<avx2> {
public:
  constexpr static int max_width = 512 / 8;
};

//
// IO policy for explain data
//
template <typename data_type_policy,
         class inst_policy>
struct coalesced_io {
  typedef data_type_policy DP;

  typedef typename DP::i_type i_type;
  typedef typename DP::o_type o_type;

  typedef inst_policy IP;
  template <int read_w>
    using reg_type = typename IP::template reg_type<read_w>;

public:
  template <int read_w, int tail = read_w>
  static inline reg_type<read_w> read(
      const i_type *in, int64_t row, int64_t stride) {
    auto *start = in + row * stride;
    if (tail == read_w)
      return IP::template loadu<read_w>(start);
    else
      return IP::template maskload<read_w>(start, tail);
  }

  template <int write_w, int tail = write_w>
  static inline void write(o_type *o,
      reg_type<write_w> a, int64_t row, int64_t stride) {
    auto *start = o + row * stride;
    if (tail == write_w)
      IP::template storeu<write_w>(start, a);
    else
      IP::template maskstore<write_w>(start, tail, a);
  }

  template <int width>
  static inline reg_type<width> zero() {
    return IP::template broadcast<8>(0.);
  }
};

struct float_to_float {
  typedef float i_type;
  typedef float o_type;
};
}

#if defined(__AVX__)
constexpr auto selector = policy::avx;
#elif defined(__AVX2__)
constexpr auto selector = policy::avx2;
#elif defined(__AVX512__)
constexpr auto selector = policy::avx512;
#else
constexpr auto selector = policy::scalar;
#endif

template <
  typename data_type_policy,
  class inst_policy,
  template <typename, typename> class io_policy
>
struct transpose_ker : public inst_policy,
  io_policy<data_type_policy, inst_policy> {
public:
  typedef data_type_policy DP;
  typedef typename DP::i_type i_type;
  typedef typename DP::o_type o_type;
  typedef inst_policy IP;
  typedef io_policy<data_type_policy, inst_policy> IO;
  template <int read_w>
  using reg_type = typename IP::template reg_type<read_w>;
  static_assert(IP::max_width >= 256/8, "requires vector instruction lane over 8");

  /* subblk = 4, and max reg width is over 4 */
  template <bool order_keep, int tail>
  static inline void __doit(
      o_type *out, const i_type *in, dim_t m, int blksize,
      policy::variant<4>) {
    if (tail == 0)
      return;
    assert("!unimplemented");
  }

  /* subblk = 16, and max reg width is over 16 */
  template <bool order_keep, int tail>
  static inline void __doit(
      o_type *out, const i_type *in, dim_t m, int blksize,
      policy::variant<16>) {
    if (tail == 0)
      return;
    assert("!unimplemented");
  }

  /* subblk = 8, and max reg width is over 8 */
  template <bool order_keep, int tail>
  static inline void __doit(
      o_type *out, const i_type *in, dim_t m, int blksize,
      policy::variant<8>) {
    if (tail == 0)
      return;

    reg_type<8> row0, row1, row2, row3, row4, row5, row6, row7;

    if (order_keep) {
      row0 = IO::template read<8, tail>(in, 0, m);
      row1 = IO::template read<8, tail>(in, 1, m);
      row2 = IO::template read<8, tail>(in, 2, m);
      row3 = IO::template read<8, tail>(in, 3, m);
      row4 = IO::template read<8, tail>(in, 4, m);
      row5 = IO::template read<8, tail>(in, 5, m);
      row6 = IO::template read<8, tail>(in, 6, m);
      row7 = IO::template read<8, tail>(in, 7, m);
    } else {
      switch(tail) {
        case 8:
          row0 = IO::template read<8>(in, 0, blksize);
          row1 = IO::template read<8>(in, 1, blksize);
          row2 = IO::template read<8>(in, 2, blksize);
          row3 = IO::template read<8>(in, 3, blksize);
          row4 = IO::template read<8>(in, 4, blksize);
          row5 = IO::template read<8>(in, 5, blksize);
          row6 = IO::template read<8>(in, 6, blksize);
          row7 = IO::template read<8>(in, 7, blksize);
          break;
        case 7:
          row0 = IO::template read<8>(in, 0, blksize);
          row1 = IO::template read<8>(in, 1, blksize);
          row2 = IO::template read<8>(in, 2, blksize);
          row3 = IO::template read<8>(in, 3, blksize);
          row4 = IO::template read<8>(in, 4, blksize);
          row5 = IO::template read<8>(in, 5, blksize);
          row6 = IO::template read<8>(in, 6, blksize);
          break;
        case 6:
          row0 = IO::template read<8>(in, 0, blksize);
          row1 = IO::template read<8>(in, 1, blksize);
          row2 = IO::template read<8>(in, 2, blksize);
          row3 = IO::template read<8>(in, 3, blksize);
          row4 = IO::template read<8>(in, 4, blksize);
          row5 = IO::template read<8>(in, 5, blksize);
          break;
        case 5:
          row0 = IO::template read<8>(in, 0, blksize);
          row1 = IO::template read<8>(in, 1, blksize);
          row2 = IO::template read<8>(in, 2, blksize);
          row3 = IO::template read<8>(in, 3, blksize);
          row4 = IO::template read<8>(in, 4, blksize);
          break;
        case 4:
          row0 = IO::template read<8>(in, 0, blksize);
          row1 = IO::template read<8>(in, 1, blksize);
          row2 = IO::template read<8>(in, 2, blksize);
          row3 = IO::template read<8>(in, 3, blksize);
          break;
        case 3:
          row0 = IO::template read<8>(in, 0, blksize);
          row1 = IO::template read<8>(in, 1, blksize);
          row2 = IO::template read<8>(in, 2, blksize);
          break;
        case 2:
          row0 = IO::template read<8>(in, 0, blksize);
          row1 = IO::template read<8>(in, 1, blksize);
          break;
        case 1:
          row0 = IO::template read<8>(in, 0, blksize);
        default:
          break;
      }
    }

    auto __t0 = IP::unpacklo(row0, row1);
    auto __t1 = IP::unpackhi(row0, row1);
    auto __t2 = IP::unpacklo(row2, row3);
    auto __t3 = IP::unpackhi(row2, row3);
    auto __t4 = IP::unpacklo(row4, row5);
    auto __t5 = IP::unpackhi(row4, row5);
    auto __t6 = IP::unpacklo(row6, row7);
    auto __t7 = IP::unpackhi(row6, row7);

    // Shuffle 1
    auto _tt0 = IP::template shuffle<_MM_SHUFFLE(1, 0, 1, 0)>(__t0, __t2);
    auto _tt1 = IP::template shuffle<_MM_SHUFFLE(3, 2, 3, 2)>(__t0, __t2);
    auto _tt2 = IP::template shuffle<_MM_SHUFFLE(1, 0, 1, 0)>(__t1, __t3);
    auto _tt3 = IP::template shuffle<_MM_SHUFFLE(3, 2, 3, 2)>(__t1, __t3);
    auto _tt4 = IP::template shuffle<_MM_SHUFFLE(1, 0, 1, 0)>(__t4, __t6);
    auto _tt5 = IP::template shuffle<_MM_SHUFFLE(3, 2, 3, 2)>(__t4, __t6);
    auto _tt6 = IP::template shuffle<_MM_SHUFFLE(1, 0, 1, 0)>(__t5, __t7);
    auto _tt7 = IP::template shuffle<_MM_SHUFFLE(3, 2, 3, 2)>(__t5, __t7);

    // shuffle 2
    auto new0 = IP::template permute2f128<0x20>(_tt0, _tt4);
    auto new1 = IP::template permute2f128<0x20>(_tt1, _tt5);
    auto new2 = IP::template permute2f128<0x20>(_tt2, _tt6);
    auto new3 = IP::template permute2f128<0x20>(_tt3, _tt7);
    auto new4 = IP::template permute2f128<0x31>(_tt0, _tt4);
    auto new5 = IP::template permute2f128<0x31>(_tt1, _tt5);
    auto new6 = IP::template permute2f128<0x31>(_tt2, _tt6);
    auto new7 = IP::template permute2f128<0x31>(_tt3, _tt7);

    if (order_keep) {
      switch(tail) {
        case 8:
          IO::template write<8>(out, new0, 0, blksize);
          IO::template write<8>(out, new1, 1, blksize);
          IO::template write<8>(out, new2, 2, blksize);
          IO::template write<8>(out, new3, 3, blksize);
          IO::template write<8>(out, new4, 4, blksize);
          IO::template write<8>(out, new5, 5, blksize);
          IO::template write<8>(out, new6, 6, blksize);
          IO::template write<8>(out, new7, 7, blksize);
          break;
        case 7:
          IO::template write<8>(out, new0, 0, blksize);
          IO::template write<8>(out, new1, 1, blksize);
          IO::template write<8>(out, new2, 2, blksize);
          IO::template write<8>(out, new3, 3, blksize);
          IO::template write<8>(out, new4, 4, blksize);
          IO::template write<8>(out, new5, 5, blksize);
          IO::template write<8>(out, new6, 6, blksize);
          break;
        case 6:
          IO::template write<8>(out, new0, 0, blksize);
          IO::template write<8>(out, new1, 1, blksize);
          IO::template write<8>(out, new2, 2, blksize);
          IO::template write<8>(out, new3, 3, blksize);
          IO::template write<8>(out, new4, 4, blksize);
          IO::template write<8>(out, new5, 5, blksize);
          break;
        case 5:
          IO::template write<8>(out, new0, 0, blksize);
          IO::template write<8>(out, new1, 1, blksize);
          IO::template write<8>(out, new2, 2, blksize);
          IO::template write<8>(out, new3, 3, blksize);
          IO::template write<8>(out, new4, 4, blksize);
          break;
        case 4:
          IO::template write<8>(out, new0, 0, blksize);
          IO::template write<8>(out, new1, 1, blksize);
          IO::template write<8>(out, new2, 2, blksize);
          IO::template write<8>(out, new3, 3, blksize);
          break;
        case 3:
          IO::template write<8>(out, new0, 0, blksize);
          IO::template write<8>(out, new1, 1, blksize);
          IO::template write<8>(out, new2, 2, blksize);
          break;
        case 2:
          IO::template write<8>(out, new0, 0, blksize);
          IO::template write<8>(out, new1, 1, blksize);
          break;
        case 1:
          IO::template write<8>(out, new0, 0, blksize);
        default:
          break;
      }
    } else {
      IO::template write<8, tail>(out, new0, 0, m);
      IO::template write<8, tail>(out, new1, 1, m);
      IO::template write<8, tail>(out, new2, 2, m);
      IO::template write<8, tail>(out, new3, 3, m);
      IO::template write<8, tail>(out, new4, 4, m);
      IO::template write<8, tail>(out, new5, 5, m);
      IO::template write<8, tail>(out, new6, 6, m);
      IO::template write<8, tail>(out, new7, 7, m);
    }
  }

  /* subblk <= max reg width == true */
  // terminate recursive, pass to real manual kernels
  template <int subblk, bool order_keep, int tail>
  static inline void _doit(o_type *out, const i_type *in, dim_t m, int blksize,
      policy::variant<true>) {
    __doit<order_keep, tail>(out, in, m, blksize, policy::variant<subblk>());
  }

  /* subblk <= max reg width == false */
  // ! recursive template
  template <int subblk, bool order_keep, int tail>
  static inline void _doit(o_type *out, const i_type *in, dim_t m, int blksize,
      policy::variant<false>) {
    constexpr auto newblk = subblk/2;
    constexpr auto l_tail = tail < newblk ? tail : newblk;
    constexpr auto u_tail = tail < newblk ? 0 : tail - newblk;

    // favor write
    if (order_keep) {
      // out[][blksize] <--> in[][m]
      //
      // out[0][0] <--> in[0][0]
      // out[0][newblk] <--> in[newblk][0]
      // out[newblk][0] <--> in[0][newblk]
      // out[newblk][newblk] <--> in[newblk][newblk]
      //
      _doit<newblk, true, l_tail>(
          out, in, m, blksize,
          policy::variant<newblk <= IP::max_width/sizeof(i_type)>());

      _doit<newblk, true, l_tail>(
          out + newblk, in + m * newblk, m, blksize,
          policy::variant<newblk <= IP::max_width/sizeof(i_type)>());

      _doit<newblk, true, u_tail>(
          out + blksize * newblk, in + newblk, m, blksize,
          policy::variant<newblk <= IP::max_width/sizeof(i_type)>());

      _doit<newblk, true, u_tail>(
          out + blksize * newblk + newblk, in + m * newblk + newblk, m, blksize,
          policy::variant<newblk <= IP::max_width/sizeof(i_type)>());

    } else {
      // out[][m] <--> in[][blksize]
      //
      // out[0][0] <--> in[0][0]
      // out[0][newblk] <--> in[newblk][0]
      // out[newblk][0] <--> in[0][newblk]
      // out[newblk][newblk] <--> in[newblk][newblk]
      //
      _doit<newblk, false, l_tail>(
          out, in, m, blksize,
          policy::variant<newblk <= IP::max_width/sizeof(i_type)>());

      _doit<newblk, false, u_tail>(
          out + newblk, in + blksize * newblk, m, blksize,
          policy::variant<newblk <= IP::max_width/sizeof(i_type)>());

      _doit<newblk, false, l_tail>(
          out + m * newblk, in + newblk, m, blksize,
          policy::variant<newblk <= IP::max_width/sizeof(i_type)>());

      _doit<newblk, false, u_tail>(
          out + m * newblk + newblk, in + blksize * newblk + newblk, m, blksize,
          policy::variant<newblk <= IP::max_width/sizeof(i_type)>());
    }
  }

  template <int blksize, bool order_keep, int tail>
  static inline void doit(
      o_type *out,
      const i_type *in,
      dim_t stride) {
    // subblk == blksize at first
    _doit<blksize, order_keep, tail>(out, in, stride, blksize,
        policy::variant<blksize <= IP::max_width/sizeof(i_type)>());
  }

  // cover blksize 4, 8, 16, case 1 ... 4
  template <int blksize, bool order_keep>
  static inline void doit(
      o_type *out,
      const i_type *in,
      dim_t stride,
      dim_t tail,
      policy::variant<1, 4>) {
    switch (tail) {
    case 1:
      doit<blksize, order_keep, 1>(out, in, stride);
      break;
    case 2:
      doit<blksize, order_keep, 2>(out, in, stride);
      break;
    case 3:
      doit<blksize, order_keep, 3>(out, in, stride);
      break;
    case 4:
      doit<blksize, order_keep, 4>(out, in, stride);
      break;
    }
  }

  template <int blksize, bool order_keep>
  static inline void doit(
      o_type *out,
      const i_type *in,
      dim_t stride,
      dim_t tail,
      policy::variant<1, 8>) {
    switch (tail) {
    case 1 ... 4:
      doit<blksize, order_keep>(
          out, in, stride, tail, policy::variant<1, 4>());
      break;
    case 5:
      doit<blksize, order_keep, 5>(out, in, stride);
      break;
    case 6:
      doit<blksize, order_keep, 6>(out, in, stride);
      break;
    case 7:
      doit<blksize, order_keep, 7>(out, in, stride);
      break;
    case 8:
      doit<blksize, order_keep, 8>(out, in, stride);
      break;
    }
  }

  template <int blksize, bool order_keep>
  static inline void doit(
      o_type *out,
      const i_type *in,
      dim_t stride,
      dim_t tail,
      policy::variant<1, 16>) {
    switch(tail) {
    case 1 ... 8:
      doit<blksize, order_keep>(
          out, in, stride, tail, policy::variant<1, 8>());
      break;
    case 9:
      doit<blksize, order_keep, 9>(out, in, stride);
      break;
    case 10:
      doit<blksize, order_keep, 10>(out, in, stride);
      break;
    case 11:
      doit<blksize, order_keep, 11>(out, in, stride);
      break;
    case 12:
      doit<blksize, order_keep, 12>(out, in, stride);
      break;
    case 13:
      doit<blksize, order_keep, 13>(out, in, stride);
      break;
    case 14:
      doit<blksize, order_keep, 14>(out, in, stride);
      break;
    case 15:
      doit<blksize, order_keep, 15>(out, in, stride);
      break;
    case 16:
      doit<blksize, order_keep, 16>(out, in, stride);
      break;
    }
  }

  // Interface for switching to tail cases
  template <int blksize, bool order_keep>
  static inline void doit(
      o_type *out,
      const i_type *in,
      dim_t stride,
      dim_t tail) {
      doit<blksize, order_keep>(
          out, in, stride, tail, policy::variant<1, blksize> ());
  }
};

/*
typedef transpose_ker<
        policy::float_to_float,
        policy::inst_policy<selector>,
        policy::io_policy> f32_simd_blocking_transpose;
*/

} // cpu
}
}
