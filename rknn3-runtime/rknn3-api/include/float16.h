/*
 * Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
 */

#ifndef __RKNN3_FLOAT16_H_
#define __RKNN3_FLOAT16_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __riscv

#include <riscv_vector.h>

typedef _Float16      float16;
static inline float   fp16_to_fp32(float16 data) { return (float)data; }
static inline float16 fp32_to_fp16(float data) { return (float16)(data); }

#else

#include <stdint.h>

#define BF16_EXP_MAX (256 - 1)  //  2^8 - 1
#define FP16_EXP_MAX (32 - 1)   //  2^5 - 1
#define FP32_EXP_MAX (256 - 1)  //  2^8 - 1
#define FP64_EXP_MAX (2048 - 1) //  2^11 - 1

#define FP16_NAN ((FP16_EXP_MAX << 10) + 1)
#define FP16_INF ((FP16_EXP_MAX << 10) + 0)
#define BF16_NAN ((BF16_EXP_MAX << 7) + 1)
#define BF16_INF ((BF16_EXP_MAX << 7) + 0)
#define FP32_NAN ((FP32_EXP_MAX << 23) + 1)
#define FP32_INF ((FP32_EXP_MAX << 23) + 0)

#define FP16_FRAC_BITS 10 // fp16 fraction (mantissa) width
#define FP32_FRAC_BITS 23 // fp32 fraction (mantissa) width

#define FP16_EXP_BIAS 15  // fp16 exponent bias
#define FP32_EXP_BIAS 127 // fp32 exponent bias

// Fraction alignment between fp16 and fp32: 23 - 10 = 13
#define FP16_FP32_FRAC_SHIFT (FP32_FRAC_BITS - FP16_FRAC_BITS)

// One past the largest 10-bit fraction: an RNE carry that overflows the fp16
// fraction field lands here (e.g. 1.999... rounds up to 2.0)
#define FP16_FRAC_CARRY (1u << FP16_FRAC_BITS)

// Mask of the 23 fp32 fraction bits
#define FP32_FRAC_MASK ((1u << FP32_FRAC_BITS) - 1)

#ifdef __GNUC__
#define PACKAGE_MARK __attribute__((packed))
#else
#define PACKAGE_MARK
#endif

struct fp16_pack
{
  uint16_t frac : 10;
  uint16_t exp : 5;
  uint16_t sign : 1;
} PACKAGE_MARK;

struct fp32_pack
{
  uint32_t frac : 23;
  uint32_t exp : 8;
  uint32_t sign : 1;
} PACKAGE_MARK;

typedef struct fp16_pack float16;

static inline float fp16_to_fp32(float16 data)
{
  float             f;
  struct fp32_pack* fp32 = (struct fp32_pack*)&f;
  struct fp16_pack* fp16 = &data;

  int32_t exp = fp16->exp;

  if (exp == FP16_EXP_MAX && fp16->frac != 0) {
    // NaN: keep the payload (consistent with numpy / IEEE 754)
    fp32->sign = fp16->sign;
    fp32->exp  = FP32_EXP_MAX;
    fp32->frac = ((uint32_t)fp16->frac) << FP16_FP32_FRAC_SHIFT;

    return f;
  }

  if (exp == FP16_EXP_MAX)
    exp = FP32_EXP_MAX;
  else if (exp == 0)
    exp = 0; // subnormal / zero: exponent stays 0
  else
    exp = (exp - FP16_EXP_BIAS) + FP32_EXP_BIAS;

  fp32->exp  = exp;
  fp32->sign = fp16->sign;
  if (fp16->exp == 0) {
    // fp16 subnormal (incl. +/-0): value = frac * 2^-24.
    // Normalize: shift frac up until bit 23 (the implicit-one position) is
    // set, which takes `shift` steps:
    //   value = (1.M) * 2^(23 - 24 - shift)
    //   => exp8 = 126 - shift and the low 23 bits of M are the fp32 fraction.
    uint32_t frac = fp16->frac;
    if (frac == 0) {
      fp32->frac = 0; // +/-0
      return f;
    }
    int32_t shift = 0;
    while ((frac & (1u << FP32_FRAC_BITS)) == 0) {
      frac <<= 1;
      shift++;
    }
    fp32->exp  = FP32_EXP_BIAS - 1 - shift;
    fp32->frac = frac & FP32_FRAC_MASK;
  } else {
    fp32->frac = ((uint32_t)fp16->frac) << FP16_FP32_FRAC_SHIFT;
  }

  return f;
}

static inline float16 fp32_to_fp16(float data)
{
  struct fp32_pack* fp32 = (struct fp32_pack*)&data;
  struct fp16_pack  fp16;

  int32_t exp = fp32->exp;

  if (fp32->exp == FP32_EXP_MAX && fp32->frac != 0) {
    // NaN: keep the top payload bits; if the payload falls entirely below
    // bit 13 the fraction would be 0, so force the lowest bit to 1 and make
    // sure the result stays a NaN instead of turning into inf.
    fp16.exp  = FP16_EXP_MAX;
    fp16.frac = fp32->frac >> FP16_FP32_FRAC_SHIFT;
    if (fp16.frac == 0)
      fp16.frac = 1;
    fp16.sign = fp32->sign;

    return fp16;
  }

  if (exp == 0) {
    // fp32 subnormal / zero: |x| < 2^-126 << 2^-24 (the smallest fp16
    // subnormal), round to +/-0.
    fp16.exp  = 0;
    fp16.frac = 0;
    fp16.sign = fp32->sign;
    return fp16;
  }

  if ((exp - FP32_EXP_BIAS) < -14) {
    // fp16 subnormal range (|x| < 2^-14): value = M24 * 2^(e-23), where M24
    // is the 24-bit significand including the implicit 1.
    //   m10 = value * 2^24 = M24 >> (-e - 1), with e = exp - 127 in
    //   [-126, -15], rounded to nearest even (RNE).
    int32_t  e      = exp - FP32_EXP_BIAS;
    uint32_t m24    = (1u << FP32_FRAC_BITS) | fp32->frac;
    int32_t  rshift = -e - 1;                  // in [14, 125]
    uint32_t m10    = 0;
    if (rshift < 25) { // rshift >= 25: |x| < 2^-25 always rounds to 0
      m10 = m24 >> rshift;
      uint32_t rem  = m24 & ((1u << rshift) - 1); // dropped bits (RNE)
      uint32_t half = 1u << (rshift - 1);
      if (rem > half || (rem == half && (m10 & 1)))
        m10 += 1;
    }
    if (m10 >= FP16_FRAC_CARRY) { // rounded up to 2^-14: smallest normal
      fp16.exp  = 1;
      fp16.frac = 0;
      fp16.sign = fp32->sign;
      return fp16;
    }
    fp16.exp  = 0;
    fp16.frac = m10;
    fp16.sign = fp32->sign;
    return fp16;
  } else if ((exp - FP32_EXP_BIAS) > 15) {
    // Overflow (|x| >= 2^16 > 65504): infinity
    fp16.exp  = FP16_EXP_MAX;
    fp16.frac = 0;
    fp16.sign = fp32->sign;
    return fp16;
  }

  // fp16 normal range: round the mantissa to nearest even (RNE)
  {
    uint32_t m10  = fp32->frac >> FP16_FP32_FRAC_SHIFT;
    uint32_t rem  = fp32->frac & ((1u << FP16_FP32_FRAC_SHIFT) - 1);
    uint32_t half = 1u << (FP16_FP32_FRAC_SHIFT - 1);
    if (rem > half || (rem == half && (m10 & 1)))
      m10 += 1;
    if (m10 >= FP16_FRAC_CARRY) { // mantissa carried (e.g. 1.999... -> 2.0)
      m10 = 0;
      exp += 1;
      if (exp - FP32_EXP_BIAS > 15) { // 65520 and above round to inf
        fp16.exp  = FP16_EXP_MAX;
        fp16.frac = 0;
        fp16.sign = fp32->sign;
        return fp16;
      }
    }
    fp16.exp  = exp - FP32_EXP_BIAS + FP16_EXP_BIAS;
    fp16.frac = m10;
    fp16.sign = fp32->sign;
  }

  return fp16;
}
#endif

#ifdef __cplusplus
}
#endif
#endif
