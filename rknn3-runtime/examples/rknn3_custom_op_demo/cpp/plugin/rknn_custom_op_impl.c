/*
 * rknn_custom_op_impl.c - Custom operator implementation for RknnCustomOpExam.
 *
 * Computes: y = clamp(x * scale + shift, min_val, max_val)
 *
 * Demonstrates get_param() with all 6 dtype tags:
 *   i  (int)    : min_val, max_val
 *   f  (float)  : scale, shift
 *   s  (string) : mode
 *   is (ints)   : strides
 *   fs (floats) : weights
 *   ss (strings): tags
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "rknn3_api.h"
#include "float16.h"
#include "rknn_custom_op_impl.h"

/* Debug logging switch. Set to 0 to silence the per-attribute / compute-trace
 * prints in compute (error prints are always emitted). Default enabled. */
#define DEBUG_MODE 1

/* ------------------------------------------------------------------ */
/*  NC1HWC2 <-> NCHW layout conversion                                */
/* ------------------------------------------------------------------ */

#define ALIGNED_SIZE(x, bytes) (((x) + (bytes - 1)) & (~(bytes - 1)))

/* NC1HWC2 (N, C1, H, W, C2) -> NCHW (N, C, H, W), element by element.
 *   cn        = C2 (channels per C1 plane)
 *   channel   = logical C (= C1 * C2)
 *   align_stride = ALIGNED_SIZE(W, 1)
 *   align_hw   = ALIGNED_SIZE(W*H, 4)  (surface alignment)
 * src/dst are fp16. */
static int NC1HWC2_fp16_to_NCHW_fp16(const float16 *src, float16 *dst, int cn, int channel,
                                     int h, int w, int align_stride, int align_hw)
{
  if (!src || !dst) return -1;
  int idx = 0;
  for (int c = 0; c < channel; ++c) {
    int            plane  = c / cn;
    const float16 *src_c  = plane * align_hw * cn + src;
    int            offset = c % cn;
    for (int cur_h = 0; cur_h < h; ++cur_h) {
      for (int cur_w = 0; cur_w < w; ++cur_w) {
        int cur_hw = cur_h * align_stride + cur_w;
        dst[idx++] = src_c[cn * cur_hw + offset];
      }
    }
  }
  return 0;
}

/* NC1HWC2 (packed) -> NCHW fp16 for a whole batch.
 *   dims[0..4] = [N, C1, H, W, C2]  (i.e. attr->shape, NC1HWC2 is 5-dim)
 *   channel    = logical C  (use attr->orig_shape[1]) */
static int nc1hwc2_to_nchw_fp16(const void *src, float16 *dst, rknn3_tensor_type dtype,
                                const uint32_t *dims, int channel)
{
  if (!src || !dst || !dims) return -1;
  uint32_t batch        = dims[0];
  uint32_t height       = dims[2];
  uint32_t width        = dims[3];
  uint32_t cn           = dims[4];
  uint32_t align_surf   = width * height == 1 ? 1 : ALIGNED_SIZE(width * height, 4);
  uint32_t align_stride = ALIGNED_SIZE(width, 1);

  uint32_t src_batch_elems = dims[1] * align_surf * dims[4];
  uint32_t dst_batch_elems = (uint32_t)channel * width * height;
  if (dtype == RKNN3_TENSOR_FLOAT16) {
    const float16 *src_ptr = (const float16 *)src;
    float16       *dst_ptr = dst;
    for (uint32_t i = 0; i < batch; ++i) {
      NC1HWC2_fp16_to_NCHW_fp16(src_ptr, dst_ptr, (int)cn, channel, (int)height, (int)width,
                                (int)align_stride, (int)align_surf);
      src_ptr += src_batch_elems;
      dst_ptr += dst_batch_elems;
    }
  } else {
    printf("[RknnCustomOpExam] unsupported input dtype %d in NC1HWC2\n", dtype);
    return RKNN3_ERR_ARGUMENT_INVALID;
  }
  return 0;
}

/* NCHW (N, C, H, W) -> NC1HWC2 (N, C1, H, W, C2), element by element. */
static int NCHW_fp16_to_NC1HWC2_fp16(const float16 *src, float16 *dst, int cn, int channel,
                                     int h, int w, int align_stride, int align_hw)
{
  if (!src || !dst) return -1;
  for (int c = 0; c < channel; ++c) {
    int      plane  = c / cn;
    int      offset = c % cn;
    float16 *dst_c  = dst + (size_t)plane * align_hw * cn;
    for (int cur_h = 0; cur_h < h; ++cur_h) {
      for (int cur_w = 0; cur_w < w; ++cur_w) {
        int cur_hw                  = cur_h * align_stride + cur_w;
        dst_c[cn * cur_hw + offset] = src[c * h * w + cur_h * w + cur_w];
      }
    }
  }
  return 0;
}

/* NCHW fp16 -> NC1HWC2 (packed) for a whole batch.
 *   dims[0..4] = [N, C1, H, W, C2]
 *   channel    = logical C */
static int nchw_fp16_to_nc1hwc2(const float16 *src, void *dst, rknn3_tensor_type dtype,
                                const uint32_t *dims, int channel)
{
  if (!src || !dst || !dims) return -1;
  uint32_t batch        = dims[0];
  uint32_t height       = dims[2];
  uint32_t width        = dims[3];
  uint32_t cn           = dims[4];
  uint32_t align_surf   = width * height == 1 ? 1 : ALIGNED_SIZE(width * height, 4);
  uint32_t align_stride = ALIGNED_SIZE(width, 1);

  uint32_t src_batch_elems = (uint32_t)channel * width * height;
  uint32_t dst_batch_elems = dims[1] * align_surf * cn;
  if (dtype == RKNN3_TENSOR_FLOAT16) {
    const float16 *src_ptr = src;
    float16       *dst_ptr = (float16 *)dst;
    for (uint32_t i = 0; i < batch; ++i) {
      NCHW_fp16_to_NC1HWC2_fp16(src_ptr, dst_ptr, (int)cn, channel, (int)height, (int)width,
                                (int)align_stride, (int)align_surf);
      src_ptr += src_batch_elems;
      dst_ptr += dst_batch_elems;
    }
  } else {
    printf("[RknnCustomOpExam] unsupported output dtype %d in NC1HWC2\n", dtype);
    return RKNN3_ERR_ARGUMENT_INVALID;
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/*  get_param helper macros                                            */
/* ------------------------------------------------------------------ */

#define GET_PARAM_INT(op_ctx, name, out_var)                              \
  do {                                                                    \
    rknn3_custom_op_attr _attr;                                           \
    int _ret = op_ctx->get_param(op_ctx, name, &_attr);                   \
    if (_ret != 0 || !_attr.data) {                                       \
      printf("[RknnCustomOpExam] get_param('%s') failed, ret=%d\n", name, _ret); \
      return -1;                                                          \
    }                                                                     \
    (out_var) = *(int32_t *)_attr.data;                                   \
  } while (0)

#define GET_PARAM_FLOAT(op_ctx, name, out_var)                            \
  do {                                                                    \
    rknn3_custom_op_attr _attr;                                           \
    int _ret = op_ctx->get_param(op_ctx, name, &_attr);                   \
    if (_ret != 0 || !_attr.data) {                                       \
      printf("[RknnCustomOpExam] get_param('%s') failed, ret=%d\n", name, _ret); \
      return -1;                                                          \
    }                                                                     \
    (out_var) = *(float *)_attr.data;                                     \
  } while (0)

#define GET_PARAM_STRING(op_ctx, name, out_buf, out_cap, out_len)         \
  do {                                                                    \
    rknn3_custom_op_attr _attr;                                           \
    int _ret = op_ctx->get_param(op_ctx, name, &_attr);                   \
    if (_ret != 0 || !_attr.data) {                                       \
      printf("[RknnCustomOpExam] get_param('%s') failed, ret=%d\n", name, _ret); \
      return -1;                                                          \
    }                                                                     \
    if (_attr.n_elems >= (out_cap)) {                                     \
      printf("[RknnCustomOpExam] get_param('%s') too long: %u >= %u\n",   \
             name, _attr.n_elems, (uint32_t)(out_cap));                   \
      return -1;                                                          \
    }                                                                     \
    (out_len) = _attr.n_elems;                                            \
    memcpy((out_buf), _attr.data, _attr.n_elems);                         \
    (out_buf)[_attr.n_elems] = '\0';                                      \
  } while (0)

/* ------------------------------------------------------------------ */
/*  Compute callback                                                  */
/* ------------------------------------------------------------------ */

int rknn_custom_op_compute(rknn3_custom_op_context *op_ctx,
                                rknn3_tensor *inputs, uint32_t n_inputs,
                                rknn3_tensor *outputs, uint32_t n_outputs)
{
#if DEBUG_MODE
  printf("[RknnCustomOpExam] compute called (n_inputs=%u, n_outputs=%u)\n", n_inputs, n_outputs);
#endif

  /* ---- Read all params via get_param (demonstrates all 6 dtypes) ---- */

  /* dtype "i": int32 scalars */
  int32_t min_val = 0, max_val = 0;
  GET_PARAM_INT(op_ctx, "min_val", min_val);
  GET_PARAM_INT(op_ctx, "max_val", max_val);
#if DEBUG_MODE
  printf("[RknnCustomOpExam] min_val (i) = %d\n", min_val);
  printf("[RknnCustomOpExam] max_val (i) = %d\n", max_val);
#endif

  /* dtype "f": float scalars */
  float scale = 0.0f, shift = 0.0f;
  GET_PARAM_FLOAT(op_ctx, "scale", scale);
  GET_PARAM_FLOAT(op_ctx, "shift", shift);
#if DEBUG_MODE
  printf("[RknnCustomOpExam] scale   (f) = %f\n", scale);
  printf("[RknnCustomOpExam] shift   (f) = %f\n", shift);
#endif

  /* dtype "s": string */
  char mode_str[256];
  uint32_t mode_len = 0;
  GET_PARAM_STRING(op_ctx, "mode", mode_str, sizeof(mode_str), mode_len);
#if DEBUG_MODE
  printf("[RknnCustomOpExam] mode    (s) = '%s' (len=%u)\n", mode_str, mode_len);
#endif

  /* dtype "is": int32 array */
  {
    rknn3_custom_op_attr attr;
    int ret = op_ctx->get_param(op_ctx, "strides", &attr);
    if (ret != 0 || !attr.data) {
      printf("[RknnCustomOpExam] get_param('strides') failed, ret=%d\n", ret);
      return -1;
    }
#if DEBUG_MODE
    printf("[RknnCustomOpExam] strides (is) = [");
    for (uint32_t i = 0; i < attr.n_elems; i++) {
      printf("%d%s", ((int32_t *)attr.data)[i], i < attr.n_elems - 1 ? ", " : "");
    }
    printf("] (n_elems=%u)\n", attr.n_elems);
#endif
  }

  /* dtype "fs": float array */
  {
    rknn3_custom_op_attr attr;
    int ret = op_ctx->get_param(op_ctx, "weights", &attr);
    if (ret != 0 || !attr.data) {
      printf("[RknnCustomOpExam] get_param('weights') failed, ret=%d\n", ret);
      return -1;
    }
#if DEBUG_MODE
    printf("[RknnCustomOpExam] weights (fs) = [");
    for (uint32_t i = 0; i < attr.n_elems; i++) {
      printf("%f%s", ((float *)attr.data)[i], i < attr.n_elems - 1 ? ", " : "");
    }
    printf("] (n_elems=%u)\n", attr.n_elems);
#endif
  }

  /* dtype "ss": string array (NUL-separated in data, n_elems = count) */
  {
    rknn3_custom_op_attr attr;
    int ret = op_ctx->get_param(op_ctx, "tags", &attr);
    if (ret != 0 || !attr.data) {
      printf("[RknnCustomOpExam] get_param('tags') failed, ret=%d\n", ret);
      return -1;
    }
#if DEBUG_MODE
    printf("[RknnCustomOpExam] tags    (ss) = [");
    const char *p = (const char *)attr.data;
    for (uint32_t i = 0; i < attr.n_elems; i++) {
      printf("'%s'%s", p, i < attr.n_elems - 1 ? ", " : "");
      p += strlen(p) + 1;
    }
    printf("] (n_elems=%u)\n", attr.n_elems);
#endif
  }

  /* ---- Actual computation: y = clamp(x * scale + shift, min_val, max_val) ---- */

  if (n_inputs < 1 || n_outputs < 1) {
    printf("[RknnCustomOpExam] invalid n_inputs=%u n_outputs=%u\n", n_inputs, n_outputs);
    return RKNN3_ERR_ARGUMENT_INVALID;
  }

  rknn3_tensor_attr *in_attr  = inputs[0].attr;
  rknn3_tensor_attr *out_attr = outputs[0].attr;

  /* Logical NCHW dims come from orig_shape (the user-visible shape).
   * The packed NC1HWC2 shape is in attr->shape (5-dim: N, C1, H, W, C2). */
  int N = (int)in_attr->orig_shape[0];
  int C = (int)in_attr->orig_shape[1];
  int H = (int)in_attr->orig_shape[2];
  int W = (int)in_attr->orig_shape[3];
  uint64_t n_elems = (uint64_t)N * C * H * W;

  void *input_raw  = inputs[0].mem ? inputs[0].mem->virt_addr : NULL;
  void *output_raw = outputs[0].mem ? outputs[0].mem->virt_addr : NULL;
  if (!input_raw || !output_raw) {
    printf("[RknnCustomOpExam] null tensor memory (in=%p out=%p)\n", input_raw, output_raw);
    return RKNN3_ERR_ARGUMENT_INVALID;
  }

  /* Step 1: unpack input to NCHW fp16, then convert to fp32.
   * For NC1HWC2 the conversion needs the 5-dim packed shape + logical C. */
  float *input_f32 = (float *)malloc(n_elems * sizeof(float));
  if (!input_f32) {
    printf("[RknnCustomOpExam] malloc input_f32 failed\n");
    return RKNN3_ERR_FAIL;
  }

  float16 *input_fp16 = (float16 *)malloc(n_elems * sizeof(float16));
  if (!input_fp16) {
    printf("[RknnCustomOpExam] malloc input_fp16 failed\n");
    free(input_f32);
    return RKNN3_ERR_FAIL;
  }

  if (in_attr->layout == RKNN3_TENSOR_NC1HWC2) {
    if (in_attr->dtype != RKNN3_TENSOR_FLOAT16) {
      printf("[RknnCustomOpExam] NC1HWC2 only supports FLOAT16 input, got dtype=%d\n", in_attr->dtype);
      free(input_f32); free(input_fp16);
      return RKNN3_ERR_ARGUMENT_INVALID;
    }
    int ret = nc1hwc2_to_nchw_fp16(input_raw, input_fp16, in_attr->dtype,
                                   in_attr->shape, C);
    if (ret) { free(input_f32); free(input_fp16); return ret; }
  } else {
    /* NCHW (or other): straight copy / dtype convert to fp16 */
    if (in_attr->dtype == RKNN3_TENSOR_FLOAT16) {
      memcpy(input_fp16, input_raw, n_elems * sizeof(float16));
    } else if (in_attr->dtype == RKNN3_TENSOR_FLOAT32) {
      const float *src = (const float *)input_raw;
      for (uint64_t i = 0; i < n_elems; i++) input_fp16[i] = fp32_to_fp16(src[i]);
    } else {
      printf("[RknnCustomOpExam] unsupported input dtype %d (layout=%d)\n", in_attr->dtype, in_attr->layout);
      free(input_f32); free(input_fp16);
      return RKNN3_ERR_ARGUMENT_INVALID;
    }
  }

  /* fp16 -> fp32 for the math */
  for (uint64_t i = 0; i < n_elems; i++) {
    input_f32[i] = fp16_to_fp32(input_fp16[i]);
  }

  /* Step 2: compute y = clamp(x * scale + shift, min_val, max_val) in fp32 */
  float *output_f32 = (float *)malloc(n_elems * sizeof(float));
  if (!output_f32) {
    printf("[RknnCustomOpExam] malloc output_f32 failed\n");
    free(input_f32); free(input_fp16);
    return RKNN3_ERR_FAIL;
  }

  float fmin = (float)min_val;
  float fmax = (float)max_val;
  for (uint64_t i = 0; i < n_elems; i++) {
    float v = input_f32[i] * scale + shift;
    if (v < fmin) v = fmin;
    if (v > fmax) v = fmax;
    output_f32[i] = v;
  }

  /* Step 3: convert output back to the output tensor's layout/dtype. */
  float16 *output_fp16 = (float16 *)malloc(n_elems * sizeof(float16));
  if (!output_fp16) {
    printf("[RknnCustomOpExam] malloc output_fp16 failed\n");
    free(input_f32); free(input_fp16); free(output_f32);
    return RKNN3_ERR_FAIL;
  }
  for (uint64_t i = 0; i < n_elems; i++) {
    output_fp16[i] = fp32_to_fp16(output_f32[i]);
  }

  if (out_attr->layout == RKNN3_TENSOR_NC1HWC2) {
    if (out_attr->dtype != RKNN3_TENSOR_FLOAT16) {
      printf("[RknnCustomOpExam] NC1HWC2 output only supports FLOAT16, got dtype=%d\n", out_attr->dtype);
      free(input_f32); free(input_fp16); free(output_f32); free(output_fp16);
      return RKNN3_ERR_ARGUMENT_INVALID;
    }
    int ret = nchw_fp16_to_nc1hwc2(output_fp16, (float16 *)output_raw, out_attr->dtype,
                                   out_attr->shape, C);
    if (ret) { free(input_f32); free(input_fp16); free(output_f32); free(output_fp16); return ret; }
  } else {
    /* NCHW (or other) */
    if (out_attr->dtype == RKNN3_TENSOR_FLOAT16) {
      memcpy(output_raw, output_fp16, n_elems * sizeof(float16));
    } else if (out_attr->dtype == RKNN3_TENSOR_FLOAT32) {
      float *dst = (float *)output_raw;
      for (uint64_t i = 0; i < n_elems; i++) dst[i] = fp16_to_fp32(output_fp16[i]);
    } else {
      printf("[RknnCustomOpExam] unsupported output dtype %d (layout=%d)\n", out_attr->dtype, out_attr->layout);
      free(input_f32); free(input_fp16); free(output_f32); free(output_fp16);
      return RKNN3_ERR_ARGUMENT_INVALID;
    }
  }

  free(input_f32);
  free(input_fp16);
  free(output_f32);
  free(output_fp16);

#if DEBUG_MODE
  printf("[RknnCustomOpExam] compute done, n_elems=%llu\n", (unsigned long long)n_elems);
#endif
  return 0;
}

/* ------------------------------------------------------------------ */
/*  Postprocess op: simple pass-through (memcpy)                      */
/* ------------------------------------------------------------------ */

int rknn_postprocess_compute(rknn3_custom_op_context *op_ctx,
                                  rknn3_tensor *inputs, uint32_t n_inputs,
                                  rknn3_tensor *outputs, uint32_t n_outputs)
{
#if DEBUG_MODE
  printf("[RknnPostProcess] compute called (pass-through)\n");
#endif
  if (n_inputs > 0 && n_outputs > 0) {
    void *src = inputs[0].mem ? inputs[0].mem->virt_addr : NULL;
    void *dst = outputs[0].mem ? outputs[0].mem->virt_addr : NULL;
    if (!src || !dst) {
      printf("[RknnPostProcess] null tensor memory (in=%p out=%p)\n", src, dst);
      return RKNN3_ERR_ARGUMENT_INVALID;
    }
    /* copy min of the two to avoid overrunning the smaller buffer */
    size_t copy_sz = inputs[0].mem->size < outputs[0].mem->size
                         ? inputs[0].mem->size
                         : outputs[0].mem->size;
    memcpy(dst, src, copy_sz);
  }
  return 0;
}

int rknn_postprocess_get_attrs(rknn3_custom_op_context *op_ctx,
                                    rknn3_tensor_attr *input_attrs, uint32_t n_inputs,
                                    rknn3_tensor_attr *output_attrs, uint32_t n_outputs)
{
#if DEBUG_MODE
  printf("[RknnPostProcess] get_attrs called\n");
#endif
  if (!op_ctx || !input_attrs || (n_inputs == 0) || (n_outputs != 1) || !output_attrs) {
    printf("[RknnPostProcess] Error: Invalid params!\n");
    return -1;
  }
  /* Copy input attrs to output */
  output_attrs[0] = input_attrs[0];
  output_attrs[0].index = 0;
  strncpy(output_attrs[0].name, "postprocess_output", RKNN3_MAX_NAME_LEN - 1);
  return 0;
}

int rknn_postprocess_get_output_num(rknn3_custom_op_context *op_ctx)
{
#if DEBUG_MODE
  printf("[RknnPostProcess] get_output_num called\n");
#endif
  return 1;
}
