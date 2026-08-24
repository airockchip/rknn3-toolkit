/*
 * RKNN3 model test demo.
 *
 * End-to-end flow: parse CLI -> init device context -> load model (path, data,
 * or streaming) -> allocate I/O tensors -> fill inputs -> run inference loop ->
 * optionally verify outputs against golden .npy files.
 */
#include "cnpy.h"
#include "float16.h"
#include "rknn3_api.h"
#include "rknn3_test_utils.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <algorithm>
#include <string>
#include <vector>

#define DEFAULT_LOOP_COUNT 1
#define DEFAULT_SHAPE_ID   0
#define DEFAULT_ENABLE_OUTPUT_CHECKSUM 1

// --- Utility helpers ---

// Portable strdup for splitting '#' separated file lists.
static char* _strdup(const char* str)
{
  size_t len     = strlen(str) + 1;
  char*  new_str = (char*)malloc(len);
  if (new_str) {
    memcpy(new_str, str, len);
  }
  return new_str;
}
/** Print shape, stride, layout, dtype, and quantization info for one tensor. */
static void dump_tensor_attr(rknn3_tensor_attr* attrs)
{
  std::string shape_str = "";
  for (uint32_t j = 0; j < attrs->n_dims; j++) {
    shape_str += std::to_string(attrs->shape[j]);
    if (j < attrs->n_dims - 1) {
      shape_str += ", ";
    }
  }

  std::string stride_str = "";
  for (uint32_t j = 0; j < attrs->n_stride; j++) {
    stride_str += std::to_string(attrs->stride[j]);
    if (j < attrs->n_stride - 1) {
      stride_str += ", ";
    }
  }

  printf("  name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, "
             "qnt_type=%s, scale=%.5f, zero_point=%d\n",
             attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), attrs->aligned_size, rknn3_get_layout_string(attrs->layout),
             rknn3_get_type_string(attrs->dtype), rknn3_get_qnt_type_string(attrs->qnt_type), attrs->qnt_info.scale, attrs->qnt_info.zero_point);
}

/** Print one tensor element according to its data type. */
static void print_data_value(void* data, rknn3_tensor_type type, size_t index)
{
  switch (type) {
  case RKNN3_TENSOR_FLOAT16: {
    float16* fp16_data = (float16*)data;
    uint16_t fp16_val  = *(uint16_t*)&fp16_data[index];
    printf("[%zu] FP16: %d \n", index, fp16_val);
    break;
  }
  case RKNN3_TENSOR_FLOAT32: {
    float* fp32_data = (float*)data;
    printf("[%zu] FP32: %f\n", index, fp32_data[index]);
    break;
  }
  case RKNN3_TENSOR_INT8: {
    int8_t* int8_data = (int8_t*)data;
    printf("[%zu] INT8: %d\n", index, int8_data[index]);
    break;
  }
  case RKNN3_TENSOR_INT16: {
    int16_t* int16_data = (int16_t*)data;
    printf("[%zu] INT16: %d\n", index, int16_data[index]);
    break;
  }
  case RKNN3_TENSOR_INT32: {
    int32_t* int32_data = (int32_t*)data;
    printf("[%zu] INT32: %d\n", index, int32_data[index]);
    break;
  }
  case RKNN3_TENSOR_UINT8: {
    uint8_t* uint8_data = (uint8_t*)data;
    printf("[%zu] UINT8: %u\n", index, uint8_data[index]);
    break;
  }
  case RKNN3_TENSOR_UINT16: {
    uint16_t* uint16_data = (uint16_t*)data;
    printf("[%zu] UINT16: %u\n", index, uint16_data[index]);
    break;
  }
  case RKNN3_TENSOR_UINT32: {
    uint32_t* uint32_data = (uint32_t*)data;
    printf("[%zu] UINT32: %u\n", index, uint32_data[index]);
    break;
  }
  default:
    printf("[%zu] Unknown data type\n", index);
    break;
  }
}

/** Print the first @p count elements from a tensor buffer. */
static void print_data_values(void* data, rknn3_tensor_type type, size_t size, size_t count)
{
  size_t element_size = 0;
  switch (type) {
  case RKNN3_TENSOR_FLOAT16:
    element_size = sizeof(float16);
    break;
  case RKNN3_TENSOR_FLOAT32:
    element_size = sizeof(float);
    break;
  case RKNN3_TENSOR_INT8:
  case RKNN3_TENSOR_UINT8:
    element_size = sizeof(int8_t);
    break;
  case RKNN3_TENSOR_INT16:
  case RKNN3_TENSOR_UINT16:
    element_size = sizeof(int16_t);
    break;
  case RKNN3_TENSOR_INT32:
  case RKNN3_TENSOR_UINT32:
    element_size = sizeof(int32_t);
    break;
  case RKNN3_TENSOR_INT64:
  case RKNN3_TENSOR_UINT64:
    element_size = sizeof(int64_t);
    break;
  default:
    element_size = sizeof(float16);
    break;
  }

  size_t max_elements = size / element_size;
  size_t print_count  = (count < max_elements) ? count : max_elements;

  for (size_t i = 0; i < print_count; i++) {
    print_data_value(data, type, i);
  }
}

/** Count how many bits are set in @p core_mask. */
static int get_core_num(uint32_t core_mask)
{
  int core_num = 0;
  for (int i = 0; i < 32; i++) {
    if (core_mask & (1 << i)) {
      core_num++;
    }
  }
  return core_num;
}

/** Compute a stable checksum over the complete native output buffer. */
static uint64_t calc_checksum_fnv1a64(const void* data, size_t size)
{
  const uint8_t* bytes = (const uint8_t*)data;
  uint64_t       hash  = 1469598103934665603ULL;
  for (size_t i = 0; i < size; i++) {
    hash ^= (uint64_t)bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

/** Compute cosine similarity between two float vectors. */
static float cosine_similarity(const float* a, const float* b, size_t size)
{
  float dot_product = 0.0f;
  float norm_a      = 0.0f;
  float norm_b      = 0.0f;
  for (size_t i = 0; i < size; i++) {
    float val_a = a[i];
    float val_b = b[i];

    if (!isfinite(val_a) || !isfinite(val_b))
      return 0.0f;

    dot_product += val_a * val_b;
    norm_a += val_a * val_a;
    norm_b += val_b * val_b;
  }

  if (norm_a < FLT_EPSILON || norm_b < FLT_EPSILON) {
    return 0.0f;
  }

  float norm_product = sqrtf(norm_a) * sqrtf(norm_b);

  if (norm_product < FLT_EPSILON) {
    return 0.0f;
  }

  float similarity = dot_product / norm_product;

  // Clamp to [-1.0, 1.0]
  if (similarity > 1.0f)
    similarity = 1.0f;
  if (similarity < -1.0f)
    similarity = -1.0f;

  return similarity;
}

static float euclidean_distance(const float* a, const float* b, size_t size)
{
  float sum_sq = 0.0f;
  for (size_t i = 0; i < size; i++) {
    if (!isfinite(a[i]) || !isfinite(b[i])) {
      return INFINITY;
    }
    float diff = a[i] - b[i];
    sum_sq += diff * diff;
  }
  return sqrtf(sum_sq);
}

// --- Tensor dtype conversion ---

/** Cast or copy @p n_elems FP32 values into the destination tensor type. */
static int convert_fp32_to_any_type(const float* src, void* dst, int n_elems, rknn3_tensor_type type)
{
  switch (type) {
  case RKNN3_TENSOR_FLOAT16:
    for (int i = 0; i < n_elems; i++) {
      ((float16*)dst)[i] = fp32_to_fp16(src[i]);
    }
    break;
  case RKNN3_TENSOR_FLOAT32:
    memcpy(dst, src, n_elems * sizeof(float));
    break;
  case RKNN3_TENSOR_INT64:
    for (int i = 0; i < n_elems; i++) {
      ((int64_t*)dst)[i] = (int64_t)src[i];
    }
    break;
  case RKNN3_TENSOR_INT32:
    for (int i = 0; i < n_elems; i++) {
      ((int32_t*)dst)[i] = (int32_t)src[i];
    }
    break;
  case RKNN3_TENSOR_INT16:
    for (int i = 0; i < n_elems; i++) {
      ((int16_t*)dst)[i] = (int16_t)src[i];
    }
    break;
  case RKNN3_TENSOR_INT8:
    for (int i = 0; i < n_elems; i++) {
      ((int8_t*)dst)[i] = (int8_t)src[i];
    }
    break;
  case RKNN3_TENSOR_UINT8:
    for (int i = 0; i < n_elems; i++) {
      ((uint8_t*)dst)[i] = (uint8_t)src[i];
    }
    break;
  default:
    printf("Unsupported tensor type: %d\n", type);
    return -1;
  }
  return 0;
}

/** Convert FP16 buffer to FP32 element-wise. */
static int convert_fp16_to_fp32(const float16* src, float* dst, int n_elems)
{
  for (int i = 0; i < n_elems; i++) {
    dst[i] = fp16_to_fp32(src[i]);
  }
  return 0;
}

/** Convert a native tensor buffer to FP32 for golden comparison. */
static int convert_any_type_to_fp32(const void* src, float* dst, int n_elems, rknn3_tensor_type type)
{
  switch (type) {
  case RKNN3_TENSOR_FLOAT16:
    convert_fp16_to_fp32((const float16*)src, dst, n_elems);
    break;
  case RKNN3_TENSOR_FLOAT32:
    memcpy(dst, src, n_elems * sizeof(float));
    break;
  case RKNN3_TENSOR_INT64:
    for (int i = 0; i < n_elems; i++) {
      dst[i] = (float)((int64_t*)src)[i];
    }
    break;
  case RKNN3_TENSOR_INT32:
    for (int i = 0; i < n_elems; i++) {
      dst[i] = (float)((int32_t*)src)[i];
    }
    break;
  case RKNN3_TENSOR_INT16:
    for (int i = 0; i < n_elems; i++) {
      dst[i] = (float)((int16_t*)src)[i];
    }
    break;
  case RKNN3_TENSOR_INT8:
    for (int i = 0; i < n_elems; i++) {
      dst[i] = (float)((int8_t*)src)[i];
    }
    break;
  case RKNN3_TENSOR_UINT8:
    for (int i = 0; i < n_elems; i++) {
      dst[i] = (float)((uint8_t*)src)[i];
    }
    break;
  default:
    printf("Unsupported tensor type: %d\n", type);
    return -1;
  }
  return 0;
}

/** Dequantize INT8 values to FP32 using scale and zero_point. */
static int convert_int8_to_fp32(const void *src, float *dst, int n_elems, rknn3_tensor_type type, float scale, int32_t zero_point)
{
  (void)type;
  for (int i = 0; i < n_elems; i++)
  {
    dst[i] = ((float)((int8_t *)src)[i] - zero_point) * scale;
  }

  return 0;
}

// --- Layout conversion (NCHW / NC1HWC2 / NHWC) ---

/** Repack NCHW FP32 numpy data into NC1HWC2 FP16 device layout. */
static int NCHW_fp32_to_NC1HWC2_fp16(const float* src, float16* dst, int batch, int h, int w, int channel, int sub_c, int align_stride,
                                     int align_hw)
{
  printf("NCHW_fp32_to_NC1HWC2_fp16\n");
  printf("batch=%d, h=%d, w=%d, channel=%d, sub_c=%d, align_stride=%d, "
             "align_hw=%d\n",
             batch, h, w, channel, sub_c, align_stride, align_hw);
  int hw      = w * h;
  int align_c = (channel + sub_c - 1) / sub_c * sub_c;
  for (int b = 0; b < batch; b++) {
    const float* src_b = src + b * channel * hw;
    float16*     dst_b = dst + b * align_c * align_hw;
    for (int c = 0; c < channel; ++c) {
      int      plane    = c / sub_c;
      float16* dstPlane = plane * align_hw * sub_c + dst_b;
      int      offset   = c % sub_c;
      for (int cur_h = 0; cur_h < h; ++cur_h)
        for (int cur_w = 0; cur_w < w; ++cur_w) {
          int cur_hw                        = cur_h * align_stride + cur_w;
          dstPlane[sub_c * cur_hw + offset] = fp32_to_fp16(src_b[c * hw + cur_h * w + cur_w]);
        }
    }
  }

  return 0;
}

/** Repack NCHW FP32 numpy data into NHWC FP16 device layout (zero-pad width stride). */
static int NCHW_fp32_to_NHWC_fp16(const float* src, float16* dst, int batch, int h, int w, int channel, int w_stride)
{
  printf("NCHW_fp32_to_NHWC_fp16\n");
  printf("batch=%d, h=%d, w=%d, channel=%d, w_stride=%d\n", batch, h, w, channel, w_stride);
  int hw = w * h;
  for (int b = 0; b < batch; b++) {
    const float* src_b = src + b * channel * hw;
    float16*     dst_b = dst + b * hw * channel;
    for (int c = 0; c < channel; ++c) {
      for (int cur_h = 0; cur_h < h; ++cur_h)
        for (int cur_w = 0; cur_w < w_stride; ++cur_w) {
          if (cur_w < w) {
            dst_b[cur_h * w_stride * channel + cur_w * channel + c] = fp32_to_fp16(src_b[c * hw + cur_h * w + cur_w]);
          } else {
            dst_b[cur_h * w_stride * channel + cur_w * channel + c] = fp32_to_fp16(0.0);
          }
        }
    }
  }
  return 0;
}

/** Unpack NC1HWC2 FP16 device output into NCHW FP32 for comparison. */
static int NC1HWC2_fp16_to_NCHW_fp32(const float16* src, float* dst, int batch, int C1, int C2, int hw_src, int channel, int h, int w)
{
  printf("NC1HWC2_fp16_to_NCHW_fp32\n");
  printf("batch=%d, C1=%d, C2=%d, hw_src=%d, channel=%d, h=%d, w=%d\n", batch, C1, C2, hw_src, channel, h, w);
  int hw_dst = h * w;
  for (int i = 0; i < batch; i++) {
    const float16* src_b = src + i * C1 * hw_src * C2;
    float*         dst_b = dst + i * channel * hw_dst;
    for (int c = 0; c < channel; ++c) {
      int            plane  = c / C2;
      const float16* src_bc = plane * hw_src * C2 + src_b;
      int            offset = c % C2;
      for (int cur_h = 0; cur_h < h; ++cur_h)
        for (int cur_w = 0; cur_w < w; ++cur_w) {
          int cur_hw                 = cur_h * w + cur_w;
          dst_b[c * hw_dst + cur_hw] = fp16_to_fp32(src_bc[C2 * cur_hw + offset]); // float16-->float
        }
    }
  }

  return 0;
}

/** Repack UINT8 NCHW numpy data into NC1HWC2 FP16 or INT8 device layout. */
static int convert_uint8_NCHW_to_NC1HWC2(const uint8_t* src, void* dst_ptr, int batch, int h, int w, int channel, int align_stride,
                                         int align_hw, rknn3_tensor_attr* attr)
{
  // gongga-toolkit only tags mean=0/std=1 inputs as NC1HWC2; no normalize here.
  if (attr->dtype == RKNN3_TENSOR_FLOAT16) {
    int      sub_c = attr->n_dims > 4 ? attr->shape[4] : 1;
    float16* dst   = (float16*)dst_ptr;
    printf("NCHW_uint8_to_NC1HWC2_fp16\n");
    printf("batch=%d, h=%d, w=%d, channel=%d, sub_c=%d, align_stride=%d, "
           "align_hw=%d\n",
           batch, h, w, channel, sub_c, align_stride, align_hw);
    int hw      = w * h;
    int align_c = (channel + sub_c - 1) / sub_c * sub_c;
    for (int b = 0; b < batch; b++) {
      const uint8_t* src_b = src + b * channel * hw;
      float16*       dst_b = dst + b * align_c * align_hw;
      for (int c = 0; c < channel; ++c) {
        int      plane    = c / sub_c;
        float16* dstPlane = plane * align_hw * sub_c + dst_b;
        int      offset   = c % sub_c;
        for (int cur_h = 0; cur_h < h; ++cur_h)
          for (int cur_w = 0; cur_w < w; ++cur_w) {
            int cur_hw                        = cur_h * align_stride + cur_w;
            dstPlane[sub_c * cur_hw + offset] = fp32_to_fp16(((float)src_b[c * hw + cur_h * w + cur_w]));
          }
      }
    }
  } else if (attr->dtype == RKNN3_TENSOR_INT8 || attr->dtype == RKNN3_TENSOR_UINT8) {
    int     sub_c      = attr->n_dims > 4 ? attr->shape[4] : 1;
    float   scale      = attr->qnt_info.scale;
    int32_t zero_point = attr->qnt_info.zero_point;
    uint8_t* dst       = (uint8_t*)dst_ptr;
    if (scale == 0.0f) {
      printf("Invalid quantization scale 0 for tensor %s\n", attr->name);
      return -1;
    }
    printf("NCHW_uint8_to_NC1HWC2_%s\n", attr->dtype == RKNN3_TENSOR_INT8 ? "int8" : "uint8");
    printf("batch=%d, h=%d, w=%d, channel=%d, sub_c=%d, align_stride=%d, "
           "align_hw=%d, scale=%f, zero_point=%d\n",
           batch, h, w, channel, sub_c, align_stride, align_hw, scale, zero_point);
    int hw      = w * h;
    int align_c = (channel + sub_c - 1) / sub_c * sub_c;
    for (int b = 0; b < batch; b++) {
      const uint8_t* src_b = src + b * channel * hw;
      uint8_t*       dst_b = dst + b * align_c * align_hw;
      for (int c = 0; c < channel; ++c) {
        int      plane    = c / sub_c;
        uint8_t* dstPlane = plane * align_hw * sub_c + dst_b;
        int      offset   = c % sub_c;
        for (int cur_h = 0; cur_h < h; ++cur_h)
          for (int cur_w = 0; cur_w < w; ++cur_w) {
            int cur_hw = cur_h * align_stride + cur_w;
            int32_t quantized = (int32_t)rintf((float)src_b[c * hw + cur_h * w + cur_w] / scale) + zero_point;
            if (attr->dtype == RKNN3_TENSOR_INT8) {
              quantized = std::max(-128, std::min(127, quantized));
              ((int8_t*)dstPlane)[sub_c * cur_hw + offset] = (int8_t)quantized;
            } else {
              quantized = std::max(0, std::min(255, quantized));
              dstPlane[sub_c * cur_hw + offset] = (uint8_t)quantized;
            }
          }
      }
    }
  } else {
    printf("Unsupported dst_type: %d\n", attr->dtype);
    return -1;
  }
  return 0;
}

/** Convert UINT8 NCHW numpy data to an NHWC tensor, including width padding. */
static int convert_uint8_NCHW_to_NHWC(const uint8_t* src, void* dst, int batch, int h, int w,
                                      int channel, int w_stride, rknn3_tensor_attr* attr)
{
  for (int b = 0; b < batch; b++) {
    const uint8_t* src_b = src + b * channel * h * w;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w_stride; x++) {
        for (int c = 0; c < channel; c++) {
          size_t dst_idx = ((size_t)b * h * w_stride + y * w_stride + x) * channel + c;
          uint8_t value = x < w ? src_b[c * h * w + y * w + x] : 0;
          switch (attr->dtype) {
          case RKNN3_TENSOR_FLOAT16:
            ((float16*)dst)[dst_idx] = fp32_to_fp16((float)value);
            break;
          case RKNN3_TENSOR_FLOAT32:
            ((float*)dst)[dst_idx] = (float)value;
            break;
          case RKNN3_TENSOR_UINT8:
            ((uint8_t*)dst)[dst_idx] = value;
            break;
          case RKNN3_TENSOR_INT8:
            ((int8_t*)dst)[dst_idx] = x < w
                ? (int8_t)((float)value / attr->qnt_info.scale + attr->qnt_info.zero_point)
                : 0;
            break;
          default:
            printf("Unsupported type for UINT8-to-NHWC conversion: %s\n", rknn3_get_type_string(attr->dtype));
            return -1;
          }
        }
      }
    }
  }
  return 0;
}

/** Quantize FP32 values to INT8 using tensor scale and zero_point. */
static int convert_fp32_to_int8(const float* src, int8_t* dst, int n_elems, rknn3_tensor_attr* attr) {
  float   scale      = attr->qnt_info.scale;
  int32_t zero_point = attr->qnt_info.zero_point;
  for (int i = 0; i < n_elems; i++) {
    dst[i] = ((float)src[i]) / scale + zero_point;
  }
  return 0;
}


/** Repack FP32 NCHW numpy data into NC1HWC2 FP16 or INT8 device layout. */
static int convert_fp32_NCHW_to_NC1HWC2(const float* src, void* dst_ptr, int batch, int h, int w, int channel, int align_stride,
                                         int align_hw, rknn3_tensor_attr* attr)
{
  // gongga-toolkit only tags mean=0/std=1 inputs as NC1HWC2; no normalize here.
  if (attr->dtype == RKNN3_TENSOR_FLOAT16) {
    int      sub_c = attr->n_dims > 4 ? attr->shape[4] : 1;
    float16* dst   = (float16*)dst_ptr;
    printf("NCHW_fp32_to_NC1HWC2_fp16\n");
    printf("batch=%d, h=%d, w=%d, channel=%d, sub_c=%d, align_stride=%d, "
           "align_hw=%d\n",
           batch, h, w, channel, sub_c, align_stride, align_hw);
    int hw      = w * h;
    int align_c = (channel + sub_c - 1) / sub_c * sub_c;
    for (int b = 0; b < batch; b++) {
      const float* src_b = src + b * channel * hw;
      float16*       dst_b = dst + b * align_c * align_hw;
      for (int c = 0; c < channel; ++c) {
        int      plane    = c / sub_c;
        float16* dstPlane = plane * align_hw * sub_c + dst_b;
        int      offset   = c % sub_c;
        for (int cur_h = 0; cur_h < h; ++cur_h)
          for (int cur_w = 0; cur_w < w; ++cur_w) {
            int cur_hw                        = cur_h * align_stride + cur_w;
            dstPlane[sub_c * cur_hw + offset] = fp32_to_fp16(((float)src_b[c * hw + cur_h * w + cur_w]));
          }
      }
    }
  } else if (attr->dtype == RKNN3_TENSOR_INT8) {
    int     sub_c      = attr->n_dims > 4 ? attr->shape[4] : 1;
    int8_t* dst        = (int8_t*)dst_ptr;
    float   scale      = attr->qnt_info.scale;
    int32_t zero_point = attr->qnt_info.zero_point;
    printf("NCHW_fp32_to_NC1HWC2_int8\n");
    printf("batch=%d, h=%d, w=%d, channel=%d, sub_c=%d, align_stride=%d, "
           "align_hw=%d, scale=%f, zp=%d\n",
           batch, h, w, channel, sub_c, align_stride, align_hw, scale, zero_point);
    int hw      = w * h;
    int align_c = (channel + sub_c - 1) / sub_c * sub_c;
    for (int b = 0; b < batch; b++) {
      const float* src_b = src + b * channel * hw;
      int8_t*        dst_b = dst + b * align_c * align_hw;
      for (int c = 0; c < channel; ++c) {
        int     plane    = c / sub_c;
        int8_t* dstPlane = plane * align_hw * sub_c + dst_b;
        int     offset   = c % sub_c;
        for (int cur_h = 0; cur_h < h; ++cur_h)
          for (int cur_w = 0; cur_w < w; ++cur_w) {
            int   cur_hw                      = cur_h * align_stride + cur_w;
            float dst_value                   = ((float)src_b[c * hw + cur_h * w + cur_w]) / scale + zero_point;
            dst_value = rint(dst_value);
            dstPlane[sub_c * cur_hw + offset] = (int8_t)dst_value;
          }
      }
    }
  } else {
    printf("Unsupported dst_type: %d\n", attr->dtype);
    return -1;
  }
  return 0;
}

/** Return the product of all dimensions in @p shape. */
static inline uint32_t shape_count(uint32_t* shape, int n_dims)
{
  uint32_t elems = 1;
  for (int i = 0; i < n_dims; i++) {
    elems *= shape[i];
  }
  return elems;
}

/** Unpack NC1HWC2 INT8 device output into NCHW FP32 (with dequantization). */
static int NC1HWC2_int8_to_NCHW_fp32(const int8_t* src, float* dst, int batch, int C1, int C2, int hw_src, int channel, int h, int w,
                                     rknn3_tensor_attr* attr)
{
  printf("NC1HWC2_int8_to_NCHW_fp32\n");
  printf("batch=%d, C1=%d, C2=%d, hw_src=%d, channel=%d, h=%d, w=%d\n", batch, C1, C2, hw_src, channel, h, w);

  // Read quantization parameters
  float   scale      = 1.0f;
  int32_t zero_point = 0;

  if (attr->qnt_type != RKNN3_TENSOR_QNT_NONE) {
    scale      = attr->qnt_info.scale;
    zero_point = attr->qnt_info.zero_point;
    printf("Quantization params: scale=%f, zero_point=%d\n", scale, zero_point);
  } else {
    printf("No quantization parameters found, using default values\n");
  }

  int hw_dst = h * w;
  for (int i = 0; i < batch; i++) {
    const int8_t* src_b = src + i * C1 * hw_src * C2;
    float*        dst_b = dst + i * channel * hw_dst;
    for (int c = 0; c < channel; ++c) {
      int           plane  = c / C2;
      const int8_t* src_bc = plane * hw_src * C2 + src_b;
      int           offset = c % C2;
      for (int cur_h = 0; cur_h < h; ++cur_h)
        for (int cur_w = 0; cur_w < w; ++cur_w) {
          int    cur_hw          = cur_h * w + cur_w;
          int8_t quantized_value = src_bc[C2 * cur_hw + offset];
          // Dequantize: float = (int8 - zero_point) * scale
          dst_b[c * hw_dst + cur_hw] = (float)(quantized_value - zero_point) * scale;
        }
    }
  }

  return 0;
}

/** Unpack NC1HWC2 INT32 device output into NCHW FP32. */
static int NC1HWC2_int32_to_NCHW_fp32(const int32_t* src, float* dst, int batch, int C1, int C2,
                                      int hw_src, int channel, int h, int w)
{
  int hw_dst = h * w;
  for (int b = 0; b < batch; b++) {
    const int32_t* src_b = src + b * C1 * hw_src * C2;
    float*         dst_b = dst + b * channel * hw_dst;
    for (int c = 0; c < channel; c++) {
      const int32_t* src_bc = src_b + (c / C2) * hw_src * C2;
      int offset = c % C2;
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          int hw = y * w + x;
          dst_b[c * hw_dst + hw] = (float)src_bc[C2 * hw + offset];
        }
      }
    }
  }
  return 0;
}

// --- Streaming weight upload ---

/**
 * Upload weight file in fixed-size chunks after model struct is loaded.
 * Used when model blob is passed via rknn3_load_model_from_data with NULL weight.
 */
static int upload_weight_streaming(rknn3_context ctx, const char* weight_path, size_t chunk_size)
{
  FILE* fp = fopen(weight_path, "rb");
  if (!fp) {
    printf("fopen weight file '%s' failed!\n", weight_path);
    return -1;
  }

  struct stat st;
  if (fstat(fileno(fp), &st) != 0) {
    printf("fstat weight file '%s' failed!\n", weight_path);
    fclose(fp);
    return -1;
  }
  uint64_t total_size = (uint64_t)st.st_size;
  fseek(fp, 0, SEEK_SET);

  void* buf = malloc(chunk_size);
  if (!buf) {
    printf("malloc chunk buffer (%zu bytes) failed\n", chunk_size);
    fclose(fp);
    return -1;
  }

  uint64_t remaining   = total_size;
  uint64_t file_pos    = 0;
  uint32_t chunk_count = 0;
  int      ret         = RKNN3_SUCCESS;

  printf("Streaming weight upload: total=%llu bytes, chunk_size=%.1f MB\n",
         (unsigned long long)total_size, chunk_size / (1024.0 * 1024.0));

  while (remaining > 0) {
    size_t this_chunk = (remaining > chunk_size) ? chunk_size : (size_t)remaining;
    size_t read_n     = fread(buf, 1, this_chunk, fp);
    if (read_n != this_chunk) {
      printf("fread failed: expect %zu got %zu (offset=%llu)\n",
             this_chunk, read_n, (unsigned long long)file_pos);
      ret = -1;
      break;
    }

    ret = rknn3_load_weight_chunk(ctx, file_pos, buf, (uint64_t)this_chunk);
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_load_weight_chunk failed! offset=%llu size=%zu ret=%d\n",
             (unsigned long long)file_pos, this_chunk, ret);
      break;
    }

    file_pos  += this_chunk;
    remaining -= this_chunk;
    chunk_count++;

    // NOTE: per-chunk progress logging is for debug. For small chunk_size,
    // consider throttling output (e.g. print every N percent) to reduce overhead.
    printf("  %.1f%% (%llu / %llu bytes, chunk #%u)\n",
           100.0 * file_pos / total_size,
           (unsigned long long)file_pos, (unsigned long long)total_size,
           chunk_count);
  }

  if (ret == RKNN3_SUCCESS) {
    printf("Streaming weight upload done: %u chunks, total: %.2f MB\n",
           chunk_count, total_size / (1024.0 * 1024.0));
  }

  free(buf);
  fclose(fp);
  return ret;
}

// --- Demo configuration ---

/** How model and weight blobs are passed into the runtime. */
typedef enum {
  MODEL_LOAD_PATH = 0, // rknn3_load_model_from_path
  MODEL_LOAD_DATA = 1, // rknn3_load_model_from_data (preload files into memory)
} ModelLoadMode;

/** Parsed command-line options and derived file lists. */
typedef struct {
  const char* model_path;
  const char* weight_path;
  const char* key_path;
  const char* core_mask_arg;
  int32_t     shape_id;
  size_t      chunk_size_mb;   // >0 enables streaming weight upload
  ModelLoadMode load_mode;
  int         loop_count;
  int         enable_output_checksum;
  bool        use_decrypt_key;
  bool        use_random_input;
  bool        skip_golden_comparison;
  std::vector<std::string> input_files;   // '#' separated on CLI
  std::vector<std::string> golden_files;
} DemoConfig;

/** Runtime I/O tensors plus dynamic-shape metadata. */
typedef struct {
  rknn3_input_output_num io_num;
  rknn3_shape_config     shape_config;
  rknn3_shape_info*      shape_infos;
  rknn3_tensor*          inputs;
  rknn3_tensor*          outputs;
} DemoTensors;

/** Per-inference timing breakdown in microseconds. */
typedef struct {
  uint64_t sync_in_us;   // host -> device before rknn3_run
  uint64_t run_us;       // rknn3_run
  uint64_t sync_out_us;  // device -> host after rknn3_run
} InferenceTiming;

/** Elapsed time between two CLOCK_MONOTONIC samples, in microseconds. */
static uint64_t timespec_elapsed_us(const struct timespec* start, const struct timespec* end)
{
  int64_t sec  = (int64_t)end->tv_sec - (int64_t)start->tv_sec;
  int64_t nsec = (int64_t)end->tv_nsec - (int64_t)start->tv_nsec;
  if (nsec < 0) {
    sec -= 1;
    nsec += 1000000000LL;
  }
  if (sec < 0) {
    return 0;
  }
  return (uint64_t)(sec * 1000000LL + nsec / 1000LL);
}

/** Convenience wrapper: elapsed microseconds as milliseconds. */
static double elapsed_ms(const struct timespec* start, const struct timespec* end)
{
  return timespec_elapsed_us(start, end) / 1000.0;
}

/** Print CLI usage and exit hints. */
static void print_demo_usage(const char* prog)
{
  printf("Usage: %s <model_path> <weight_path> [core_mask] [input_npy_paths] [golden_output_npy_paths] "
         "[shape_id] [loop_count] [chunk_size_mb] [load_mode] [key_path] [enable_output_checksum]\n",
         prog);
  printf("  - core_mask: hex bitmask; if omitted, auto-generated from core_number\n");
  printf("  - input_npy_paths / golden_output_npy_paths: '#' separated lists or a .txt list file\n");
  printf("  - If input_npy_paths is omitted, random input is generated\n");
  printf("  - If golden_output_npy_paths is omitted, cosine similarity is skipped\n");
  printf("  - shape_id: optional, default 0, for dynamic shape models\n");
  printf("  - loop_count: optional, default 1\n");
  printf("  - chunk_size_mb: optional, default 0; if > 0, enables streaming weight loading\n");
  printf("  - load_mode: path (default) or data; ignored when chunk_size_mb > 0\n");
  printf("  - key_path: optional decryption key file; use \"none\" to skip decryption\n");
  printf("  - enable_output_checksum: optional, 0 disables, 1 enables (default)\n");
  printf("  - Optional positional args must not be skipped; use \"\" for an empty slot\n");
}

/** Parse '#' separated paths paths from a .txt file. */
static std::vector<std::string> parse_npy_paths(const char* source)
{
  std::vector<std::string> files;
  if (!source || strlen(source) == 0) {
    return files;
  }

  size_t source_len = strlen(source);
  bool is_txt = source_len >= 4 && strcmp(source + source_len - 4, ".txt") == 0;
  char* content = NULL;
  if (is_txt) {
    FILE* fp = fopen(source, "rb");
    if (!fp) {
      printf("Failed to open npy path list file: %s (errno=%d)\n", source, errno);
      return files;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
      printf("Failed to seek npy path list file: %s\n", source);
      fclose(fp);
      return files;
    }
    long file_size = ftell(fp);
    if (file_size < 0 || fseek(fp, 0, SEEK_SET) != 0) {
      printf("Failed to get npy path list file size: %s\n", source);
      fclose(fp);
      return files;
    }
    content = (char*)malloc((size_t)file_size + 1);
    if (!content) {
      fclose(fp);
      return files;
    }
    size_t read_size = fread(content, 1, (size_t)file_size, fp);
    content[read_size] = '\0';
    fclose(fp);
  } else {
    content = _strdup(source);
  }
  if (!content) {
    return files;
  }

  char* token = strtok(content, "#");
  while (token != NULL) {
    files.push_back(token);
    token = strtok(NULL, "#");
  }
  free(content);

  if (is_txt) {
    printf("Loaded %zu path(s) from file: %s\n", files.size(), source);
  }
  return files;
}

/**
 * Parse argv into DemoConfig. Returns -1 and prints usage when argc < 3.
 * Streaming mode (chunk_size_mb > 0) overrides load_mode.
 */
static int parse_demo_config(int argc, char* argv[], DemoConfig* cfg)
{
  *cfg = DemoConfig();
  cfg->shape_id   = DEFAULT_SHAPE_ID;
  cfg->loop_count = DEFAULT_LOOP_COUNT;
  cfg->enable_output_checksum = DEFAULT_ENABLE_OUTPUT_CHECKSUM;
  cfg->load_mode  = MODEL_LOAD_PATH;

  if (argc < 3) {
    print_demo_usage(argv[0]);
    return -1;
  }

  cfg->model_path  = argv[1];
  cfg->weight_path = argv[2];

  cfg->core_mask_arg       = (argc > 3 && strlen(argv[3]) > 0) ? argv[3] : NULL;
  const char* input_paths  = (argc > 4 && strlen(argv[4]) > 0) ? argv[4] : NULL;
  const char* golden_paths = (argc > 5 && strlen(argv[5]) > 0) ? argv[5] : NULL;

  if (argc > 10 && strlen(argv[10]) > 0) {
    cfg->key_path = argv[10];
    if (strcmp(cfg->key_path, "none") != 0 && strcmp(cfg->key_path, "NONE") != 0) {
      cfg->use_decrypt_key = true;
      printf("Decryption key: %s\n", cfg->key_path);
    } else {
      printf("Decryption key: not provided, loading as plain model\n");
    }
  }

  cfg->shape_id = (argc > 6 && strlen(argv[6]) > 0) ? atoi(argv[6]) : DEFAULT_SHAPE_ID;
  if (cfg->shape_id < 0) {
    printf("Invalid shape_id: %d (must be >= 0)\n", cfg->shape_id);
    return -1;
  }
  printf("shape_id: %d\n", cfg->shape_id);

  cfg->loop_count = (argc > 7 && strlen(argv[7]) > 0) ? atoi(argv[7]) : DEFAULT_LOOP_COUNT;
  if (cfg->loop_count < 1) {
    printf("Invalid loop_count: %d (must be >= 1)\n", cfg->loop_count);
    return -1;
  }

  if (argc > 8 && strlen(argv[8]) > 0) {
    char* endptr;
    errno             = 0;
    unsigned long val = strtoul(argv[8], &endptr, 10);
    if (errno == ERANGE || *endptr != '\0' || val > SIZE_MAX / (1024 * 1024)) {
      printf("Invalid chunk_size_mb: %s\n", argv[8]);
      return -1;
    }
    cfg->chunk_size_mb = (size_t)val;
  }
  if (cfg->chunk_size_mb > 0) {
    printf("Streaming weight loading enabled, chunk_size=%zu MB\n", cfg->chunk_size_mb);
  }

  if (argc > 9 && strlen(argv[9]) > 0) {
    if (strcmp(argv[9], "data") == 0 || strcmp(argv[9], "1") == 0) {
      cfg->load_mode = MODEL_LOAD_DATA;
    } else if (strcmp(argv[9], "path") == 0 || strcmp(argv[9], "0") == 0) {
      cfg->load_mode = MODEL_LOAD_PATH;
    } else {
      printf("Invalid load_mode: %s (use path or data)\n", argv[9]);
      return -1;
    }
  }

  cfg->enable_output_checksum =
      (argc > 11 && strlen(argv[11]) > 0) ? atoi(argv[11]) : DEFAULT_ENABLE_OUTPUT_CHECKSUM;
  if (cfg->enable_output_checksum != 0 && cfg->enable_output_checksum != 1) {
    printf("Invalid enable_output_checksum: %d (must be 0 or 1)\n", cfg->enable_output_checksum);
    return -1;
  }
  printf("enable_output_checksum: %d\n", cfg->enable_output_checksum);
  if (cfg->chunk_size_mb > 0) {
    printf("Model load: streaming (rknn3_load_model_from_data + weight chunks)\n");
  } else if (cfg->load_mode == MODEL_LOAD_DATA) {
    printf("Model load: data (rknn3_load_model_from_data)\n");
  } else {
    printf("Model load: path (rknn3_load_model_from_path)\n");
  }

  cfg->use_random_input       = (input_paths == NULL);
  cfg->skip_golden_comparison = (golden_paths == NULL);
  if (cfg->use_random_input) {
    printf("Input paths not provided, using random input\n");
  }
  if (cfg->skip_golden_comparison) {
    printf("Golden output paths not provided, skipping cosine similarity\n");
  }

  cfg->input_files  = parse_npy_paths(input_paths);
  cfg->golden_files = parse_npy_paths(golden_paths);

  return 0;
}

/**
 * Discover RK182X devices, call rknn3_init on the first one, optionally set
 * decrypt key, and print device memory info.
 */
static int init_rknn3_context(rknn3_context* ctx, const DemoConfig* cfg)
{
  rknn3_devices devs;
  memset(&devs, 0, sizeof(devs));

  int ret = rknn3_find_devices(&devs);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_find_devices failed! ret=%d\n", ret);
    return -1;
  }
  printf("Found %d RK182X devices\n", devs.n_devices);
  for (int i = 0; i < devs.n_devices; i++) {
    printf("  Device %d: transfer_type=%s, id=%s\n", i, devs.devices[i].type, devs.devices[i].id);
  }

  if (devs.n_devices == 0) {
    printf("No RK182X devices found\n");
    return -1;
  } else if (devs.n_devices == 1) {
    printf("Info: Only one device found (id=%s), init_extend can be NULL\n", devs.devices[0].id);
  } else {
    printf("Multiple devices found, using the first one (id=%s)\n", devs.devices[0].id);
  }

  rknn3_init_extend init_extend = {0};
  init_extend.device_id         = devs.devices[0].id;

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  ret = rknn3_init(ctx, &init_extend);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_init failed! ret=%d\n", ret);
    return -1;
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("rknn3_init success, cost %.3f ms\n", elapsed_ms(&start, &end));

  if (cfg->use_decrypt_key) {
    printf("Setting decrypt key from: %s\n", cfg->key_path);
    ret = rknn3_set_decrypt_key_from_path(*ctx, cfg->key_path);
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_set_decrypt_key_from_path failed! ret=%d\n", ret);
      rknn3_destroy(*ctx);
      return -1;
    }
    printf("rknn3_set_decrypt_key_from_path success\n");
  }

  rknn3_dev_mem_info dev_mem_info;
  ret = rknn3_query(*ctx, RKNN3_QUERY_DEVICE_MEM_INFO, &dev_mem_info, sizeof(dev_mem_info));
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_query device mem info failed! ret=%d\n", ret);
    rknn3_destroy(*ctx);
    return -1;
  }
  printf("Device Memory Info: total=%zu MB, free=%zu MB\n",
         dev_mem_info.sys_total / (1024 * 1024), dev_mem_info.sys_free / (1024 * 1024));
  for (int i = 0; i < dev_mem_info.node_num; i++) {
    printf("  Node %d: total=%zu MB, free=%zu MB\n", i,
           dev_mem_info.node_mem_info[i].total / (1024 * 1024),
           dev_mem_info.node_mem_info[i].free / (1024 * 1024));
  }

  return 0;
}

// --- Model loading (path / data / streaming) ---

/** Read an entire file into a newly malloc'd buffer; caller must free(*data). */
static int read_file_to_buffer(const char* path, void** data, size_t* size)
{
  FILE* fp = fopen(path, "rb");
  if (fp == NULL) {
    printf("fopen %s fail!\n", path);
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  long file_size = ftell(fp);
  if (file_size < 0) {
    printf("ftell %s fail!\n", path);
    fclose(fp);
    return -1;
  }
  fseek(fp, 0, SEEK_SET);
  void* buf = malloc((size_t)file_size);
  if (!buf) {
    printf("malloc %s fail! size=%ld\n", path, file_size);
    fclose(fp);
    return -1;
  }
  if ((size_t)file_size != fread(buf, 1, (size_t)file_size, fp)) {
    printf("fread %s fail!\n", path);
    free(buf);
    fclose(fp);
    return -1;
  }
  fclose(fp);
  *data = buf;
  *size = (size_t)file_size;
  return 0;
}

/**
 * Resolve run_core_mask from the parsed core_mask argument or auto-generate it.
 * Validates that the number of set bits matches RKNN3_QUERY_CORE_NUMBER.
 */
static int resolve_core_mask(rknn3_context ctx, const DemoConfig* cfg, uint32_t* core_mask)
{
  uint32_t core_num = 0;
  int      ret      = rknn3_query(ctx, RKNN3_QUERY_CORE_NUMBER, &core_num, sizeof(core_num));
  if (ret < 0) {
    printf("rknn3_query core number fail! ret=%d\n", ret);
    return -1;
  }
  printf("Core number: %u\n", core_num);

  if (cfg->core_mask_arg) {
    *core_mask = strtoul(cfg->core_mask_arg, NULL, 16);
    if (get_core_num(*core_mask) != (int)core_num) {
      printf("Error: core_mask 0x%x does not match core number %u\n", *core_mask, core_num);
      return -1;
    }
  } else {
    *core_mask = 0;
    for (uint32_t i = 0; i < core_num; i++) {
      *core_mask |= (1 << i);
    }
    printf("Auto-generated core_mask: 0x%x for %d cores\n", *core_mask, core_num);
  }
  return 0;
}

/**
 * Load model struct from memory, init runtime, then stream weight file in chunks.
 * Host model buffer is freed before return.
 */
static int load_model_streaming_mode(rknn3_context ctx, const DemoConfig* cfg)
{
  struct timespec start, end;
  uint32_t        core_mask = 0;
  void*           model     = NULL;
  size_t          model_len = 0;

  if (read_file_to_buffer(cfg->model_path, &model, &model_len) != 0) {
    return -1;
  }
  printf("model_len=%zu, model=%p\n", model_len, model);

  clock_gettime(CLOCK_MONOTONIC, &start);
  int ret = rknn3_load_model_from_data(ctx, model, (uint64_t)model_len, NULL, 0);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_load_model_from_data (streaming mode) failed! ret=%d\n", ret);
    free(model);
    return -1;
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("rknn3_load_model_from_data (streaming mode, weight=NULL) success, cost %.3f ms\n", elapsed_ms(&start, &end));

  if (resolve_core_mask(ctx, cfg, &core_mask) != 0) {
    free(model);
    return -1;
  }

  rknn3_config config;
  memset(&config, 0, sizeof(config));
  config.run_core_mask = core_mask;
  clock_gettime(CLOCK_MONOTONIC, &start);
  ret = rknn3_model_init(ctx, &config);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_model_init failed! ret=%d\n", ret);
    free(model);
    return -1;
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("rknn3_model_init success, cost %.3f ms\n", elapsed_ms(&start, &end));

  size_t chunk_size = cfg->chunk_size_mb * 1024 * 1024;
  clock_gettime(CLOCK_MONOTONIC, &start);
  ret = upload_weight_streaming(ctx, cfg->weight_path, chunk_size);
  clock_gettime(CLOCK_MONOTONIC, &end);
  if (ret != RKNN3_SUCCESS) {
    printf("Streaming weight upload failed! ret=%d\n", ret);
    free(model);
    return -1;
  }
  printf("Streaming weight upload success, cost %.3f ms\n", elapsed_ms(&start, &end));

  free(model);
  return 0;
}

/**
 * Preload model and weight into host memory, call rknn3_load_model_from_data,
 * then rknn3_model_init. Host buffers are freed before return.
 */
static int load_model_from_data_mode(rknn3_context ctx, const DemoConfig* cfg)
{
  struct timespec start, end;
  uint32_t        core_mask   = 0;
  void*           model       = NULL;
  void*           weight_data = NULL;
  size_t          model_len   = 0;
  size_t          weight_len  = 0;

  if (read_file_to_buffer(cfg->model_path, &model, &model_len) != 0) {
    return -1;
  }
  if (read_file_to_buffer(cfg->weight_path, &weight_data, &weight_len) != 0) {
    free(model);
    return -1;
  }
  printf("model_len=%zu, model=%p, weight_len=%zu, weight_data=%p\n",
         model_len, model, weight_len, weight_data);

  clock_gettime(CLOCK_MONOTONIC, &start);
  int ret = rknn3_load_model_from_data(ctx, model, (uint64_t)model_len, weight_data, (uint64_t)weight_len);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_load_model_from_data failed! ret=%d\n", ret);
    free(model);
    free(weight_data);
    return -1;
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("rknn3_load_model_from_data success%s, cost %.3f ms\n",
         cfg->use_decrypt_key ? " (encrypted model)" : "", elapsed_ms(&start, &end));

  if (resolve_core_mask(ctx, cfg, &core_mask) != 0) {
    free(model);
    free(weight_data);
    return -1;
  }

  rknn3_config config;
  memset(&config, 0, sizeof(config));
  config.run_core_mask = core_mask;
  clock_gettime(CLOCK_MONOTONIC, &start);
  ret = rknn3_model_init(ctx, &config);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_model_init failed! ret=%d\n", ret);
    free(model);
    free(weight_data);
    return -1;
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("rknn3_model_init success, cost %.3f ms\n", elapsed_ms(&start, &end));

  free(model);
  free(weight_data);
  return 0;
}

/** Load model and weight directly from file paths, then rknn3_model_init. */
static int load_model_path_mode(rknn3_context ctx, const DemoConfig* cfg)
{
  struct timespec start, end;
  uint32_t        core_mask = 0;

  printf("model_path=%s, weight_path=%s\n", cfg->model_path, cfg->weight_path);
  clock_gettime(CLOCK_MONOTONIC, &start);
  int ret = rknn3_load_model_from_path(ctx, cfg->model_path, cfg->weight_path);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_load_model_from_path failed! ret=%d\n", ret);
    return -1;
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("rknn3_load_model_from_path success%s, cost %.3f ms\n",
         cfg->use_decrypt_key ? " (encrypted model)" : "", elapsed_ms(&start, &end));

  if (resolve_core_mask(ctx, cfg, &core_mask) != 0) {
    return -1;
  }

  rknn3_config config;
  memset(&config, 0, sizeof(config));
  config.run_core_mask = core_mask;
  clock_gettime(CLOCK_MONOTONIC, &start);
  ret = rknn3_model_init(ctx, &config);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_model_init failed! ret=%d\n", ret);
    return -1;
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("rknn3_model_init success, cost %.3f ms\n", elapsed_ms(&start, &end));

  return 0;
}

// --- Tensor setup and teardown ---

/** Free per-shape input/output attribute arrays allocated in setup_demo_tensors. */
static void free_shape_infos(rknn3_shape_info* shape_infos, uint32_t n_shapes)
{
  if (!shape_infos) {
    return;
  }
  for (uint32_t i = 0; i < n_shapes; i++) {
    if (shape_infos[i].input_attrs) {
      free(shape_infos[i].input_attrs);
    }
    if (shape_infos[i].output_attrs) {
      free(shape_infos[i].output_attrs);
    }
  }
  free(shape_infos);
}

/** Release tensor memory, shape metadata, and numpy shape arrays. */
static void free_demo_tensors(rknn3_context ctx, DemoTensors* tensors,
                              std::vector<npy_array>& input_arrays,
                              std::vector<npy_array>& golden_arrays)
{
  if (!tensors) {
    return;
  }
  for (uint32_t i = 0; i < tensors->io_num.n_input; i++) {
    if (input_arrays[i].shape) {
      free(input_arrays[i].shape);
    }
  }
  for (uint32_t i = 0; i < tensors->io_num.n_output; i++) {
    if (golden_arrays[i].shape) {
      free(golden_arrays[i].shape);
    }
  }
  if (tensors->inputs) {
    for (uint32_t i = 0; i < tensors->io_num.n_input; i++) {
      if (tensors->inputs[i].attr) {
        free(tensors->inputs[i].attr);
      }
      if (tensors->inputs[i].mem) {
        rknn3_destroy_mem(ctx, tensors->inputs[i].mem);
      }
    }
    free(tensors->inputs);
    tensors->inputs = NULL;
  }
  if (tensors->outputs) {
    for (uint32_t i = 0; i < tensors->io_num.n_output; i++) {
      if (tensors->outputs[i].attr) {
        free(tensors->outputs[i].attr);
      }
      if (tensors->outputs[i].mem) {
        rknn3_destroy_mem(ctx, tensors->outputs[i].mem);
      }
    }
    free(tensors->outputs);
    tensors->outputs = NULL;
  }
  free_shape_infos(tensors->shape_infos, tensors->shape_config.n_shapes);
  tensors->shape_infos = NULL;
}

/** Free tensor resources allocated by setup_demo_tensors on failure paths. */
static void teardown_partial_tensors(rknn3_context ctx, DemoTensors* tensors)
{
  if (!tensors) {
    return;
  }
  if (tensors->inputs) {
    for (uint32_t i = 0; i < tensors->io_num.n_input; i++) {
      if (tensors->inputs[i].attr) {
        free(tensors->inputs[i].attr);
      }
      if (tensors->inputs[i].mem) {
        rknn3_destroy_mem(ctx, tensors->inputs[i].mem);
      }
    }
    free(tensors->inputs);
    tensors->inputs = NULL;
  }
  if (tensors->outputs) {
    for (uint32_t i = 0; i < tensors->io_num.n_output; i++) {
      if (tensors->outputs[i].attr) {
        free(tensors->outputs[i].attr);
      }
      if (tensors->outputs[i].mem) {
        rknn3_destroy_mem(ctx, tensors->outputs[i].mem);
      }
    }
    free(tensors->outputs);
    tensors->outputs = NULL;
  }
  if (tensors->shape_infos) {
    free_shape_infos(tensors->shape_infos, tensors->shape_config.n_shapes);
    tensors->shape_infos = NULL;
  }
}

/** Return max aligned_size for a tensor across all shape variants. */
static size_t max_tensor_aligned_size(const rknn3_shape_info* shape_infos, uint32_t n_shapes,
                                      bool is_input, uint32_t tensor_idx, uint32_t* core_id_out)
{
  size_t   max_size = 0;
  uint32_t core_id  = 0;
  for (uint32_t s = 0; s < n_shapes; s++) {
    const rknn3_tensor_attr* attr =
        is_input ? &shape_infos[s].input_attrs[tensor_idx] : &shape_infos[s].output_attrs[tensor_idx];
    if (attr->aligned_size > max_size) {
      max_size = attr->aligned_size;
      core_id  = attr->core_id;
    }
  }
  if (core_id_out) {
    *core_id_out = core_id;
  }
  return max_size;
}

/**
 * Query I/O and dynamic-shape info, activate @p shape_id, allocate device
 * memory for inputs/outputs using the largest aligned_size across shapes.
 */
static int setup_demo_tensors(rknn3_context ctx, int32_t shape_id, DemoTensors* tensors)
{
  // Query I/O tensor counts
  int ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &tensors->io_num, sizeof(tensors->io_num));
  if (ret != RKNN3_SUCCESS) {
    printf("Query input/output number failed! ret=%d\n", ret);
    return -1;
  }

  // Query dynamic shape configuration
  ret = rknn3_query(ctx, RKNN3_QUERY_DYNAMIC_SHAPE_CONFIG, &tensors->shape_config, sizeof(tensors->shape_config));
  if (ret != RKNN3_SUCCESS) {
    printf("Query dynamic shape config failed! ret=%d\n", ret);
    return -1;
  }

  printf("Model supports %d shape combinations\n", tensors->shape_config.n_shapes);
  printf("Current shape ID: %d\n", tensors->shape_config.current_shape_id);

  // Validate shape_id
  if (shape_id < 0 || shape_id >= (int32_t)tensors->shape_config.n_shapes) {
    printf("Invalid shape_id %d, model only supports %d shapes\n", shape_id, tensors->shape_config.n_shapes);
    return -1;
  }

  // Query all supported shape variants (static and dynamic)
  tensors->shape_infos = (rknn3_shape_info*)malloc(sizeof(rknn3_shape_info) * tensors->shape_config.n_shapes);
  if (!tensors->shape_infos) {
    printf("Failed to allocate memory for shape info\n");
    return -1;
  }

  // Allocate per-shape input/output attribute buffers
  for (uint32_t i = 0; i < tensors->shape_config.n_shapes; i++) {
    tensors->shape_infos[i].shape_id     = i;
    tensors->shape_infos[i].input_attrs  = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr) * tensors->io_num.n_input);
    tensors->shape_infos[i].output_attrs = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr) * tensors->io_num.n_output);

    if (!tensors->shape_infos[i].input_attrs || !tensors->shape_infos[i].output_attrs) {
      printf("Failed to allocate memory for shape attributes\n");
      // Roll back partial allocations
      for (uint32_t j = 0; j <= i; j++) {
        if (tensors->shape_infos[j].input_attrs)
          free(tensors->shape_infos[j].input_attrs);
        if (tensors->shape_infos[j].output_attrs)
          free(tensors->shape_infos[j].output_attrs);
      }
      free(tensors->shape_infos);
      tensors->shape_infos = NULL;
      return -1;
    }
  }

  // Fill shape_infos via RKNN3_QUERY_DYNAMIC_SHAPE_INFO
  ret = rknn3_query(ctx, RKNN3_QUERY_DYNAMIC_SHAPE_INFO, tensors->shape_infos, sizeof(rknn3_shape_info) * tensors->shape_config.n_shapes);
  if (ret != RKNN3_SUCCESS) {
    printf("Query shape info failed! ret=%d\n", ret);
    for (uint32_t i = 0; i < tensors->shape_config.n_shapes; i++) {
      if (tensors->shape_infos[i].input_attrs)
        free(tensors->shape_infos[i].input_attrs);
      if (tensors->shape_infos[i].output_attrs)
        free(tensors->shape_infos[i].output_attrs);
    }
    free(tensors->shape_infos);
    tensors->shape_infos = NULL;
    return -1;
  }

  // Log all shape variants
  for (uint32_t i = 0; i < tensors->shape_config.n_shapes; i++) {
    printf("Shape %d (ID: %d)%s:\n", i, tensors->shape_infos[i].shape_id, tensors->shape_infos[i].is_default ? " [Default]" : "");

    for (uint32_t j = 0; j < tensors->shape_infos[i].n_inputs; j++) {
      rknn3_tensor_attr* attr = &tensors->shape_infos[i].input_attrs[j];
      printf("  Input %d (%s): [", attr->index, attr->name);
      for (uint32_t k = 0; k < attr->n_dims; k++) {
        printf("%d%s", attr->shape[k], (k < attr->n_dims - 1) ? ", " : "");
      }
      printf("] Aligned size: %lu bytes\n", attr->aligned_size);
    }
  }

  // Activate the requested shape_id
  if (shape_id != (int32_t)tensors->shape_config.current_shape_id) {
    ret = rknn3_set_shape(ctx, shape_id);
    if (ret != RKNN3_SUCCESS) {
      printf("Set shape failed! ret=%d\n", ret);
      for (uint32_t i = 0; i < tensors->shape_config.n_shapes; i++) {
        if (tensors->shape_infos[i].input_attrs)
          free(tensors->shape_infos[i].input_attrs);
        if (tensors->shape_infos[i].output_attrs)
          free(tensors->shape_infos[i].output_attrs);
      }
      free(tensors->shape_infos);
      tensors->shape_infos = NULL;
      return -1;
    } else {
      printf("Active shape ID set to: %d\n", shape_id);
    }
  }

  printf("input tensors:\n");
  tensors->inputs = (rknn3_tensor*)calloc(tensors->io_num.n_input, sizeof(rknn3_tensor));
  if (!tensors->inputs) {
    printf("Failed to allocate input tensor array\n");
    teardown_partial_tensors(ctx, tensors);
    return -1;
  }
  for (uint32_t i = 0; i < tensors->io_num.n_input; i++) {
    tensors->inputs[i].attr        = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
    tensors->inputs[i].attr->index = i;

    uint32_t core_id  = 0;
    size_t   alloc_size = max_tensor_aligned_size(tensors->shape_infos, tensors->shape_config.n_shapes,
                                                  true, i, &core_id);
    tensors->inputs[i].mem = rknn3_create_mem(ctx, alloc_size, core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
    if (tensors->inputs[i].mem == NULL) {
      printf("rknn3_create_mem for input %d failed!\n", i);
      teardown_partial_tensors(ctx, tensors);
      return -1;
    }
    memcpy(tensors->inputs[i].attr, &tensors->shape_infos[shape_id].input_attrs[i], sizeof(rknn3_tensor_attr));
    dump_tensor_attr(tensors->inputs[i].attr);
  }

  printf("output tensors:\n");
  tensors->outputs = (rknn3_tensor*)calloc(tensors->io_num.n_output, sizeof(rknn3_tensor));
  if (!tensors->outputs) {
    printf("Failed to allocate output tensor array\n");
    teardown_partial_tensors(ctx, tensors);
    return -1;
  }
  for (uint32_t i = 0; i < tensors->io_num.n_output; i++) {
    tensors->outputs[i].attr        = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
    tensors->outputs[i].attr->index = i;

    uint32_t core_id    = 0;
    size_t   alloc_size = max_tensor_aligned_size(tensors->shape_infos, tensors->shape_config.n_shapes,
                                                  false, i, &core_id);
    tensors->outputs[i].mem = rknn3_create_mem(ctx, alloc_size, core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
    if (tensors->outputs[i].mem == NULL) {
      printf("rknn3_create_mem for output %d failed!\n", i);
      teardown_partial_tensors(ctx, tensors);
      return -1;
    }
    memcpy(tensors->outputs[i].attr, &tensors->shape_infos[shape_id].output_attrs[i], sizeof(rknn3_tensor_attr));
    dump_tensor_attr(tensors->outputs[i].attr);
  }


  return 0;
}

/** Free raw file buffers loaded by load_npy_files. */
static void free_npy_data_ptrs(const std::vector<void*>& input_data_ptrs,
                               const std::vector<void*>& golden_data_ptrs)
{
  for (void* ptr : input_data_ptrs) {
    free(ptr);
  }
  for (void* ptr : golden_data_ptrs) {
    free(ptr);
  }
}

/** Read one .npy file into memory, parse header, and track buffer for cleanup. */
static int load_one_npy_file(const std::string& path, npy_array* arr, std::vector<void*>& data_ptrs)
{
  void*  data = NULL;
  size_t size = 0;
  if (read_file_to_buffer(path.c_str(), &data, &size) != 0) {
    printf("Failed to read numpy file: %s\n", path.c_str());
    return -1;
  }
  data_ptrs.push_back(data);
  if (parse_npy_header_from_mem(data, size, arr) != 0) {
    printf("Failed to parse numpy header: %s\n", path.c_str());
    return -1;
  }
  return 0;
}

/** Load input .npy files into memory and parse their headers. */
static int load_input_npy_files(const DemoConfig& cfg, const rknn3_input_output_num& io_num,
                                std::vector<npy_array>& input_arrays, std::vector<void*>& input_data_ptrs)
{
  if (cfg.input_files.size() != io_num.n_input) {
    printf("Input file count mismatch: got %zu, model expects %u\n",
           cfg.input_files.size(), io_num.n_input);
    return -1;
  }
  for (uint32_t i = 0; i < io_num.n_input; i++) {
    if (load_one_npy_file(cfg.input_files[i], &input_arrays[i], input_data_ptrs) != 0) {
      return -1;
    }
  }
  return 0;
}

/** Load golden output .npy files into memory and parse their headers. */
static int load_golden_npy_files(const DemoConfig& cfg, const rknn3_input_output_num& io_num,
                                 std::vector<npy_array>& golden_arrays, std::vector<void*>& golden_data_ptrs)
{
  if (cfg.golden_files.size() != io_num.n_output) {
    printf("Golden file count mismatch: got %zu, model expects %u\n",
           cfg.golden_files.size(), io_num.n_output);
    return -1;
  }
  for (uint32_t i = 0; i < io_num.n_output; i++) {
    if (load_one_npy_file(cfg.golden_files[i], &golden_arrays[i], golden_data_ptrs) != 0) {
      return -1;
    }
  }
  return 0;
}

/**
 * Fill input device buffers with random values (or fixed values for special
 * tensor names like Th/Tc/Ts/Tsr).
 */
static int fill_random_input_data(DemoTensors& tensors)
{
  srand((unsigned)time(NULL));

  for (uint32_t in_idx = 0; in_idx < tensors.io_num.n_input; in_idx++) {
    rknn3_tensor_attr* input_attr = tensors.inputs[in_idx].attr;
    rknn3_tensor_mem*  input_mem  = tensors.inputs[in_idx].mem;
    uint32_t           dst_elems  = shape_count(input_attr->shape, input_attr->n_dims);

    printf("Generating random input data for input %d (name: %s)\n", in_idx, input_attr->name);

    bool  is_special_tensor = false;
    float special_value     = 0.0f;

    if (strstr(input_attr->name, "Th") != NULL && input_attr->n_dims == 1 && input_attr->shape[0] == 1) {
      is_special_tensor = true;
      special_value     = 0.0f;
      printf("Special tensor 'Th' detected, setting all values to 0\n");
    } else if (strstr(input_attr->name, "Tc") != NULL && input_attr->n_dims == 1 && input_attr->shape[0] == 1) {
      is_special_tensor = true;
      special_value     = 1.0f;
      for (uint32_t k = 0; k < tensors.io_num.n_input; k++) {
        rknn3_tensor_attr* attr = tensors.inputs[k].attr;
        if (strstr(attr->name, "attention_mask") != NULL && attr->n_dims >= 2) {
          special_value = (float)attr->shape[1];
          break;
        }
      }
      printf("Special tensor '%s' detected, setting all values to %d\n", input_attr->name, (int)special_value);
    } else if ((strstr(input_attr->name, "Ts") != NULL || strstr(input_attr->name, "Tsr") != NULL) &&
               input_attr->n_dims == 1 && input_attr->shape[0] == 1) {
      is_special_tensor = true;
      special_value     = 0.0f;
      printf("Special tensor '%s' detected, setting all values to 0\n", input_attr->name);
    } else if (strstr(input_attr->name, "num_logits_to_keep") != NULL &&
               input_attr->n_dims == 1 && input_attr->shape[0] == 1) {
      is_special_tensor = true;
      special_value     = 0.0f;
      for (uint32_t k = 0; k < tensors.io_num.n_input; k++) {
        rknn3_tensor_attr* attr = tensors.inputs[k].attr;
        if (strstr(attr->name, "attention_mask") != NULL && attr->n_dims >= 2) {
          special_value = (float)attr->shape[1] - 1.0f;
          break;
        }
      }
      printf("Special tensor '%s' detected, setting all values to %d\n", input_attr->name, (int)special_value);
    }

    switch (input_attr->dtype) {
    case RKNN3_TENSOR_FLOAT16: {
      float16* data = (float16*)input_mem->virt_addr;
      if (is_special_tensor) {
        for (uint32_t i = 0; i < dst_elems; i++) {
          data[i] = fp32_to_fp16(special_value);
        }
      } else {
        for (uint32_t i = 0; i < dst_elems; i++) {
          float random_val = (float)rand() / RAND_MAX * 2.0f - 1.0f;
          data[i]          = fp32_to_fp16(random_val);
        }
      }
      break;
    }
    case RKNN3_TENSOR_INT32: {
      int32_t* data = (int32_t*)input_mem->virt_addr;
      if (is_special_tensor) {
        for (uint32_t i = 0; i < dst_elems; i++) {
          data[i] = (int32_t)special_value;
        }
      } else {
        for (uint32_t i = 0; i < dst_elems; i++) {
          data[i] = (int32_t)(rand() % 2001 - 1000);
        }
      }
      break;
    }
    case RKNN3_TENSOR_INT8: {
      int8_t* data = (int8_t*)input_mem->virt_addr;
      if (is_special_tensor) {
        for (uint32_t i = 0; i < dst_elems; i++) {
          data[i] = (int8_t)special_value;
        }
      } else {
        for (uint32_t i = 0; i < dst_elems; i++) {
          data[i] = (int8_t)(rand() % 256 - 128);
        }
      }
      break;
    }
    case RKNN3_TENSOR_UINT8: {
      uint8_t* data = (uint8_t*)input_mem->virt_addr;
      if (is_special_tensor) {
        for (uint32_t i = 0; i < dst_elems; i++) {
          data[i] = (uint8_t)special_value;
        }
      } else {
        for (uint32_t i = 0; i < dst_elems; i++) {
          data[i] = (uint8_t)(rand() % 256);
        }
      }
      break;
    }
    default:
      printf("Unsupported tensor type for random generation: %s\n", rknn3_get_type_string(input_attr->dtype));
      return -1;
    }

    printf("Input %d size check passed: %u elements\n", in_idx, dst_elems);
    printf("Input[%d] first values (type: %d):\n", in_idx, input_attr->dtype);
    print_data_values((void*)input_mem->virt_addr, input_attr->dtype, input_mem->size, 10);
  }

  return 0;
}

// --- Input preparation ---

/**
 * Convert loaded numpy inputs into device tensor layout/dtype and copy into
 * rknn3_tensor_mem buffers.
 */
static int prepare_input_tensors(DemoTensors& tensors, std::vector<npy_array>& input_arrays)
{
  for (uint32_t in_idx = 0; in_idx < tensors.io_num.n_input; in_idx++) {
    rknn3_tensor_attr* input_attr = tensors.inputs[in_idx].attr;
    rknn3_tensor_mem*  input_mem  = tensors.inputs[in_idx].mem;

    uint32_t dst_elems = shape_count(input_attr->shape, input_attr->n_dims);
    size_t src_elems = 1;
      for (size_t i = 0; i < input_arrays[in_idx].ndim; i++)
      {
        src_elems *= input_arrays[in_idx].shape[i];
      }

      // Verify input size
      if (src_elems > dst_elems)
      {
        printf("Input %d size mismatch! Expected %u number of elements, got %lu number of elements\n", in_idx, dst_elems, src_elems);
        return -1;
      }

      int batch = input_arrays[in_idx].ndim > 0 ? input_arrays[in_idx].shape[0] : 1;
      int channel = input_arrays[in_idx].ndim > 1 ? input_arrays[in_idx].shape[1] : 1;
      int h = input_arrays[in_idx].ndim > 2 ? input_arrays[in_idx].shape[2] : 1;
      int w = input_arrays[in_idx].ndim > 3 ? input_arrays[in_idx].shape[3] : 1;
      int w_stride = w; // TODO: derive from tensor stride when needed

      if (input_arrays[in_idx].dtype == CNPY_TYPE_FLOAT32)
      {
        float *input_data = (float *)((char *)input_arrays[in_idx].raw_data + input_arrays[in_idx].data_begin);
        if (input_attr->layout == RKNN3_TENSOR_NC1HWC2)
        {
          if (input_attr->dtype == RKNN3_TENSOR_FLOAT16)
          {
            int sub_c = input_attr->n_dims > 4 ? input_attr->shape[4] : 1;
            int align_stride = w;
            int align_hw = input_attr->stride[1] / input_attr->stride[3];
            NCHW_fp32_to_NC1HWC2_fp16(input_data, (float16 *)input_mem->virt_addr, batch, h, w, channel, sub_c, align_stride, align_hw);
          }
          else if (input_attr->dtype == RKNN3_TENSOR_INT8)
          {
            int align_hw = input_attr->stride[1] / input_attr->stride[3];
            convert_fp32_NCHW_to_NC1HWC2(input_data, input_mem->virt_addr, batch, h, w, channel, w, align_hw, input_attr);
          }
          else
          {
            printf("Unsupported type for NC1HWC2 format: %s\n", rknn3_get_type_string(input_attr->dtype));
            return -1;
          }
        }
        else if (input_attr->layout == RKNN3_TENSOR_NHWC)
        {
          if (input_attr->dtype == RKNN3_TENSOR_FLOAT16)
          {
            NCHW_fp32_to_NHWC_fp16(input_data, (float16 *)input_mem->virt_addr, batch, h, w, channel, w_stride);
          }
          else
          {
            printf("Unsupported type for NHWC format: %s\n", rknn3_get_type_string(input_attr->dtype));
            return -1;
          }
        }
        else if (input_attr->layout == RKNN3_TENSOR_NCHW || input_attr->layout == RKNN3_TENSOR_UNDEFINED)
        {
          if (input_attr->dtype == RKNN3_TENSOR_INT8)
          {
            convert_fp32_to_int8(input_data, (int8_t *)input_mem->virt_addr, dst_elems, input_attr);
          }
          else
          {
            convert_fp32_to_any_type(input_data, input_mem->virt_addr, dst_elems, input_attr->dtype);
          }
        }
        else
        {
          printf("Unsupported format: %s\n", rknn3_get_layout_string(input_attr->layout));
          return -1;
        }
      }
      else if (input_arrays[in_idx].dtype == CNPY_TYPE_UINT8)
      {
        uint8_t *input_data = (uint8_t *)((char *)input_arrays[in_idx].raw_data + input_arrays[in_idx].data_begin);

        if (input_attr->layout == RKNN3_TENSOR_NHWC)
        {
          if (convert_uint8_NCHW_to_NHWC(input_data, input_mem->virt_addr, batch, h, w, channel,
                                         w_stride, input_attr) != 0) {
            return -1;
          }
        }
        else if (input_attr->layout == RKNN3_TENSOR_NC1HWC2)
        {
          int align_hw = input_attr->stride[1] / input_attr->stride[3];
          convert_uint8_NCHW_to_NC1HWC2(input_data, input_mem->virt_addr, batch, h, w, channel, w, align_hw, input_attr);
        }
        else
        {
          printf("Error: uint8 input only supports NHWC layout conversion, got: %s\n", rknn3_get_layout_string(input_attr->layout));
          return -1;
        }
      }
      else if (input_arrays[in_idx].dtype == CNPY_TYPE_INT32 || input_arrays[in_idx].dtype == CNPY_TYPE_UINT32)
      {
        uint8_t *input_data = (uint8_t *)((char *)input_arrays[in_idx].raw_data + input_arrays[in_idx].data_begin);
        memcpy(input_mem->virt_addr, input_data, src_elems * sizeof(int32_t));
      }
      else if (input_arrays[in_idx].dtype == CNPY_TYPE_BOOLEAN)
      {
        uint8_t *input_data = (uint8_t *)((char *)input_arrays[in_idx].raw_data + input_arrays[in_idx].data_begin);
        if (input_attr->layout == RKNN3_TENSOR_UNDEFINED)
        {
          memcpy(input_mem->virt_addr, input_data, dst_elems * sizeof(uint8_t));
        }
        else
        {
          printf("Error: bool input only supports undefine layout conversion, got: %s\n", rknn3_get_layout_string(input_attr->layout));
          return -1;
        }
      }
      else
      {
        printf("Unsupported input numpy data type: %d\n", input_arrays[in_idx].dtype);
        return -1;
      }

    printf("Input %d size check passed: %u elements\n", in_idx, dst_elems);

    // Log first 10 input values after conversion
    printf("Input[%d] first values (type: %d):\n", in_idx, input_attr->dtype);
    print_data_values((void*)input_mem->virt_addr, input_attr->dtype, input_mem->size, 10);
  }


  return 0;
}

/** Fill model inputs: random data or load .npy files and convert to device tensors. */
static int prepare_inputs(const DemoConfig& cfg, DemoTensors& tensors,
                                std::vector<npy_array>& input_arrays,
                                std::vector<void*>& input_data_ptrs)
{
  if (cfg.use_random_input) {
    return fill_random_input_data(tensors);
  }

  if (load_input_npy_files(cfg, tensors.io_num, input_arrays, input_data_ptrs) != 0) {
    return -1;
  }
  return prepare_input_tensors(tensors, input_arrays);
}

/** Load golden output .npy files for post-inference comparison. */
static int load_golden_outputs(const DemoConfig& cfg, const DemoTensors& tensors,
                               std::vector<npy_array>& golden_arrays,
                               std::vector<void*>& golden_data_ptrs)
{
  if (cfg.skip_golden_comparison) {
    return 0;
  }
  return load_golden_npy_files(cfg, tensors.io_num, golden_arrays, golden_data_ptrs);
}

// --- Inference ---

/**
 * Run one inference: sync inputs to device, rknn3_run, sync outputs back.
 * Populates @p timing when non-NULL.
 */
static int run_model_inference(rknn3_context ctx, DemoTensors* tensors, InferenceTiming* timing)
{
  struct timespec start, end;
  int             ret = RKNN3_SUCCESS;

  if (timing) {
    memset(timing, 0, sizeof(*timing));
  }

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < (int)tensors->io_num.n_input; i++) {
    ret = rknn3_mem_sync(ctx, tensors->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_mem_sync input %d failed! ret=%d\n", i, ret);
      return ret;
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  if (timing) {
    timing->sync_in_us = timespec_elapsed_us(&start, &end);
  }

  clock_gettime(CLOCK_MONOTONIC, &start);
  ret = rknn3_run(ctx, tensors->inputs, tensors->io_num.n_input, tensors->outputs, tensors->io_num.n_output);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_run failed! ret=%d\n", ret);
    return ret;
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  if (timing) {
    timing->run_us = timespec_elapsed_us(&start, &end);
  }

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < (int)tensors->io_num.n_output; i++) {
    ret = rknn3_mem_sync(ctx, tensors->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_mem_sync output %d failed! ret=%d\n", i, ret);
      return ret;
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  if (timing) {
    timing->sync_out_us = timespec_elapsed_us(&start, &end);
  }

  return RKNN3_SUCCESS;
}

// --- Golden output verification ---

static const size_t kOutputTopK = 5;

/** Print top-k values (by magnitude) for one output tensor. */
static void print_output_top5(const rknn3_tensor* out_tensor, uint32_t output_idx, size_t top_k)
{
  if (!out_tensor || !out_tensor->mem || !out_tensor->attr) {
    printf("Output %u: no data\n", output_idx);
    return;
  }

  rknn3_tensor_attr* attr    = out_tensor->attr;
  uint32_t           n_elems = shape_count(attr->shape, attr->n_dims);
  if (n_elems == 0) {
    printf("Output %u: empty tensor\n", output_idx);
    return;
  }

  std::vector<float> fp32_data(n_elems);
  if (attr->dtype == RKNN3_TENSOR_FLOAT32) {
    memcpy(fp32_data.data(), out_tensor->mem->virt_addr, (size_t)n_elems * sizeof(float));
  } else if (attr->dtype == RKNN3_TENSOR_INT8 && attr->qnt_type != RKNN3_TENSOR_QNT_NONE) {
    convert_int8_to_fp32(out_tensor->mem->virt_addr, fp32_data.data(), (int)n_elems, attr->dtype,
                         attr->qnt_info.scale, attr->qnt_info.zero_point);
  } else if (convert_any_type_to_fp32(out_tensor->mem->virt_addr, fp32_data.data(), (int)n_elems, attr->dtype) != 0) {
    printf("Output %u: unsupported dtype %s for top-%zu\n", output_idx,
           rknn3_get_type_string(attr->dtype), top_k);
    return;
  }

  std::vector<size_t> indices(n_elems);
  for (size_t j = 0; j < n_elems; ++j) {
    indices[j] = j;
  }

  size_t k = top_k < n_elems ? top_k : n_elems;
  std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                    [&](size_t a, size_t b) { return fp32_data[a] > fp32_data[b]; });

  printf("Output %u top%zu:\n", output_idx, top_k);
  for (size_t t = 0; t < k; ++t) {
    size_t idx = indices[t];
    printf("  [%zu]: %.6f\n", idx, fp32_data[idx]);
  }
}

/** Print top-k values for every model output. */
static void print_outputs_top5(const DemoTensors& tensors, size_t top_k)
{
  for (uint32_t i = 0; i < tensors.io_num.n_output; ++i) {
    print_output_top5(&tensors.outputs[i], i, top_k);
  }
}

/** Convert golden numpy array payload to FP32 for comparison. */
static int golden_npy_to_fp32(const npy_array& arr, float* dst, int n_elems)
{
  const char* raw = (const char*)arr.raw_data + arr.data_begin;
  switch (arr.dtype) {
  case CNPY_TYPE_FLOAT32:
    memcpy(dst, raw, (size_t)n_elems * sizeof(float));
    return 0;
  case CNPY_TYPE_FLOAT64:
    for (int i = 0; i < n_elems; i++) {
      dst[i] = (float)((const double*)raw)[i];
    }
    return 0;
  case CNPY_TYPE_INT8:
    for (int i = 0; i < n_elems; i++) {
      dst[i] = (float)((const int8_t*)raw)[i];
    }
    return 0;
  case CNPY_TYPE_UINT8:
    for (int i = 0; i < n_elems; i++) {
      dst[i] = (float)((const uint8_t*)raw)[i];
    }
    return 0;
  case CNPY_TYPE_INT16:
    for (int i = 0; i < n_elems; i++) {
      dst[i] = (float)((const int16_t*)raw)[i];
    }
    return 0;
  case CNPY_TYPE_INT32:
    for (int i = 0; i < n_elems; i++) {
      dst[i] = (float)((const int32_t*)raw)[i];
    }
    return 0;
  default:
    printf("Unsupported golden numpy dtype %d\n", arr.dtype);
    return -1;
  }
}

/** Convert a native output tensor to a contiguous FP32 buffer and logical shape. */
static int prepare_output_tensor_as_fp32(const rknn3_tensor* output, const npy_array* golden,
                                         float** output_data, size_t* output_shape,
                                         size_t* output_ndim, size_t* output_elems)
{
  if (!output || !output->attr || !output->mem || !output_data || !output_shape ||
      !output_ndim || !output_elems) {
    return -1;
  }

  const rknn3_tensor_attr* attr = output->attr;
  bool use_golden = golden && golden->shape && golden->ndim > 0;
  size_t ndim = 0;
  size_t elems = 1;

  if (use_golden) {
    if (golden->ndim > RKNN3_MAX_DIMS) {
      printf("Golden output dims %zu exceed max supported dims %d\n", golden->ndim, RKNN3_MAX_DIMS);
      return -1;
    }
    ndim = golden->ndim;
    for (size_t i = 0; i < ndim; i++) {
      output_shape[i] = golden->shape[i];
      elems *= output_shape[i];
    }
  } else if (attr->layout == RKNN3_TENSOR_NC1HWC2) {
    if (attr->n_dims < 5) {
      printf("NC1HWC2 output %s has invalid dims: %u\n", attr->name, attr->n_dims);
      return -1;
    }
    ndim = 4;
    output_shape[0] = attr->shape[0];
    output_shape[1] = (size_t)attr->shape[1] * attr->shape[4];
    output_shape[2] = attr->shape[2];
    output_shape[3] = attr->shape[3];
    elems = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];
  } else {
    ndim = attr->n_dims ? attr->n_dims : 1;
    if (attr->n_dims == 0) {
      output_shape[0] = attr->n_elems;
      elems = attr->n_elems;
    } else {
      for (size_t i = 0; i < ndim; i++) {
        output_shape[i] = attr->shape[i];
        elems *= output_shape[i];
      }
    }
  }

  if (elems == 0) {
    printf("Output tensor %s has zero elements\n", attr->name);
    return -1;
  }

  float* data = (float*)malloc(elems * sizeof(float));
  if (!data) {
    printf("Failed to allocate %zu floats for output %s\n", elems, attr->name);
    return -1;
  }

  int convert_ret = 0;
  if (attr->layout == RKNN3_TENSOR_NC1HWC2) {
    int batch = (int)output_shape[0];
    int channel = ndim > 1 ? (int)output_shape[1] : 1;
    int h = ndim > 2 ? (int)output_shape[2] : 1;
    int w = ndim > 3 ? (int)output_shape[3] : 1;
    int C1 = attr->shape[1];
    int C2 = attr->shape[4];
    int align_hw = h * w;
    if (attr->n_stride > 3 && attr->stride[3] != 0) {
      align_hw = attr->stride[1] / attr->stride[3];
    }

    if (attr->dtype == RKNN3_TENSOR_FLOAT16) {
      convert_ret = NC1HWC2_fp16_to_NCHW_fp32((float16*)output->mem->virt_addr, data,
                                               batch, C1, C2, align_hw, channel, h, w);
    } else if (attr->dtype == RKNN3_TENSOR_INT8) {
      convert_ret = NC1HWC2_int8_to_NCHW_fp32((int8_t*)output->mem->virt_addr, data,
                                               batch, C1, C2, align_hw, channel, h, w,
                                               (rknn3_tensor_attr*)attr);
    } else if (attr->dtype == RKNN3_TENSOR_INT32) {
      convert_ret = NC1HWC2_int32_to_NCHW_fp32((int32_t*)output->mem->virt_addr, data,
                                                batch, C1, C2, align_hw, channel, h, w);
    } else {
      printf("Unsupported NC1HWC2 output dtype: %s\n", rknn3_get_type_string(attr->dtype));
      convert_ret = -1;
    }
  } else if (attr->dtype == RKNN3_TENSOR_INT8) {
    convert_ret = convert_int8_to_fp32(output->mem->virt_addr, data, (int)elems, attr->dtype,
                                       attr->qnt_info.scale, attr->qnt_info.zero_point);
  } else {
    convert_ret = convert_any_type_to_fp32(output->mem->virt_addr, data, (int)elems, attr->dtype);
  }

  if (convert_ret != 0) {
    free(data);
    return -1;
  }

  *output_data = data;
  *output_ndim = ndim;
  *output_elems = elems;
  return 0;
}

static int save_output_as_npy(uint32_t index, float* data, size_t elems,
                              size_t* shape, size_t ndim)
{
  char filename[64];
  snprintf(filename, sizeof(filename), "rt_output%u.npy", index);
  int ret = npy_save_float_buffer_to_file(filename, data, elems, shape, ndim);
  if (ret != 0) {
    printf("Failed to save output[%u] to %s\n", index, filename);
    return -1;
  }
  printf("Saved output[%u] to %s\n", index, filename);
  return 0;
}

static void save_outputs_as_npy(DemoTensors& tensors, const std::vector<npy_array>* golden_arrays)
{
  for (uint32_t i = 0; i < tensors.io_num.n_output; i++) {
    size_t shape[RKNN3_MAX_DIMS] = {0};
    size_t ndim = 0;
    size_t elems = 0;
    float* data = NULL;
    const npy_array* golden = golden_arrays ? &(*golden_arrays)[i] : NULL;
    if (prepare_output_tensor_as_fp32(&tensors.outputs[i], golden, &data, shape, &ndim, &elems) != 0) {
      printf("Failed to prepare output[%u] for numpy dump\n", i);
      continue;
    }
    save_output_as_npy(i, data, elems, shape, ndim);
    free(data);
  }
}

/**
 * Convert device outputs to FP32, save as output_data_*.npy, print sample
 * values, and report cosine similarity against golden .npy files.
 */
static void verify_outputs_against_golden(DemoTensors& tensors, std::vector<npy_array>& golden_arrays)
{
    for (uint32_t i = 0; i < tensors.io_num.n_output; i++)
    {
      rknn3_tensor_attr *output_attr = tensors.outputs[i].attr;
      rknn3_tensor_mem *output_mem = tensors.outputs[i].mem;

      // rt_hw_cpu_dcache_ops(RT_HW_CACHE_INVALIDATE, (void*)output_mem->virt_addr, output_mem->size);

      int batch = golden_arrays[i].ndim > 0 ? golden_arrays[i].shape[0] : 1;
      int channel = golden_arrays[i].ndim > 1 ? golden_arrays[i].shape[1] : 1;
      int h = golden_arrays[i].ndim > 2 ? golden_arrays[i].shape[2] : 1;
      int w = golden_arrays[i].ndim > 3 ? golden_arrays[i].shape[3] : 1;
      int src_elems = shape_count(output_attr->shape, output_attr->n_dims);
      int dst_elems = 1;
      for (uint32_t j = 0; j < golden_arrays[i].ndim; j++)
      {
        dst_elems *= golden_arrays[i].shape[j];
      }

      float *output_data = (float *)malloc(dst_elems * sizeof(float));
      if (output_data == NULL)
      {
        printf("Failed to allocate memory for output_data\n");
        continue;
      }

      int convert_ok = 0;
      if (output_attr->layout == RKNN3_TENSOR_NC1HWC2)
      {
        if (output_attr->dtype == RKNN3_TENSOR_FLOAT16)
        {
          int C1 = output_attr->shape[1];
          int C2 = output_attr->shape[4];
          int hw = output_attr->stride[1] / output_attr->stride[3];
          NC1HWC2_fp16_to_NCHW_fp32((float16 *)output_mem->virt_addr, output_data, batch, C1, C2, hw, channel, h, w);
          convert_ok = 1;
        }
        else if (output_attr->dtype == RKNN3_TENSOR_INT8)
        {
          int C1 = output_attr->shape[1];
          int C2 = output_attr->shape[4];
          int hw = output_attr->stride[1] / output_attr->stride[3];
          NC1HWC2_int8_to_NCHW_fp32((int8_t *)output_mem->virt_addr, output_data, batch, C1, C2, hw, channel, h, w, output_attr);
          convert_ok = 1;
        }
        else
        {
          printf("Unsupported type for NC1HWC2 format: %s\n", rknn3_get_type_string(output_attr->dtype));
        }
      }
      else if (output_attr->layout == RKNN3_TENSOR_NCHW || output_attr->layout == RKNN3_TENSOR_UNDEFINED)
      {
        if (output_attr->dtype == RKNN3_TENSOR_INT8)
        {
          float scale = output_attr->qnt_info.scale;
          int32_t zero_point = output_attr->qnt_info.zero_point;
          convert_int8_to_fp32(output_mem->virt_addr, output_data, dst_elems, output_attr->dtype, scale, zero_point);
          convert_ok = 1;
        }
        else if (convert_any_type_to_fp32(output_mem->virt_addr, output_data, dst_elems, output_attr->dtype) == 0)
        {
          convert_ok = 1;
        }
      }
      else
      {
        printf("Unsupported output format: %s\n", rknn3_get_layout_string(output_attr->layout));
      }

      if (!convert_ok) {
        free(output_data);
        continue;
      }

      float *golden_fp32 = (float *)malloc(dst_elems * sizeof(float));
      if (golden_fp32 == NULL)
      {
        printf("Failed to allocate memory for golden FP32 buffer\n");
        free(output_data);
        continue;
      }
      if (golden_npy_to_fp32(golden_arrays[i], golden_fp32, dst_elems) != 0)
      {
        free(output_data);
        free(golden_fp32);
        continue;
      }

      // Compare first 10 elements against golden
      int first_count = dst_elems < 10 ? dst_elems : 10;
      printf("\nComparing first %d values of Output %d:\n", first_count, i);
      for (int j = 0; j < first_count; j++)
      {
        printf("Index[%d]: Output=%.5f, Golden=%.5f\n", j, output_data[j], golden_fp32[j]);
      }

      // Compare last 10 elements against golden
      int last_start = dst_elems > 10 ? dst_elems - 10 : 0;
      printf("\nComparing last %d values of Output %d:\n", dst_elems - last_start, i);
      for (int j = last_start; j < dst_elems; j++)
      {
        printf("Index[%d]: Output=%.5f, Golden=%.5f\n", j, output_data[j], golden_fp32[j]);
      }

      printf("output native elems: %d, golden elems: %d\n", src_elems, dst_elems);

      // Cosine similarity over full output
      float similarity = cosine_similarity(output_data, golden_fp32, dst_elems);
      float distance = euclidean_distance(output_data, golden_fp32, dst_elems);
      printf("Output %d cosine similarity: %.5f\n", i, similarity);
      printf("Output %d euclidean distance: %.5f\n", i, distance);
      free(output_data);
      free(golden_fp32);
    }
 
}

// --- Entry point ---

int main(int argc, char* argv[])
{
  int           ret        = -1;
  DemoConfig    cfg;
  DemoTensors   tensors;
  rknn3_context ctx;
  bool          ctx_inited = false;
  std::vector<rknn3_tensor_mem*> kvcache_mems;

  std::vector<npy_array> input_arrays;
  std::vector<npy_array> golden_arrays;
  std::vector<void*>     input_data_ptrs;
  std::vector<void*>     golden_data_ptrs;

  memset(&tensors, 0, sizeof(tensors));

  // Phase 1: parse CLI and initialize RKNN3 context
  if (parse_demo_config(argc, argv, &cfg) != 0) {
    goto cleanup;
  }
  if (init_rknn3_context(&ctx, &cfg) != 0) {
    goto cleanup;
  }
  ctx_inited = true;

  // Phase 2: load model (streaming > data > path)
  if (cfg.chunk_size_mb > 0) {
    if (load_model_streaming_mode(ctx, &cfg) != 0) {
      goto cleanup;
    }
  } else if (cfg.load_mode == MODEL_LOAD_DATA) {
    if (load_model_from_data_mode(ctx, &cfg) != 0) {
      goto cleanup;
    }
  } else if (load_model_path_mode(ctx, &cfg) != 0) {
    goto cleanup;
  }

  if (setup_kvcache_mems(ctx, &kvcache_mems) != RKNN3_SUCCESS) {
    goto cleanup;
  }

  // Phase 3: allocate I/O tensors for the selected shape
  if (setup_demo_tensors(ctx, cfg.shape_id, &tensors) != 0) {
    goto cleanup;
  }

  input_arrays.resize(tensors.io_num.n_input);
  golden_arrays.resize(tensors.io_num.n_output);

  // Phase 4: fill inputs (random or from .npy)
  if (prepare_inputs(cfg, tensors, input_arrays, input_data_ptrs) != 0) {
    goto cleanup;
  }
  if (load_golden_outputs(cfg, tensors, golden_arrays, golden_data_ptrs) != 0) {
    goto cleanup;
  }

  // Phase 5: inference loop with per-stage timing
  {
    InferenceTiming total_timing;
    std::vector<uint64_t> baseline_output_checksums(tensors.io_num.n_output, 0);
    bool output_checksum_initialized = false;
    memset(&total_timing, 0, sizeof(total_timing));

    printf("Running model %d times...\n", cfg.loop_count);
    for (int loop = 0; loop < cfg.loop_count; loop++) {
      InferenceTiming stage_timing;
      ret = run_model_inference(ctx, &tensors, &stage_timing);
      if (ret != RKNN3_SUCCESS) {
        printf("Run failed at loop %d, ret=%d\n", loop + 1, ret);
        goto cleanup;
      }

      if (cfg.enable_output_checksum) {
        for (uint32_t out_idx = 0; out_idx < tensors.io_num.n_output; out_idx++) {
          rknn3_tensor_mem* output_mem = tensors.outputs[out_idx].mem;
          uint64_t checksum = calc_checksum_fnv1a64(output_mem->virt_addr, output_mem->size);
          if (!output_checksum_initialized) {
            baseline_output_checksums[out_idx] = checksum;
            printf("Loop %d output[%u] baseline checksum: 0x%016llx\n", loop + 1, out_idx,
                   (unsigned long long)checksum);
          } else if (checksum != baseline_output_checksums[out_idx]) {
            printf("ERROR: output checksum mismatch at loop %d, output[%u], expected=0x%016llx, actual=0x%016llx\n",
                   loop + 1, out_idx, (unsigned long long)baseline_output_checksums[out_idx],
                   (unsigned long long)checksum);
            ret = RKNN3_ERR_FAIL;
            goto cleanup;
          }
        }
        output_checksum_initialized = true;
      }

      total_timing.sync_in_us  += stage_timing.sync_in_us;
      total_timing.run_us      += stage_timing.run_us;
      total_timing.sync_out_us += stage_timing.sync_out_us;
      printf("loop %d: input sync %.3f ms, run %.3f ms, output sync %.3f ms, total %.3f ms\n",
             loop + 1,
             stage_timing.sync_in_us / 1000.0,
             stage_timing.run_us / 1000.0,
             stage_timing.sync_out_us / 1000.0,
             (stage_timing.sync_in_us + stage_timing.run_us + stage_timing.sync_out_us) / 1000.0);
    }

    {
      uint64_t loop_total_us = total_timing.sync_in_us + total_timing.run_us + total_timing.sync_out_us;

      printf("All %d loops completed successfully\n", cfg.loop_count);
      printf("Average input sync time: %.3f ms\n", total_timing.sync_in_us / (cfg.loop_count * 1000.0));
      printf("Average model run time: %.3f ms\n", total_timing.run_us / (cfg.loop_count * 1000.0));
      printf("Average output sync time: %.3f ms\n", total_timing.sync_out_us / (cfg.loop_count * 1000.0));
      printf("Average total time per loop: %.3f ms\n", loop_total_us / (cfg.loop_count * 1000.0));
    }

    // Phase 6: save last-loop outputs, then optionally compare against golden data.
    save_outputs_as_npy(tensors, cfg.skip_golden_comparison ? NULL : &golden_arrays);
    if (!cfg.skip_golden_comparison) {
      verify_outputs_against_golden(tensors, golden_arrays);
    } else {
      print_outputs_top5(tensors, kOutputTopK);
    }
  }

  ret = 0;

cleanup:
  // Release numpy file buffers, tensor memory, and RKNN3 context
  free_npy_data_ptrs(input_data_ptrs, golden_data_ptrs);
  if (ctx_inited) {
    free_demo_tensors(ctx, &tensors, input_arrays, golden_arrays);
    cleanup_kvcache_mems(ctx, &kvcache_mems);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    rknn3_destroy(ctx);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("rknn3_destroy success, cost %.3f ms\n", elapsed_ms(&start, &end));
  }

  return ret == 0 ? 0 : -1;
}
