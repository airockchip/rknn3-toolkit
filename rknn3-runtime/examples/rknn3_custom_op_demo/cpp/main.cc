/*
 * main.cc - RKNN3 custom op demo.
 *
 * Loads a RKNN model containing RknnCustomOpExam, registers the custom op plugin,
 * runs inference, and compares the output with the ONNX reference output
 * via cosine similarity.
 *
 *
 * Usage:
 *   ./rknn3_custom_op_demo <model_path> <weight_path> <ref_input.npy> <ref_output.npy> <core_mask> <plugin_path>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <vector>
#include <string>

#include "rknn3_api.h"
#include "float16.h"

/* ------------------------------------------------------------------ */
/*  App context                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    rknn3_tensor *inputs;
    rknn3_tensor *outputs;

    /* logical NCHW input dims (from input attr orig_shape) */
    int model_batch;
    int model_channel;
    int model_height;
    int model_width;
} rknn_app_context_t;

/* ------------------------------------------------------------------ */
/*  Layout conversion helpers                                         */
/*  Kept structurally identical to plugin/rknn_custom_op_impl.c so    */
/*  the two can be read side-by-side. Only the dtype differs: here    */
/*  fp32<->fp16 (host numpy side), there fp16<->fp16 (device side).   */
/* ------------------------------------------------------------------ */

/* Identical to plugin/rknn_custom_op_impl.c:28 */
#define ALIGNED_SIZE(x, bytes) (((x) + (bytes - 1)) & (~(bytes - 1)))

/* NC1HWC2 (N, C1, H, W, C2) -> NCHW (N, C, H, W), element by element.
 *   cn        = C2 (channels per C1 plane)
 *   channel   = logical C (= C1 * C2)
 *   align_stride = ALIGNED_SIZE(W, 1)
 *   align_hw   = ALIGNED_SIZE(W*H, 4)  (surface alignment)
 * src is fp16, dst is fp32.
 */
static int NC1HWC2_fp16_to_NCHW_fp32(const float16 *src, float *dst, int cn, int channel,
                                     int h, int w, int align_stride, int align_hw)
{
  if (!src || !dst) return -1;
  size_t idx = 0;
  for (int c = 0; c < channel; ++c) {
    int            plane  = c / cn;
    const float16 *src_c  = (size_t)plane * align_hw * cn + src;
    int            offset = c % cn;
    for (int cur_h = 0; cur_h < h; ++cur_h) {
      for (int cur_w = 0; cur_w < w; ++cur_w) {
        size_t cur_hw = (size_t)cur_h * align_stride + cur_w;
        dst[idx++] = fp16_to_fp32(src_c[(size_t)cn * cur_hw + offset]);
      }
    }
  }
  return 0;
}

/* NC1HWC2 (packed) -> NCHW fp32 for a whole batch.
 *   dims[0..4] = [N, C1, H, W, C2]  (i.e. attr->shape, NC1HWC2 is 5-dim)
 *   channel    = logical C  (use attr->orig_shape[1])
 */
static int nc1hwc2_to_nchw_fp32(const void *src, float *dst, const uint32_t *dims, int channel)
{
  if (!src || !dst || !dims) return -1;
  size_t batch        = dims[0];
  size_t height       = dims[2];
  size_t width        = dims[3];
  size_t cn           = dims[4];
  size_t align_surf   = width * height == 1 ? 1 : ALIGNED_SIZE(width * height, 4);
  size_t align_stride = ALIGNED_SIZE(width, 1);

  size_t src_batch_elems = (size_t)dims[1] * align_surf * dims[4];
  size_t dst_batch_elems = (size_t)channel * width * height;
  const float16 *src_ptr = (const float16 *)src;
  float         *dst_ptr = dst;
  for (size_t i = 0; i < batch; ++i) {
    NC1HWC2_fp16_to_NCHW_fp32(src_ptr, dst_ptr, (int)cn, channel, (int)height, (int)width,
                              (int)align_stride, (int)align_surf);
    src_ptr += src_batch_elems;
    dst_ptr += dst_batch_elems;
  }
  return 0;
}

/* NCHW (N, C, H, W) -> NC1HWC2 (N, C1, H, W, C2), element by element. */
static int NCHW_fp32_to_NC1HWC2_fp16_e(const float *src, float16 *dst, int cn, int channel,
                                       int h, int w, int align_stride, int align_hw)
{
  if (!src || !dst) return -1;
  for (int c = 0; c < channel; ++c) {
    int      plane  = c / cn;
    int      offset = c % cn;
    float16 *dst_c  = dst + (size_t)plane * align_hw * cn;
    for (int cur_h = 0; cur_h < h; ++cur_h) {
      for (int cur_w = 0; cur_w < w; ++cur_w) {
        size_t cur_hw              = (size_t)cur_h * align_stride + cur_w;
        dst_c[(size_t)cn * cur_hw + offset] = fp32_to_fp16(src[(size_t)c * h * w + (size_t)cur_h * w + cur_w]);
      }
    }
  }
  return 0;
}

/* NCHW fp32 -> NC1HWC2 (packed) fp16 for a whole batch.
 *   dims[0..4] = [N, C1, H, W, C2]
 *   channel    = logical C
 */
static int nchw_fp32_to_nc1hwc2_fp16(const float *src, void *dst, const uint32_t *dims, int channel)
{
  if (!src || !dst || !dims) return -1;
  size_t batch        = dims[0];
  size_t height       = dims[2];
  size_t width        = dims[3];
  size_t cn           = dims[4];
  size_t align_surf   = width * height == 1 ? 1 : ALIGNED_SIZE(width * height, 4);
  size_t align_stride = ALIGNED_SIZE(width, 1);

  size_t align_c          = ((size_t)channel + cn - 1) / cn * cn;
  size_t src_batch_elems  = (size_t)channel * width * height;
  size_t dst_batch_elems  = (size_t)dims[1] * align_surf * cn;
  /* zero the whole dst so padding channels are 0 */
  memset(dst, 0, batch * align_c * align_surf * sizeof(float16));
  const float *src_ptr = src;
  float16     *dst_ptr = (float16 *)dst;
  for (size_t i = 0; i < batch; ++i) {
    NCHW_fp32_to_NC1HWC2_fp16_e(src_ptr, dst_ptr, (int)cn, channel, (int)height, (int)width,
                                (int)align_stride, (int)align_surf);
    src_ptr += src_batch_elems;
    dst_ptr += dst_batch_elems;
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/*  minimal npy reader (float32 only)                                 */
/* ------------------------------------------------------------------ */

static int read_npy_f32(const char *path, std::vector<float> &data,
                        std::vector<int> &shape)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        return -1;
    }

    char magic[6];
    if (fread(magic, 1, 6, f) != 6) {
        fprintf(stderr, "Short read: magic for %s\n", path);
        fclose(f);
        return -1;
    }
    if (memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "Not a npy file: %s\n", path);
        fclose(f);
        return -1;
    }

    unsigned char version[2];
    if (fread(version, 1, 2, f) != 2) {
        fprintf(stderr, "Short read: version for %s\n", path);
        fclose(f);
        return -1;
    }

    /* numpy v1.x: header_len is 2 bytes (uint16); v2.x: 4 bytes (uint32) */
    uint32_t header_len;
    if (version[0] >= 2) {
        uint32_t hl;
        if (fread(&hl, 1, 4, f) != 4) {
            fprintf(stderr, "Short read: header_len for %s\n", path);
            fclose(f);
            return -1;
        }
        header_len = hl;
    } else {
        unsigned short hl;
        if (fread(&hl, 1, 2, f) != 2) {
            fprintf(stderr, "Short read: header_len for %s\n", path);
            fclose(f);
            return -1;
        }
        header_len = hl;
    }

    char header[4096];
    /* index header_len is used for the terminating NUL, so header_len must be
     * strictly less than sizeof(header) to avoid an out-of-bounds write. */
    if (header_len == 0 || header_len >= sizeof(header)) {
        fprintf(stderr, "Invalid header_len=%u for %s\n", header_len, path);
        fclose(f);
        return -1;
    }

    size_t nread = fread(header, 1, header_len, f);
    if (nread != header_len) {
        fprintf(stderr, "Short read: got %zu, expected %u for %s\n", nread, header_len, path);
        fclose(f);
        return -1;
    }
    header[header_len] = '\0';

    shape.clear();
    char *p = strstr(header, "'shape'");
    if (p) {
        p = strchr(p, '(');
        if (p) {
            p++;
            while (*p && *p != ')') {
                /* skip non-digit characters */
                while (*p && *p != ')' && !(*p >= '0' && *p <= '9')) p++;
                if (*p == ')' || !*p) break;
                int dim = 0;
                if (sscanf(p, "%d", &dim) == 1) {
                    shape.push_back(dim);
                    while (*p && (*p >= '0' && *p <= '9')) p++;
                } else {
                    p++;
                }
            }
        }
    }

    if (shape.empty()) {
        fprintf(stderr, "Failed to parse shape from %s\n", path);
        fclose(f);
        return -1;
    }

    size_t total = 1;
    for (size_t i = 0; i < shape.size(); i++) total *= shape[i];

    data.resize(total);
    size_t nread2 = fread(data.data(), sizeof(float), total, f);
    fclose(f);
    if (nread2 != total) {
        fprintf(stderr, "Short read: data got %zu, expected %zu for %s\n", nread2, total, path);
        return -1;
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/*  cosine similarity                                                 */
/* ------------------------------------------------------------------ */

static float cosine_similarity(const float *a, const float *b, size_t n)
{
    double dot = 0, norm_a = 0, norm_b = 0;
    for (size_t i = 0; i < n; i++) {
        dot   += (double)a[i] * b[i];
        norm_a += (double)a[i] * a[i];
        norm_b += (double)b[i] * b[i];
    }
    if (norm_a < 1e-12 || norm_b < 1e-12) return 0.0f;
    return (float)(dot / (sqrt(norm_a) * sqrt(norm_b)));
}

/* ------------------------------------------------------------------ */
/*  init / inference / release                                        */
/* ------------------------------------------------------------------ */

/* Init: load model, register plugin, query attrs, allocate I/O memory. */
static int init_custom_op_model(const char *model_path, const char *weight_path,
                                const char *plugin_path, uint32_t core_mask,
                                const std::vector<int> &input_shape,
                                rknn_app_context_t *app_ctx)
{
    int ret;
    rknn3_context ctx = 0;
    rknn3_input_output_num io_num = {0};
    rknn3_tensor_attr *input_attrs  = NULL;
    rknn3_tensor_attr *output_attrs = NULL;
    rknn3_tensor *inputs  = NULL;
    rknn3_tensor *outputs = NULL;
    uint32_t i;

    memset(app_ctx, 0, sizeof(*app_ctx));

    printf("[Demo] rknn3_init...\n");
    ret = rknn3_init(&ctx, NULL);
    if (ret) {
        fprintf(stderr, "rknn3_init fail! ret=%d\n", ret);
        return ret;
    }

    printf("[Demo] rknn3_load_model_from_path: %s %s\n", model_path, weight_path);
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret) {
        fprintf(stderr, "rknn3_load_model_from_path fail! ret=%d\n", ret);
        goto cleanup;
    }

    printf("[Demo] rknn3_model_init...\n");
    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        fprintf(stderr, "rknn3_model_init fail! ret=%d\n", ret);
        goto cleanup;
    }

    printf("[Demo] rknn3_register_custom_ops_plugins: %s\n", plugin_path);
#if defined(INTERFACE_LINUX_OR_ANDROID)
    ret = rknn3_register_custom_ops_plugins(ctx, plugin_path, 0);
#else
    ret = rknn3_register_custom_ops_plugins(ctx, plugin_path, strlen(plugin_path));
#endif
    if (ret != RKNN3_SUCCESS) {
        fprintf(stderr, "rknn3_register_custom_ops_plugins fail! ret=%d\n", ret);
        goto cleanup;
    }
    printf("[Demo] Plugin registered successfully\n");

    /* Query input/output number */
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret) {
        fprintf(stderr, "rknn3_query io_num fail! ret=%d\n", ret);
        goto cleanup;
    }
    printf("[Demo] n_inputs=%u, n_outputs=%u\n", io_num.n_input, io_num.n_output);

    /* Query input/output attrs (one by one) */
    input_attrs = (rknn3_tensor_attr *)malloc(io_num.n_input * sizeof(rknn3_tensor_attr));
    if (!input_attrs) {
        fprintf(stderr, "malloc input_attrs failed\n");
        ret = -1;
        goto cleanup;
    }
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn3_tensor_attr));
    for (i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn3_tensor_attr));
        if (ret) {
            fprintf(stderr, "rknn3_query input[%u] fail! ret=%d\n", i, ret);
            goto cleanup;
        }
    }

    output_attrs = (rknn3_tensor_attr *)malloc(io_num.n_output * sizeof(rknn3_tensor_attr));
    if (!output_attrs) {
        fprintf(stderr, "malloc output_attrs failed\n");
        ret = -1;
        goto cleanup;
    }
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn3_tensor_attr));
    for (i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn3_tensor_attr));
        if (ret) {
            fprintf(stderr, "rknn3_query output[%u] fail! ret=%d\n", i, ret);
            goto cleanup;
        }
    }

    /* Create memory and tensor arrays */
    inputs = (rknn3_tensor *)malloc(io_num.n_input * sizeof(rknn3_tensor));
    if (!inputs) {
        fprintf(stderr, "malloc inputs failed\n");
        ret = -1;
        goto cleanup;
    }
    outputs = (rknn3_tensor *)malloc(io_num.n_output * sizeof(rknn3_tensor));
    if (!outputs) {
        fprintf(stderr, "malloc outputs failed\n");
        ret = -1;
        goto cleanup;
    }
    memset(inputs, 0, io_num.n_input * sizeof(rknn3_tensor));
    memset(outputs, 0, io_num.n_output * sizeof(rknn3_tensor));

    for (i = 0; i < io_num.n_input; i++) {
        inputs[i].mem = rknn3_create_mem(ctx, input_attrs[i].aligned_size,
                                         input_attrs[i].core_id,
                                         RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!inputs[i].mem) {
            fprintf(stderr, "rknn3_create_mem input[%u] failed\n", i);
            ret = -1;
            goto cleanup;
        }
        inputs[i].attr = &input_attrs[i];
    }
    for (i = 0; i < io_num.n_output; i++) {
        outputs[i].mem = rknn3_create_mem(ctx, output_attrs[i].aligned_size,
                                          output_attrs[i].core_id,
                                          RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!outputs[i].mem) {
            fprintf(stderr, "rknn3_create_mem output[%u] failed\n", i);
            ret = -1;
            goto cleanup;
        }
        outputs[i].attr = &output_attrs[i];
    }

    /* Fill app context */
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->inputs = inputs;
    app_ctx->outputs = outputs;
    app_ctx->model_batch   = input_shape.size() > 0 ? input_shape[0] : 1;
    app_ctx->model_channel = input_shape.size() > 1 ? input_shape[1] : 1;
    app_ctx->model_height  = input_shape.size() > 2 ? input_shape[2] : 1;
    app_ctx->model_width   = input_shape.size() > 3 ? input_shape[3] : 1;
    printf("[Demo] model input batch=%d channel=%d height=%d width=%d\n",
           app_ctx->model_batch, app_ctx->model_channel, app_ctx->model_height, app_ctx->model_width);

    return RKNN3_SUCCESS;

cleanup:
    /* unified error cleanup: destroy created mems, free arrays, destroy ctx.
     * mems are zero-initialized by memset, so NULL checks are safe even if
     * the array was only partially filled. */
    if (inputs) {
        for (i = 0; i < io_num.n_input; i++) {
            if (inputs[i].mem) rknn3_destroy_mem(ctx, inputs[i].mem);
        }
        free(inputs);
    }
    if (outputs) {
        for (i = 0; i < io_num.n_output; i++) {
            if (outputs[i].mem) rknn3_destroy_mem(ctx, outputs[i].mem);
        }
        free(outputs);
    }
    if (output_attrs) free(output_attrs);
    if (input_attrs)  free(input_attrs);
    if (ctx) rknn3_destroy(ctx);
    return ret;
}

/* Inference */
static int inference_custom_op_model(rknn_app_context_t *app_ctx,
                                     const std::vector<float> &input_f32,
                                     std::vector<float> &output_f32)
{
    int ret;
    rknn3_context ctx = app_ctx->rknn_ctx;
    rknn3_tensor_attr *in_attr = app_ctx->inputs[0].attr;
    rknn3_tensor_attr *out_attr = app_ctx->outputs[0].attr;
    void *input_raw = app_ctx->inputs[0].mem->virt_addr;
    void *output_raw = app_ctx->outputs[0].mem->virt_addr;

    int batch = app_ctx->model_batch;
    int chan  = app_ctx->model_channel;
    int h     = app_ctx->model_height;
    int w     = app_ctx->model_width;

    /* Pack input NCHW fp32 into the device layout. */
    if (in_attr->layout == RKNN3_TENSOR_NC1HWC2 && in_attr->dtype == RKNN3_TENSOR_FLOAT16) {
        nchw_fp32_to_nc1hwc2_fp16(input_f32.data(), input_raw, in_attr->shape, chan);
    } else if (in_attr->layout == RKNN3_TENSOR_NCHW && in_attr->dtype == RKNN3_TENSOR_FLOAT16) {
        float16 *d = (float16 *)input_raw;
        for (size_t i = 0; i < input_f32.size(); i++) d[i] = fp32_to_fp16(input_f32[i]);
    } else {
        memcpy(input_raw, input_f32.data(), input_f32.size() * sizeof(float));
    }

    ret = rknn3_mem_sync(ctx, app_ctx->inputs[0].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
    if (ret) { fprintf(stderr, "mem_sync input fail! ret=%d\n", ret); return ret; }

    printf("[Demo] Running inference...\n");
    ret = rknn3_run(ctx, app_ctx->inputs, app_ctx->io_num.n_input,
                    app_ctx->outputs, app_ctx->io_num.n_output);
    if (ret < 0) { fprintf(stderr, "rknn3_run fail! ret=%d\n", ret); return ret; }

    ret = rknn3_mem_sync(ctx, app_ctx->outputs[0].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
    if (ret) { fprintf(stderr, "mem_sync output fail! ret=%d\n", ret); return ret; }

    /* Unpack output device layout -> NCHW fp32. */
    size_t out_total = (size_t)batch * chan * h * w;
    output_f32.resize(out_total);
    if (out_attr->layout == RKNN3_TENSOR_NC1HWC2 && out_attr->dtype == RKNN3_TENSOR_FLOAT16) {
        nc1hwc2_to_nchw_fp32(output_raw, output_f32.data(), out_attr->shape, chan);
    } else if (out_attr->layout == RKNN3_TENSOR_NCHW && out_attr->dtype == RKNN3_TENSOR_FLOAT16) {
        const float16 *s = (const float16 *)output_raw;
        for (size_t i = 0; i < out_total; i++) output_f32[i] = fp16_to_fp32(s[i]);
    } else {
        memcpy(output_f32.data(), output_raw, out_total * sizeof(float));
    }

    return RKNN3_SUCCESS;
}

/* Release: free I/O memory and context.*/
static int release_custom_op_model(rknn_app_context_t *app_ctx)
{
    rknn3_context ctx = app_ctx->rknn_ctx;

    for (uint32_t i = 0; i < app_ctx->io_num.n_input; i++) {
        if (app_ctx->inputs[i].mem) {
            rknn3_destroy_mem(ctx, app_ctx->inputs[i].mem);
            app_ctx->inputs[i].mem = NULL;
        }
    }
    /* attr 是连续数组，input_attrs[i] 挂到 inputs[i].attr，只有首地址可 free */
    if (app_ctx->io_num.n_input > 0 && app_ctx->inputs[0].attr) {
        free(app_ctx->inputs[0].attr);
        for (uint32_t i = 0; i < app_ctx->io_num.n_input; i++)
            app_ctx->inputs[i].attr = NULL;
    }

    for (uint32_t i = 0; i < app_ctx->io_num.n_output; i++) {
        if (app_ctx->outputs[i].mem) {
            rknn3_destroy_mem(ctx, app_ctx->outputs[i].mem);
            app_ctx->outputs[i].mem = NULL;
        }
    }
    if (app_ctx->io_num.n_output > 0 && app_ctx->outputs[0].attr) {
        free(app_ctx->outputs[0].attr);
        for (uint32_t i = 0; i < app_ctx->io_num.n_output; i++)
            app_ctx->outputs[i].attr = NULL;
    }

    free(app_ctx->inputs);
    free(app_ctx->outputs);
    app_ctx->inputs = NULL;
    app_ctx->outputs = NULL;

    if (ctx) {
        rknn3_destroy(ctx);
        app_ctx->rknn_ctx = 0;
    }
    return RKNN3_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  main                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    if (argc < 7) {
        printf("%s <model_path> <weight_path> <ref_input.npy> <ref_output.npy> <core_mask> <plugin_path>\n",
               argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *weight_path = argv[2];
    const char *ref_input_path = argv[3];
    const char *ref_output_path = argv[4];
    const char *plugin_path = argv[6];

    /* Parse core_mask (hexadecimal) */
    char *endptr = NULL;
    errno = 0;
    unsigned long parsed_mask = strtoul(argv[5], &endptr, 16);
    if (endptr == argv[5] || *endptr != '\0' || errno != 0 || parsed_mask > 0xFF) {
        fprintf(stderr, "Error: Invalid core_mask '%s'. Expected a hexadecimal string (e.g., 0x01), low 8 bits valid.\n", argv[5]);
        return -1;
    }
    uint32_t core_mask = (uint32_t)parsed_mask;

    /* 1. Load reference input and output */
    std::vector<float> input_data;
    std::vector<int> input_shape;
    if (read_npy_f32(ref_input_path, input_data, input_shape) != 0) {
        fprintf(stderr, "Failed to read reference input: %s\n", ref_input_path);
        return -1;
    }
    printf("[Demo] Input shape:");
    for (size_t i = 0; i < input_shape.size(); i++) printf(" %d", input_shape[i]);
    printf(", total=%zu elems\n", input_data.size());

    std::vector<float> ref_output;
    std::vector<int> ref_shape;
    if (read_npy_f32(ref_output_path, ref_output, ref_shape) != 0) {
        fprintf(stderr, "Failed to read reference output: %s\n", ref_output_path);
        return -1;
    }
    printf("[Demo] Reference output shape:");
    for (size_t i = 0; i < ref_shape.size(); i++) printf(" %d", ref_shape[i]);
    printf(", total=%zu elems\n", ref_output.size());

    /* 2. Init model */
    rknn_app_context_t app_ctx;
    int ret = init_custom_op_model(model_path, weight_path, plugin_path, core_mask,
                                   input_shape, &app_ctx);
    if (ret != RKNN3_SUCCESS) {
        return ret;
    }

    /* 3. Inference */
    std::vector<float> rknn_output;
    ret = inference_custom_op_model(&app_ctx, input_data, rknn_output);
    if (ret != RKNN3_SUCCESS) {
        release_custom_op_model(&app_ctx);
        return ret;
    }

    /* 4. Compare with reference output */
    if (rknn_output.size() != ref_output.size()) {
        fprintf(stderr, "Output size mismatch: rknn=%zu, ref=%zu\n", rknn_output.size(), ref_output.size());
        release_custom_op_model(&app_ctx);
        return -1;
    }
    size_t output_total = ref_output.size();
    float max_diff = 0;
    for (size_t i = 0; i < output_total; i++) {
        float diff = fabsf(rknn_output[i] - ref_output[i]);
        if (diff > max_diff) max_diff = diff;
    }
    float cos_sim = cosine_similarity(rknn_output.data(), ref_output.data(), output_total);

    printf("\n========== Results ==========\n");
    printf("Max diff:        %.6f\n", max_diff);
    printf("Cosine sim:      %.6f\n", cos_sim);
    if (cos_sim > 0.999f) {
        printf(">>> PASS (cosine > 0.999)\n");
    } else {
        printf(">>> CHECK (cosine <= 0.999)\n");
    }
    printf("=============================\n");

    /* 5. Release */
    release_custom_op_model(&app_ctx);

    return RKNN3_SUCCESS;
}
