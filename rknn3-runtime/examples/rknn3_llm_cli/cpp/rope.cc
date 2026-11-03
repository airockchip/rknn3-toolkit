// External rope cache support for the rknn3 LLM demos.
// Some models (Qwen3.5 / Gemma-4 with rope_cache_host_storage=1) keep the
// rope cos/sin caches on the host and feed them per generation step through
// the session's input_callback. This mirrors examples/multicard.

#include "rope.h"
#include "helpers.h"

#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "nlohmann/json.hpp"

static const char* GEMMA4_ROPE_CACHE_NAMES[4] = {
    "rope_cos_cache_0", "rope_sin_cache_0",
    "rope_cos_cache_1", "rope_sin_cache_1"
};

static const char* QWEN35_ROPE_CACHE_NAMES[2] = {
    "rope_cos_cache", "rope_sin_cache"
};

static size_t get_dtype_elem_size(int dtype)
{
  switch (dtype) {
  case 0:  return 4;   /* FLOAT32   */
  case 1:  return 2;   /* FLOAT16   */
  case 2:  return 1;   /* INT8      */
  case 3:  return 1;   /* UINT8     */
  case 4:  return 2;   /* INT16     */
  case 5:  return 2;   /* UINT16    */
  case 6:  return 4;   /* INT32     */
  case 7:  return 4;   /* UINT32    */
  case 8:  return 8;   /* INT64     */
  case 9:  return 8;   /* UINT64    */
  case 10: return 1;   /* BOOL      */
  case 11: return 1;   /* INT4      */
  case 12: return 1;   /* FLOAT8E4M3FN */
  case 13: return 2;   /* BFLOAT16  */
  case 14: return 1;   /* FLOAT8E8M0   */
  case 15: return 1;   /* FLOAT4E2M1   */
  default: return 1;
  }
}

static int find_rope_cache_index(RopeCacheFormat format, const char* name)
{
  const char* const* names = nullptr;
  int                count = 0;
  if (format == RopeCacheFormat::GEMMA4) {
    names = GEMMA4_ROPE_CACHE_NAMES;
    count = 4;
  } else if (format == RopeCacheFormat::QWEN35) {
    names = QWEN35_ROPE_CACHE_NAMES;
    count = 2;
  }

  for (int i = 0; i < count; ++i) {
    if (strcmp(name, names[i]) == 0) {
      return i;
    }
  }
  return -1;
}

static int load_rope_cache(const char* path, rope_cache_tensor caches[4],
                           RopeCacheFormat* format_out,
                           int* fd_out, void** mmap_base_out, size_t* mmap_size_out)
{
  int         fd          = -1;
  void*       map         = MAP_FAILED;
  uint64_t    header_size = 0;
  struct stat st;
  int         ret = -1;

  *format_out = RopeCacheFormat::NONE;

  fd = open(path, O_RDONLY);
  if (fd < 0) {
    printf("Failed to open rope cache file: %s\n", path);
    goto err;
  }
  if (fstat(fd, &st) < 0) {
    printf("Failed to stat rope cache file: %s\n", path);
    goto err;
  }

  if (read(fd, &header_size, 8) != 8) {
    printf("Failed to read rope cache header size\n");
    goto err;
  }
  if (header_size == 0 || header_size > (uint64_t)st.st_size - 8) {
    printf("Invalid rope cache header size: %" PRIu64 "\n", header_size);
    goto err;
  }

  map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) {
    printf("Failed to mmap rope cache file: %s\n", path);
    goto err;
  }

  {
    const char*    json_ptr  = (const char*)map + 8;
    const uint8_t* data_base = (const uint8_t*)map + 8 + header_size;
    try {
      nlohmann::json j = nlohmann::json::parse(json_ptr, json_ptr + header_size);

      nlohmann::json meta_index = nlohmann::json::parse(
          j.at("__metadata__").at("index").get<std::string>());

      const char* const* rope_names = nullptr;
      int                n_rope_caches = 0;
      RopeCacheFormat    rope_format = RopeCacheFormat::NONE;
      auto has_rope_tensor = [&meta_index, &j](const char* name) {
        return meta_index.find(name) != meta_index.end() && j.find(name) != j.end();
      };

      bool has_gemma4_names = true;
      for (int i = 0; i < 4; ++i) {
        if (!has_rope_tensor(GEMMA4_ROPE_CACHE_NAMES[i])) {
          has_gemma4_names = false;
          break;
        }
      }
      bool has_qwen35_names = true;
      for (int i = 0; i < 2; ++i) {
        if (!has_rope_tensor(QWEN35_ROPE_CACHE_NAMES[i])) {
          has_qwen35_names = false;
          break;
        }
      }

      if (has_gemma4_names) {
        rope_names = GEMMA4_ROPE_CACHE_NAMES;
        n_rope_caches = 4;
        rope_format = RopeCacheFormat::GEMMA4;
      } else if (has_qwen35_names) {
        rope_names = QWEN35_ROPE_CACHE_NAMES;
        n_rope_caches = 2;
        rope_format = RopeCacheFormat::QWEN35;
      } else {
        printf("rope cache must contain either Gemma4 rope caches ");
        printf("(rope_cos_cache_0/1, rope_sin_cache_0/1) or Qwen3.5 ");
        printf("rope caches (rope_cos_cache, rope_sin_cache)\n");
        ret = -1;
      }

      ret = rope_format == RopeCacheFormat::NONE ? -1 : 0;
      for (int i = 0; i < n_rope_caches && ret == 0; i++) {
        const auto& meta_t = meta_index.at(rope_names[i]);
        int         dtype  = meta_t.at("dtype").get<int>();
        int         layout = meta_t.at("layout").get<int>();

        const auto& t      = j.at(rope_names[i]);
        auto shape_v   = t.at("shape").get<std::vector<int>>();
        auto offsets_v = t.at("data_offsets").get<std::vector<int64_t>>();

        int n_dims = (int)shape_v.size();
        if (n_dims != 5 || layout != 3) {
          printf("Tensor '%s': expected 5-D NC1HWC2 (layout=%d, n_dims=%d)\n",
                 rope_names[i], layout, n_dims);
          ret = -1;
          break;
        }
        /* bounds validation for a corrupted / truncated / mismatched file */
        if (offsets_v.size() != 2 || offsets_v[0] < 0) {
          printf("Tensor '%s': invalid data_offsets\n", rope_names[i]);
          ret = -1;
          break;
        }
        size_t tensor_bytes = 1;
        bool   shape_ok = true;
        for (int d = 0; d < n_dims; d++) {
          if (shape_v[d] <= 0) { shape_ok = false; break; }
          tensor_bytes *= (size_t)shape_v[d];
        }
        if (!shape_ok) {
          printf("Tensor '%s': invalid shape\n", rope_names[i]);
          ret = -1;
          break;
        }
        tensor_bytes *= get_dtype_elem_size(dtype);
        if (tensor_bytes == 0 ||
            (uint64_t)offsets_v[0] + tensor_bytes > (uint64_t)st.st_size) {
          printf("Tensor '%s': data offset+size out of file bounds\n", rope_names[i]);
          ret = -1;
          break;
        }
        caches[i].data      = (void*)(data_base + offsets_v[0]);
        caches[i].data_size = tensor_bytes;
        caches[i].n_dims    = n_dims;
        caches[i].dtype     = dtype;
        caches[i].layout    = layout;
        for (int d = 0; d < n_dims; d++) caches[i].shape[d] = shape_v[d];
        printf("Loaded %-24s  dtype=%-2d  shape=[%d,%d,%d,%d,%d]\n",
               rope_names[i], dtype,
               caches[i].shape[0], caches[i].shape[1], caches[i].shape[2],
               caches[i].shape[3], caches[i].shape[4]);
      }
      if (ret == 0) {
        *format_out = rope_format;
      }
    } catch (const nlohmann::json::exception& e) {
      printf("Failed to parse rope cache JSON: %s\n", e.what());
      ret = -1;
    }
  }

err:
  if (ret != 0) {
    if (map != MAP_FAILED) munmap(map, (size_t)st.st_size);
    if (fd >= 0) close(fd);
    return ret;
  }
  *fd_out        = fd;
  *mmap_base_out = map;
  *mmap_size_out = (size_t)st.st_size;
  return 0;
}

/* collect the indices of the model's rope-cache / per_layer_inputs inputs */
static int query_ext_input_indices(rknn3_context ctx, RopeState* st)
{
  rknn3_input_output_num io_num;
  memset(&io_num, 0, sizeof(io_num));
  int ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_query IN_OUT_NUM for ext inputs failed, ret=%d\n", ret);
    return -1;
  }

  for (uint32_t i = 0; i < io_num.n_input; ++i) {
    rknn3_tensor_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.index = i;
    ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_query INPUT_ATTR[%u] failed, ret=%d\n", i, ret);
      return -1;
    }
    if (strcmp(attr.name, "per_layer_inputs") == 0) {
      st->n_ext_inputs++;
    } else if (strstr(attr.name, "rope_cos_cache") || strstr(attr.name, "rope_sin_cache")) {
      st->n_ext_inputs++;
    }
  }

  if (st->n_ext_inputs == 0) {
    printf("no per_layer_inputs / rope_cos_cache / rope_sin_cache input tensors found\n");
    return -1;
  }

  st->ext_input_indices = (int*)malloc(st->n_ext_inputs * sizeof(int));
  if (!st->ext_input_indices) {
    printf("malloc ext_input_indices failed\n");
    st->n_ext_inputs = 0;
    return -1;
  }

  st->n_ext_inputs = 0;
  for (uint32_t i = 0; i < io_num.n_input; ++i) {
    rknn3_tensor_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.index = i;
    ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    if (ret != RKNN3_SUCCESS) {
      return -1;
    }
    if (strcmp(attr.name, "per_layer_inputs") == 0) {
      st->ext_input_indices[st->n_ext_inputs++] = (int)i;
    } else if (strstr(attr.name, "rope_cos_cache") || strstr(attr.name, "rope_sin_cache")) {
      st->ext_input_indices[st->n_ext_inputs++] = (int)i;
    }
  }

  printf("found %d ext input tensor(s) (per_layer_inputs/rope_caches)\n", st->n_ext_inputs);
  return 0;
}

int rope_setup(RopeState* st, rknn3_context ctx, const char* rope_path)
{
  if (!st || !ctx || !rope_path || rope_path[0] == '\0')
    return -1;

  int ret = load_rope_cache(rope_path, st->rope_caches, &st->rope_format,
                            &st->rope_fd, &st->rope_mmap_base, &st->rope_mmap_size);
  if (ret != 0)
    return ret;

  ret = query_ext_input_indices(ctx, st);
  if (ret != 0) {
    rope_teardown(st);
    return ret;
  }

  st->enabled = true;
  return 0;
}

int rope_input_callback(void* userdata, rknn3_tensor* input_tensors, uint32_t n_input_tensors,
                        LLMInputCallbackParam param)
{
  RopeState* cb_data = (RopeState*)userdata;

  for (uint32_t i = 0; i < n_input_tensors; ++i) {
    int cache_index = find_rope_cache_index(cb_data->rope_format, input_tensors[i].attr->name);
    if (cache_index < 0) {
      continue;
    }

    const rope_cache_tensor* cache      = &cb_data->rope_caches[cache_index];
    const size_t             elem_sz    = get_dtype_elem_size(cache->dtype);
    const int                C1         = cache->shape[1];
    const size_t             c2_bytes   = (size_t)cache->shape[4] * elem_sz;
    const size_t             src_stride = (size_t)cache->shape[3] * c2_bytes;
    const size_t             dst_stride = (size_t)input_tensors[i].attr->shape[3] * c2_bytes;
    const size_t             copy_stride = src_stride < dst_stride ? src_stride : dst_stride;
    if (param.pos < 0) {
      printf("rope input_callback: pos %d out of bounds (cache %zu bytes)\n",
             (int)param.pos, cache->data_size);
      return -1;
    }
    const uint8_t*           src = (const uint8_t*)cache->data + (size_t)param.pos * c2_bytes;
    uint8_t*                 dst = (uint8_t*)input_tensors[i].mem->virt_addr;
    for (int c1 = 0; c1 < C1; c1++, src += src_stride, dst += dst_stride) {
      memcpy(dst, src, copy_stride);
    }
  }

  return 0;
}

void rope_teardown(RopeState* st)
{
  if (!st)
    return;
  if (st->rope_mmap_base && st->rope_mmap_base != MAP_FAILED) {
    munmap(st->rope_mmap_base, st->rope_mmap_size);
    st->rope_mmap_base = nullptr;
  }
  if (st->rope_fd >= 0) {
    close(st->rope_fd);
    st->rope_fd = -1;
  }
  if (st->ext_input_indices) {
    free(st->ext_input_indices);
    st->ext_input_indices = nullptr;
  }
  st->n_ext_inputs = 0;
  st->enabled = false;
}
