#include "rknn3_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double bytes_to_mb(uint64_t bytes)
{
  return (double)bytes / (1024.0 * 1024.0);
}

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

static const char* get_kvcache_dtype_string(rknn3_kvcache_dtype dtype)
{
  switch (dtype) {
  case RKNN3_KVCACHE_DTYPE_INT4_TO_F16:
    return "INT4_TO_F16";
  case RKNN3_KVCACHE_DTYPE_INT4_TO_F8:
    return "INT4_TO_F8";
  case RKNN3_KVCACHE_DTYPE_INT8_TO_F16:
    return "INT8_TO_F16";
  case RKNN3_KVCACHE_DTYPE_FLOAT4_TO_F16:
    return "FLOAT4_TO_F16";
  case RKNN3_KVCACHE_DTYPE_FLOAT4_TO_F8:
    return "FLOAT4_TO_F8";
  case RKNN3_KVCACHE_DTYPE_FLOAT8_TO_F16:
    return "FLOAT8_TO_F16";
  case RKNN3_KVCACHE_DTYPE_FLOAT8_TO_F8:
    return "FLOAT8_TO_F8";
  case RKNN3_KVCACHE_DTYPE_FLOAT16:
    return "FLOAT16";
  case RKNN3_KVCACHE_DTYPE_UNDEFINED:
  default:
    return "UNDEFINED";
  }
}

static const char* get_kvcache_store_method_string(rknn3_kvcache_store_method method)
{
  switch (method) {
  case RKNN3_KVCACHE_STORE_METHOD_NORMAL:
    return "NORMAL";
  case RKNN3_KVCACHE_STORE_METHOD_GROUP_QUANT:
    return "GROUP_QUANT";
  case RKNN3_KVCACHE_STORE_METHOD_UNDEFINED:
  default:
    return "UNDEFINED";
  }
}

static const char* find_device_id(const rknn3_devices* devs, const char* device_id)
{
  if (device_id == NULL || strlen(device_id) == 0) {
    return devs->devices[0].id;
  }

  for (uint32_t i = 0; i < devs->n_devices; i++) {
    if (strcmp(devs->devices[i].id, device_id) == 0) {
      return devs->devices[i].id;
    }
  }

  return NULL;
}

static void print_device_list(const rknn3_devices* devs)
{
  printf("Found %u RK182X device(s)\n", devs->n_devices);
  for (uint32_t i = 0; i < devs->n_devices; i++) {
    const rknn3_device* dev = &devs->devices[i];
    printf("  Device %u: transfer_type=%s, id=%s\n", i, dev->type, dev->id);
  }
}

static double calc_used_percent(uint64_t total, uint64_t used)
{
  return total > 0 ? (double)used * 100.0 / total : 0.0;
}

static int print_device_mem_info(rknn3_context ctx)
{
  rknn3_dev_mem_info dev_mem_info;
  memset(&dev_mem_info, 0, sizeof(dev_mem_info));

  int ret = rknn3_query(ctx, RKNN3_QUERY_DEVICE_MEM_INFO, &dev_mem_info, sizeof(dev_mem_info));
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_query device mem info failed! ret=%d\n", ret);
    return ret;
  }

  uint64_t sum_total = 0;
  uint64_t sum_free  = 0;
  uint64_t sum_used  = 0;

  printf("\nDevice Memory (MB):\n");
  printf("    Node       Total        Free         Used       Used(%%)\n");
  printf("  -------- ------------ ------------ ------------ ----------\n");

  uint64_t sys_used = dev_mem_info.sys_total - dev_mem_info.sys_free;
  printf("  %-8s %10.2f %12.2f %12.2f %10.1f\n", "System", bytes_to_mb(dev_mem_info.sys_total),
         bytes_to_mb(dev_mem_info.sys_free), bytes_to_mb(sys_used),
         calc_used_percent(dev_mem_info.sys_total, sys_used));

  for (int i = 0; i < dev_mem_info.node_num; i++) {
    const rknn3_node_mem_info* node = &dev_mem_info.node_mem_info[i];
    uint64_t node_used = node->total - node->free;
    printf("  %-8d %10.2f %12.2f %12.2f %10.1f\n", i, bytes_to_mb(node->total),
           bytes_to_mb(node->free), bytes_to_mb(node_used), calc_used_percent(node->total, node_used));
    sum_total += node->total;
    sum_free += node->free;
    sum_used += node_used;
  }

  if (dev_mem_info.node_num > 0) {
    printf("  -------- ------------ ------------ ------------ ----------\n");
    printf("  %-8s %10.2f %12.2f %12.2f %10.1f\n", "Total", bytes_to_mb(sum_total),
           bytes_to_mb(sum_free), bytes_to_mb(sum_used), calc_used_percent(sum_total, sum_used));
  }
  return RKNN3_SUCCESS;
}

static int print_sdk_version(rknn3_context ctx)
{
  rknn3_sdk_version sdk_version;
  memset(&sdk_version, 0, sizeof(sdk_version));

  int ret = rknn3_query(ctx, RKNN3_QUERY_SDK_VERSION, &sdk_version, sizeof(sdk_version));
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_query sdk version failed! ret=%d\n", ret);
    return ret;
  }

  printf("\nSDK Version:\n  api=%s\n  drv=%s\n", sdk_version.api_version, sdk_version.drv_version);
  return RKNN3_SUCCESS;
}

static int print_model_basic_info(rknn3_context ctx)
{
  rknn3_input_output_num io_num;
  memset(&io_num, 0, sizeof(io_num));

  int ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_query in/out num failed! ret=%d\n", ret);
    return ret;
  }

  uint32_t core_num = 0;
  ret = rknn3_query(ctx, RKNN3_QUERY_CORE_NUMBER, &core_num, sizeof(core_num));
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_query core number failed! ret=%d\n", ret);
    return ret;
  }

  printf("\nModel Basic Info:\n");
  printf("  core_number=%u\n", core_num);
  printf("  input_number=%u, output_number=%u\n", io_num.n_input, io_num.n_output);
  return RKNN3_SUCCESS;
}

static int print_custom_string(rknn3_context ctx)
{
  rknn3_custom_string custom_string;
  memset(&custom_string, 0, sizeof(custom_string));

  int ret = rknn3_query(ctx, RKNN3_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_query custom string failed! ret=%d\n", ret);
    return ret;
  }

  printf("\nCustom String:\n  %s\n", custom_string.string);
  return RKNN3_SUCCESS;
}

static void print_tensor_attr(const char* prefix, uint32_t idx, const rknn3_tensor_attr* attr)
{
  printf("  %s[%u]: name=%s, dtype=%s, layout=%s, n_dims=%u, shape=[", prefix, idx, attr->name,
         rknn3_get_type_string(attr->dtype), rknn3_get_layout_string(attr->layout), attr->n_dims);
  for (uint32_t j = 0; j < attr->n_dims; j++) {
    printf("%u%s", attr->shape[j], (j < attr->n_dims - 1) ? ", " : "");
  }
  printf("], n_stride=%u, stride=[", attr->n_stride);
  for (uint32_t j = 0; j < attr->n_stride; j++) {
    printf("%lu%s", (unsigned long)attr->stride[j], (j < attr->n_stride - 1) ? ", " : "");
  }
  printf("], n_elems=%u, aligned_size=%lu bytes, qnt_type=%s, scale=%f, zero_point=%d, core_id=%d\n",
         attr->n_elems, (unsigned long)attr->aligned_size, rknn3_get_qnt_type_string(attr->qnt_type),
         attr->qnt_info.scale, attr->qnt_info.zero_point, attr->core_id);
}

static int print_tensor_attrs(rknn3_context ctx)
{
  rknn3_input_output_num io_num;
  memset(&io_num, 0, sizeof(io_num));

  int ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_query in/out num failed! ret=%d\n", ret);
    return ret;
  }

  rknn3_tensor_attr attr;

  printf("\nInput Tensor Attributes:\n");
  for (uint32_t i = 0; i < io_num.n_input; i++) {
    memset(&attr, 0, sizeof(attr));
    attr.index = i;
    ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_query input attr[%u] failed! ret=%d\n", i, ret);
      return ret;
    }
    print_tensor_attr("Input", i, &attr);
  }

  printf("\nOutput Tensor Attributes:\n");
  for (uint32_t i = 0; i < io_num.n_output; i++) {
    memset(&attr, 0, sizeof(attr));
    attr.index = i;
    ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_query output attr[%u] failed! ret=%d\n", i, ret);
      return ret;
    }
    print_tensor_attr("Output", i, &attr);
  }

  return RKNN3_SUCCESS;
}

static int print_llm_config(rknn3_context ctx)
{
  rknn3_llm_config llm_config;
  memset(&llm_config, 0, sizeof(llm_config));

  int ret = rknn3_query(ctx, RKNN3_QUERY_LLM_CONFIG, &llm_config, sizeof(llm_config));
  if (ret != RKNN3_SUCCESS) {
    printf("\nLLM Config: not available for this model (ret=%d)\n", ret);
    return 0;
  }

  printf("\n======================== LLM / KVCache Config ========================\n");
  printf("  model_type                 : %s\n", llm_config.model_type ? llm_config.model_type : "N/A");
  printf("  vocab_size                 : %u\n", llm_config.vocab_size);
  printf("  embedding_dim              : %u\n", llm_config.embedding_dim);
  printf("  max_ctx_len                : %u\n", llm_config.max_ctx_len);
  printf("  max_position_embeddings    : %u\n", llm_config.max_position_embeddings);
  printf("  kvcache_store_method       : %s\n", get_kvcache_store_method_string(llm_config.kvcache_store_method));
  printf("  kvcache_dtype              : %s\n", get_kvcache_dtype_string(llm_config.kvcache_dtype));
  printf("  kvcache_group_size         : %u\n", llm_config.kvcache_group_size);
  printf("  kvcache_residual_depth     : %u\n", llm_config.kvcache_residual_depth);
  printf("  rope_cache_host_storage    : %u\n", llm_config.rope_cache_host_storage);
  printf("=======================================================================\n");
  return RKNN3_SUCCESS;
}

static const char* get_page_cache_type_string(int32_t cache_type)
{
  switch (cache_type) {
  case RKNN3_PAGE_CACHE_GENERAL:
    return "GENERAL";
  case RKNN3_PAGE_CACHE_FULL_ATTENTION:
    return "FULL_ATTENTION";
  case RKNN3_PAGE_CACHE_SLIDING_ATTENTION:
    return "SLIDING_ATTENTION";
  case RKNN3_PAGE_CACHE_LINEAR_ATTENTION:
    return "LINEAR_ATTENTION";
  case RKNN3_PAGE_CACHE_CONV_STREAMING:
    return "CONV_STREAMING";
  case RKNN3_PAGE_CACHE_FULL_RESIDUAL:
    return "FULL_RESIDUAL";
  case RKNN3_PAGE_CACHE_SLIDING_RESIDUAL:
    return "SLIDING_RESIDUAL";
  default:
    return "UNKNOWN";
  }
}

// Column widths of the per-core page table, kept in one place so the header,
// the rows and the separators always line up.
#define KVCACHE_NAME_WIDTH 22
#define KVCACHE_NUM_WIDTH  11

// Print the indent plus one dashed cell per table column.
static void print_kvcache_table_separator(uint32_t indent)
{
  printf("%*s", (int)indent, "");
  for (int i = 0; i < KVCACHE_NAME_WIDTH; i++) {
    putchar('-');
  }
  for (int col = 0; col < 3; col++) {
    putchar(' ');
    for (int i = 0; i < KVCACHE_NUM_WIDTH; i++) {
      putchar('-');
    }
  }
  putchar('\n');
}

static bool same_core_layout(const rknn3_kvcache_page_info* a, const rknn3_kvcache_page_info* b)
{
  if (a->max_request_num != b->max_request_num || a->n_cache_types != b->n_cache_types) {
    return false;
  }
  for (uint32_t j = 0; j < b->n_cache_types && j < RKNN3_MAX_CACHE_PAGE_TYPES; j++) {
    if (a->caches[j].cache_type != b->caches[j].cache_type ||
        a->caches[j].page_size != b->caches[j].page_size ||
        a->caches[j].max_page_num != b->caches[j].max_page_num) {
      return false;
    }
  }
  return true;
}

// Append text to buf at *off, keeping buf NUL-terminated; never writes past buf_len.
static void append_to_buffer(char* buf, size_t buf_len, size_t* off, const char* text)
{
  if (buf_len == 0) {
    return;
  }
  size_t space = (*off < buf_len - 1) ? buf_len - 1 - *off : 0;
  size_t len = strlen(text);
  if (len > space) {
    len = space;
  }
  memcpy(buf + *off, text, len);
  *off += len;
  buf[*off] = '\0';
}

// snprintf at *off with explicit clamping: snprintf returns the untruncated
// length, so on truncation the offset stops at the last writable byte instead
// of overflowing or making later size calculations underflow.
static void append_formatted(char* buf, size_t buf_len, size_t* off, const char* fmt, uint32_t a,
                             uint32_t b)
{
  size_t space = buf_len - *off;
  int written = snprintf(buf + *off, space, fmt, a, b);
  if (written < 0) {
    return;
  }
  *off += ((size_t)written < space) ? (size_t)written : space - 1;
  buf[*off] = '\0';
}

// Render a sorted core id list as "0,1,2" with runs of 3 or more as "0-5".
static void format_core_list(const uint32_t* ids, uint32_t n_ids, char* buf, size_t buf_len)
{
  if (buf_len == 0) {
    return;
  }
  buf[0] = '\0';
  size_t off = 0;

  for (uint32_t i = 0; i < n_ids;) {
    uint32_t j = i;
    while (j + 1 < n_ids && ids[j + 1] == ids[j] + 1) {
      j++;
    }

    if (off > 0) {
      append_to_buffer(buf, buf_len, &off, ",");
    }
    if (j - i + 1 >= 3) {
      append_formatted(buf, buf_len, &off, "%u-%u", ids[i], ids[j]);
    } else {
      for (uint32_t k = i; k <= j; k++) {
        if (k > i) {
          append_to_buffer(buf, buf_len, &off, ",");
        }
        append_formatted(buf, buf_len, &off, "%u", ids[k], 0);
      }
    }
    i = j + 1;
  }
}

// Print one layout's table. Every distinct layout gets its own, so the columns
// stay the same as the common single-layout case.
static void print_kvcache_layout_table(const rknn3_kvcache_page_info* info, uint32_t indent)
{
  uint32_t n_cache_types = info->n_cache_types;
  if (n_cache_types > RKNN3_MAX_CACHE_PAGE_TYPES) {
    n_cache_types = RKNN3_MAX_CACHE_PAGE_TYPES;
  }

  uint64_t grand_total = 0;
  uint32_t total_pages = 0;
  for (uint32_t j = 0; j < n_cache_types; j++) {
    grand_total += info->caches[j].page_size * (uint64_t)info->caches[j].max_page_num;
    total_pages += info->caches[j].max_page_num;
  }

  printf("%*s%-*s %*s %*s %*s\n", (int)indent, "", KVCACHE_NAME_WIDTH, "Cache", KVCACHE_NUM_WIDTH,
         "Page Size", KVCACHE_NUM_WIDTH, "Max Pages", KVCACHE_NUM_WIDTH, "Total/Core");
  print_kvcache_table_separator(indent);
  for (uint32_t j = 0; j < n_cache_types; j++) {
    const rknn3_cache_page_attr* attr = &info->caches[j];
    uint64_t total = attr->page_size * (uint64_t)attr->max_page_num;
    printf("%*s%-*s %*.2f %*u %*.2f\n", (int)indent, "", KVCACHE_NAME_WIDTH,
           get_page_cache_type_string(attr->cache_type), KVCACHE_NUM_WIDTH,
           bytes_to_mb(attr->page_size), KVCACHE_NUM_WIDTH, attr->max_page_num, KVCACHE_NUM_WIDTH,
           bytes_to_mb(total));
  }
  print_kvcache_table_separator(indent);
  printf("%*s%-*s %*s %*u %*.2f\n", (int)indent, "", KVCACHE_NAME_WIDTH, "Total", KVCACHE_NUM_WIDTH,
         "", KVCACHE_NUM_WIDTH, total_pages, KVCACHE_NUM_WIDTH, bytes_to_mb(grand_total));
}

static int print_kvcache_page_info(rknn3_context ctx, uint32_t core_num)
{
  // Query every core once and keep the results, so a failing query can be
  // reported right away instead of being retried while printing.
  rknn3_kvcache_page_info* cores =
      (rknn3_kvcache_page_info*)calloc(core_num, sizeof(rknn3_kvcache_page_info));
  if (cores == NULL) {
    printf("Failed to allocate memory for kvcache page info\n");
    return -1;
  }

  for (uint32_t i = 0; i < core_num; i++) {
    cores[i].core_id = (int32_t)i;
    int ret = rknn3_query(ctx, RKNN3_QUERY_KVCACHE_PAGE_INFO, &cores[i], sizeof(cores[i]));
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_query kvcache page info failed on core %d! ret=%d\n", cores[i].core_id, ret);
      free(cores);
      return ret;
    }
  }

  // Old non-paged models report no cache types; skip this section entirely.
  if (cores[0].n_cache_types == 0) {
    free(cores);
    return RKNN3_SUCCESS;
  }

  // Most models give every core the same layout. Group cores by layout so a
  // mismatch prints one compact table per distinct layout instead of a dump.
  uint32_t* group_rep = (uint32_t*)malloc(core_num * sizeof(uint32_t));
  uint32_t* group_cores = (uint32_t*)malloc(core_num * sizeof(uint32_t));
  if (group_rep == NULL || group_cores == NULL) {
    printf("Failed to allocate memory for kvcache core grouping\n");
    free(group_rep);
    free(group_cores);
    free(cores);
    return -1;
  }

  uint32_t n_groups = 0;
  for (uint32_t i = 0; i < core_num; i++) {
    bool found = false;
    for (uint32_t g = 0; g < n_groups; g++) {
      if (same_core_layout(&cores[group_rep[g]], &cores[i])) {
        found = true;
        break;
      }
    }
    if (!found) {
      group_rep[n_groups++] = i;
    }
  }

  if (n_groups == 1) {
    printf("\nPer-Core KVCache Page Info (MB) (all %u cores share the same layout):\n", core_num);
    print_kvcache_layout_table(&cores[0], 2);
  } else {
    printf("\nPer-Core KVCache Page Info (MB) (%u cores, %u different layouts):\n", core_num,
           n_groups);
    for (uint32_t g = 0; g < n_groups; g++) {
      uint32_t n_group_cores = 0;
      for (uint32_t i = 0; i < core_num; i++) {
        if (same_core_layout(&cores[group_rep[g]], &cores[i])) {
          group_cores[n_group_cores++] = i;
        }
      }

      char core_list[256];
      format_core_list(group_cores, n_group_cores, core_list, sizeof(core_list));

      const rknn3_kvcache_page_info* rep = &cores[group_rep[g]];
      printf("\n  %s %s (%u core%s), max_request_num=%u:\n", n_group_cores == 1 ? "Core" : "Cores",
             core_list, n_group_cores, n_group_cores == 1 ? "" : "s", rep->max_request_num);
      print_kvcache_layout_table(rep, 4);
    }
  }

  free(group_rep);
  free(group_cores);
  free(cores);
  return RKNN3_SUCCESS;
}

static int print_allocation_info(rknn3_context ctx, uint32_t core_num)
{
  rknn3_allocation_info* alloc_infos =
      (rknn3_allocation_info*)calloc(core_num, sizeof(rknn3_allocation_info));
  if (alloc_infos == NULL) {
    printf("Failed to allocate memory for allocation info\n");
    return -1;
  }

  int ret = rknn3_query(ctx, RKNN3_QUERY_ALLOCATION_INFO, alloc_infos,
                        sizeof(rknn3_allocation_info) * core_num);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_query allocation info failed! ret=%d\n", ret);
    free(alloc_infos);
    return ret;
  }

  uint64_t total_command  = 0;
  uint64_t total_weight   = 0;
  uint64_t total_internal = 0;
  uint64_t total_kvcache  = 0;

  // Note: KVCache here is the raw allocation reported by RKNN3_QUERY_ALLOCATION_INFO.
  // For paged models that is the base/page geometry only; the per-core page count
  // and the resulting total are printed separately by print_kvcache_page_info().
  printf("\nPer-Core Memory Allocation (MB):\n");
  printf("    Core      Command      Weight      Internal     KVCache      Total\n");
  printf("  -------- ------------ ------------ ------------ ------------ ------------\n");
  for (uint32_t i = 0; i < core_num; i++) {
    const rknn3_allocation_info* info = &alloc_infos[i];
    double command_mb  = bytes_to_mb(info->command_mem.size);
    double weight_mb   = bytes_to_mb(info->weight_mem.size);
    double internal_mb = bytes_to_mb(info->internal_mem.size);
    double kvcache_mb  = bytes_to_mb(info->kvcache_mem.size);
    double total_mb    = command_mb + weight_mb + internal_mb + kvcache_mb;

    printf("  %-8d %10.2f %12.2f %12.2f %12.2f %12.2f\n", info->core_id, command_mb, weight_mb,
           internal_mb, kvcache_mb, total_mb);

    total_command += info->command_mem.size;
    total_weight += info->weight_mem.size;
    total_internal += info->internal_mem.size;
    total_kvcache += info->kvcache_mem.size;
  }
  printf("  -------- ------------ ------------ ------------ ------------ ------------\n");
  printf("  Total    %10.2f %12.2f %12.2f %12.2f %12.2f\n", bytes_to_mb(total_command),
         bytes_to_mb(total_weight), bytes_to_mb(total_internal), bytes_to_mb(total_kvcache),
         bytes_to_mb(total_command + total_weight + total_internal + total_kvcache));
  free(alloc_infos);
  return RKNN3_SUCCESS;
}

static void print_usage(const char* prog)
{
  printf("Usage: %s <model_path> [core_mask] [device_id] [key_path]\n", prog);
  printf("  model_path : RKNN model file path\n");
  printf("  core_mask  : optional, hex core mask, auto-generated if omitted\n");
  printf("  device_id  : optional, target device id from rknn3_find_devices()\n");
  printf("  key_path   : optional, RSA encrypted key envelope for encrypted models\n");
  printf("\n");
  printf("This demo loads model structure only (weight=NULL) and evaluates memory usage\n");
  printf("via rknn3_model_init() + rknn3_profile_mem(), without requiring a weight file.\n");
  printf("\n");
  printf("Example:\n");
  printf("  %s model.rknn\n", prog);
  printf("  %s model.rknn 0xff\n", prog);
  printf("  %s model.rknn 0xff rk1820-xxxx\n", prog);
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    print_usage(argv[0]);
    return -1;
  }

  const char* model_path = argv[1];
  const char* device_id_arg = (argc > 3) ? argv[3] : NULL;
  const char* key_path = (argc > 4) ? argv[4] : NULL;
  uint32_t core_mask = 0;

  rknn3_context ctx = 0;
  rknn3_devices devs;
  int ret = 0;

  printf("RKNN3 Model Info Demo\n");
  printf("model_path=%s\n", model_path);
  if (device_id_arg != NULL && strlen(device_id_arg) > 0) {
    printf("device_id=%s\n", device_id_arg);
  }
  if (key_path != NULL && strlen(key_path) > 0) {
    printf("key_path=%s\n", key_path);
  }

  memset(&devs, 0, sizeof(devs));
  ret = rknn3_find_devices(&devs);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_find_devices failed! ret=%d\n", ret);
    return -1;
  }
  if (devs.n_devices == 0) {
    printf("No RK182X devices found\n");
    return -1;
  }

  print_device_list(&devs);

  const char* selected_device_id = find_device_id(&devs, device_id_arg);
  if (selected_device_id == NULL) {
    printf("Device id '%s' not found\n", device_id_arg);
    return -1;
  }
  printf("Using device id=%s\n", selected_device_id);

  rknn3_init_extend init_extend = {0};
  init_extend.device_id = (char*)selected_device_id;
  ret = rknn3_init(&ctx, &init_extend);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_init failed! ret=%d\n", ret);
    return -1;
  }

  if (key_path != NULL && strlen(key_path) > 0) {
    ret = rknn3_set_decrypt_key_from_path(ctx, key_path);
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_set_decrypt_key_from_path failed! ret=%d\n", ret);
      rknn3_destroy(ctx);
      return -1;
    }
  }

  ret = rknn3_load_model_from_path(ctx, model_path, NULL);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_load_model_from_path (weight=NULL) failed! ret=%d\n", ret);
    rknn3_destroy(ctx);
    return -1;
  }
  printf("rknn3_load_model_from_path success (streaming mode, no weight file required)\n");

  uint32_t core_num = 0;
  ret = rknn3_query(ctx, RKNN3_QUERY_CORE_NUMBER, &core_num, sizeof(core_num));
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_query core number failed! ret=%d\n", ret);
    rknn3_destroy(ctx);
    return -1;
  }

  if (argc > 2 && strlen(argv[2]) > 0) {
    core_mask = (uint32_t)strtoul(argv[2], NULL, 16);
    if (get_core_num(core_mask) != (int)core_num) {
      printf("Error: core_mask 0x%x does not match core number %u\n", core_mask, core_num);
      rknn3_destroy(ctx);
      return -1;
    }
  } else {
    for (uint32_t i = 0; i < core_num; i++) {
      core_mask |= (1U << i);
    }
    printf("Auto-generated core_mask: 0x%x for %u cores\n", core_mask, core_num);
  }

  rknn3_config config;
  memset(&config, 0, sizeof(config));
  config.run_core_mask = core_mask;

  ret = rknn3_model_init(ctx, &config);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_model_init failed! ret=%d\n", ret);
    rknn3_destroy(ctx);
    return -1;
  }
  printf("rknn3_model_init success\n");

  print_sdk_version(ctx);
  print_model_basic_info(ctx);
  print_custom_string(ctx);
  print_tensor_attrs(ctx);
  print_llm_config(ctx);
  print_device_mem_info(ctx);

  if (print_kvcache_page_info(ctx, core_num) != RKNN3_SUCCESS) {
    rknn3_destroy(ctx);
    return -1;
  }

  ret = print_allocation_info(ctx, core_num);
  if (ret != RKNN3_SUCCESS) {
    rknn3_destroy(ctx);
    return -1;
  }

  // printf("\n======================== rknn3_profile_mem ========================\n");
  // ret = rknn3_profile_mem(ctx);
  // if (ret != RKNN3_SUCCESS) {
  //   printf("rknn3_profile_mem failed! ret=%d\n", ret);
  //   rknn3_destroy(ctx);
  //   return -1;
  // }

  rknn3_destroy(ctx);
  return 0;
}
