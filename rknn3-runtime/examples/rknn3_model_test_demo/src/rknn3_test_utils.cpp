#include "rknn3_test_utils.h"
#include "cnpy.h"

#include <algorithm>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <sys/time.h>

static void destroy_kvcache_mems(rknn3_context ctx, std::vector<rknn3_tensor_mem*>* mems)
{
  if (!mems) return;
  for (rknn3_tensor_mem* mem : *mems) {
    if (mem) rknn3_destroy_mem(ctx, mem);
  }
  mems->clear();
}

static void dump_kvcache_page_info(const rknn3_kvcache_page_info& info)
{
  printf("KV cache page info: core_id=%d, max_request_num=%u, n_cache_types=%u\n",
         info.core_id, info.max_request_num, info.n_cache_types);
  for (uint32_t i = 0; i < info.n_cache_types && i < RKNN3_MAX_CACHE_PAGE_TYPES; ++i) {
    printf("  cache[%u]: type=%d, page_size=%llu, max_page_num=%u\n", i,
           info.caches[i].cache_type, (unsigned long long)info.caches[i].page_size,
           info.caches[i].max_page_num);
  }
}

static int rknn3_paged_cache_page_count(const rknn3_cache_page_attr* attr,
                                               const rknn3_attention_config* attention_config,
                                               uint64_t tokens, uint32_t* pages)
{
  if (!attr || !pages || !attr->page_size || !attr->max_page_num)
    return RKNN3_ERR_ARGUMENT_INVALID;
  if (attr->cache_type != RKNN3_PAGE_CACHE_FULL_ATTENTION) {
    *pages = attr->max_page_num;
    return RKNN3_SUCCESS;
  }
  if (!attention_config || attention_config->attention_type != RKNN3_ATTENTION_TYPE_FULL_ATTENTION ||
      !tokens || !attention_config->tokens_per_page)
    return RKNN3_ERR_ARGUMENT_INVALID;
  uint64_t count = tokens / attention_config->tokens_per_page +
                   (tokens % attention_config->tokens_per_page != 0);
  if (count > attr->max_page_num)
    return RKNN3_ERR_ARGUMENT_INVALID;
  *pages = (uint32_t)count;
  return RKNN3_SUCCESS;
}

void cleanup_kvcache_mems(rknn3_context ctx, std::vector<rknn3_tensor_mem*>* mems)
{
  if (!mems) {
    return;
  }
  std::vector<int32_t> cores;
  std::vector<rknn3_kvcache_page_binding> unbinds;
  for (rknn3_tensor_mem* mem : *mems) {
    if (!mem || std::find(cores.begin(), cores.end(), mem->core_id) != cores.end()) continue;
    rknn3_kvcache_page_info info = {};
    info.core_id = mem->core_id;
    int ret = rknn3_query(ctx, RKNN3_QUERY_KVCACHE_PAGE_INFO, &info, sizeof(info));
    if (ret != RKNN3_SUCCESS) {
      printf("RKNN3_QUERY_KVCACHE_PAGE_INFO during cleanup failed, ret=%d; "
             "skipping unbind for core %d, destroying its pages anyway\n",
             ret, mem->core_id);
      continue;
    }
    cores.push_back(mem->core_id);
    for (uint32_t request = 0; request < info.max_request_num; ++request) {
      for (uint32_t type = 0; type < info.n_cache_types; ++type) {
        unbinds.push_back({info.core_id, info.caches[type].cache_type, request, 0, nullptr});
      }
    }
  }
  if (!unbinds.empty()) {
    int ret = rknn3_set_kvcache_pages(ctx, unbinds.data(), (uint32_t)unbinds.size());
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_set_kvcache_pages unbind failed, ret=%d; destroying pages anyway\n", ret);
    }
  }
  destroy_kvcache_mems(ctx, mems);
}

int setup_kvcache_mems(rknn3_context ctx, std::vector<rknn3_tensor_mem*>* kvcache_mems)
{
  if (!ctx || !kvcache_mems) {
    return RKNN3_ERR_ARGUMENT_INVALID;
  }

  kvcache_mems->clear();

  uint32_t core_num = 0;
  int ret = rknn3_query(ctx, RKNN3_QUERY_CORE_NUMBER, &core_num, sizeof(core_num));
  if (ret != RKNN3_SUCCESS) {
    printf("RKNN3_QUERY_CORE_NUMBER failed, ret=%d\n", ret);
    return ret;
  }
  if (core_num == 0) {
    return RKNN3_SUCCESS;
  }

  std::vector<rknn3_allocation_info> alloc_infos((size_t)core_num);
  ret = rknn3_query(ctx, RKNN3_QUERY_ALLOCATION_INFO, alloc_infos.data(), sizeof(rknn3_allocation_info) * (size_t)core_num);
  if (ret != RKNN3_SUCCESS) {
    printf("RKNN3_QUERY_ALLOCATION_INFO failed, ret=%d\n", ret);
    return ret;
  }

  rknn3_kvcache_page_info first_page_info = {};
  first_page_info.core_id = alloc_infos[0].core_id;
  ret = rknn3_query(ctx, RKNN3_QUERY_KVCACHE_PAGE_INFO, &first_page_info, sizeof(first_page_info));
  if (ret != RKNN3_SUCCESS) {
    printf("RKNN3_QUERY_KVCACHE_PAGE_INFO failed, ret=%d\n", ret);
    return ret;
  }
  dump_kvcache_page_info(first_page_info);
  if (first_page_info.n_cache_types != 0) {
    rknn3_llm_config llm_config = {};
    ret = rknn3_query(ctx, RKNN3_QUERY_LLM_CONFIG, &llm_config, sizeof(llm_config));
    if (ret != RKNN3_SUCCESS) return ret;
    std::vector<rknn3_attention_config> attention_configs(llm_config.n_attention_configs);
    if (attention_configs.empty()) {
      printf("Invalid model: paged KV cache is enabled, but no attention config was found\n");
      return RKNN3_ERR_MODEL_INVALID;
    }
    ret = rknn3_query(ctx, RKNN3_QUERY_ATTENTION_CONFIG, attention_configs.data(),
                      attention_configs.size() * sizeof(attention_configs[0]));
    if (ret != RKNN3_SUCCESS) return ret;
    const rknn3_attention_config* page_config = nullptr;
    for (const auto& config : attention_configs) {
      if (config.attention_type == RKNN3_ATTENTION_TYPE_FULL_ATTENTION) {
        page_config = &config;
        break;
      }
    }
    if (!page_config) {
      printf("Invalid model: paged KV cache requires a FULL_ATTENTION config; found attention types:");
      for (const auto& config : attention_configs) {
        printf(" %d", (int)config.attention_type);
      }
      printf(" (SLIDING_ATTENTION=%d, LINEAR_ATTENTION=%d)\n",
             (int)RKNN3_ATTENTION_TYPE_SLIDING_ATTENTION,
             (int)RKNN3_ATTENTION_TYPE_LINEAR_ATTENTION);
      return RKNN3_ERR_MODEL_INVALID;
    }
    printf("KV cache page config: tokens_per_page=%u, max_ctx_len=%u, residual_depth=%u, rollback_tokens_align=%u\n",
           page_config->tokens_per_page, page_config->kvcache_buffer_len, page_config->kvcache_residual_depth,
           page_config->rollback_tokens_align);
    std::vector<rknn3_kvcache_page_binding> bindings;
    std::vector<std::vector<const rknn3_tensor_mem*> > pages;
    for (const rknn3_allocation_info& allocation : alloc_infos) {
      rknn3_kvcache_page_info info = {};
      info.core_id = allocation.core_id;
      ret = rknn3_query(ctx, RKNN3_QUERY_KVCACHE_PAGE_INFO, &info, sizeof(info));
      if (ret != RKNN3_SUCCESS || info.n_cache_types == 0) {
        destroy_kvcache_mems(ctx, kvcache_mems);
        return ret != RKNN3_SUCCESS ? ret : RKNN3_ERR_MODEL_INVALID;
      }
      dump_kvcache_page_info(info);
      for (uint32_t request = 0; request < info.max_request_num; ++request) {
        for (uint32_t type = 0; type < info.n_cache_types; ++type) {
          uint32_t page_count = 0;
          ret = rknn3_paged_cache_page_count(&info.caches[type], page_config,
                                              llm_config.max_ctx_len, &page_count);
          if (ret != RKNN3_SUCCESS) {
            printf("rknn3_paged_cache_page_count failed, ret = %d\n", ret);
            destroy_kvcache_mems(ctx, kvcache_mems);
            return ret;
          }
          pages.emplace_back();
          pages.back().reserve(page_count);
          for (uint32_t page = 0; page < page_count; ++page) {
            rknn3_tensor_mem* mem = rknn3_create_mem(ctx, info.caches[type].page_size,
                                                     info.core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
            if (!mem || !mem->virt_addr) {
              if (mem) rknn3_destroy_mem(ctx, mem);
              destroy_kvcache_mems(ctx, kvcache_mems);
              return RKNN3_ERR_OUT_OF_MEMORY;
            }
            kvcache_mems->push_back(mem);
            memset(mem->virt_addr, 0, mem->size);
            ret = rknn3_mem_sync(ctx, mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
            if (ret != RKNN3_SUCCESS) {
              destroy_kvcache_mems(ctx, kvcache_mems);
              return ret;
            }
            pages.back().push_back(mem);
          }
          bindings.push_back({info.core_id, info.caches[type].cache_type, request, page_count, nullptr});
          printf("KV cache binding: core_id=%d, request=%u, cache_type=%d, n_pages=%u, page_size=%llu\n",
                 info.core_id, request, info.caches[type].cache_type, page_count,
                 (unsigned long long)info.caches[type].page_size);
        }
      }
    }
    for (size_t i = 0; i < bindings.size(); ++i) bindings[i].pages = pages[i].data();
    ret = rknn3_set_kvcache_pages(ctx, bindings.data(), (uint32_t)bindings.size());
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_set_kvcache_pages failed, ret=%d\n", ret);
      destroy_kvcache_mems(ctx, kvcache_mems);
      return ret;
    }
    printf("rknn3_set_kvcache_pages success, set %zu page memories\n", kvcache_mems->size());
    return RKNN3_SUCCESS;
  }

  std::vector<int> core_indices;
  for (uint32_t i = 0; i < core_num; ++i) {
    uint64_t kvcache_size = alloc_infos[i].kvcache_mem.size;
    int      core_id      = alloc_infos[i].core_id;
    printf("Core %d allocation info: kvcache_size=%llu\n", core_id, (unsigned long long)kvcache_size);
    if (kvcache_size == 0) {
      continue;
    }

    rknn3_tensor_mem* mem = rknn3_create_mem(ctx, kvcache_size, core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
    if (!mem) {
      printf("rknn3_create_mem for kvcache failed, core=%d, size=%llu\n", core_id, (unsigned long long)kvcache_size);
      cleanup_kvcache_mems(ctx, kvcache_mems);
      return RKNN3_ERR_OUT_OF_MEMORY;
    }

    kvcache_mems->push_back(mem);
    core_indices.push_back(core_id);
    printf("Created KV cache memory: core_id=%d, size=%llu, virt=0x%p, phys=0x%lx\n", core_id,
           (unsigned long long)mem->size, mem->virt_addr, mem->phys_addr);
  }

  if (kvcache_mems->empty()) {
    printf("No KV cache memory is required by this model\n");
    return RKNN3_SUCCESS;
  }

  ret = rknn3_set_kvcache_mem(ctx, kvcache_mems->data(), core_indices.data(), (int)core_indices.size());
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_set_kvcache_mem failed, ret=%d\n", ret);
    cleanup_kvcache_mems(ctx, kvcache_mems);
    return ret;
  }

  printf("rknn3_set_kvcache_mem success, set %zu KV cache memories\n", kvcache_mems->size());
  return RKNN3_SUCCESS;
}
