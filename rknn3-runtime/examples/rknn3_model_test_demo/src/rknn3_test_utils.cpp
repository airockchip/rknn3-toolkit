#include "rknn3_test_utils.h"

void cleanup_kvcache_mems(rknn3_context ctx, std::vector<rknn3_tensor_mem*>* mems)
{
  if (!mems) {
    return;
  }
  for (rknn3_tensor_mem* mem : *mems) {
    if (mem) {
      rknn3_destroy_mem(ctx, mem);
    }
  }
  mems->clear();
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
